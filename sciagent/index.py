"""Hybrid retrieval: BM25 (lexical) + dense (semantic), fused with RRF.

Why hybrid: scientific queries mix exact technical tokens ("FNO", "Re=100",
"Allen-Cahn") where BM25 shines, with paraphrased concepts ("why do PINNs
fail to train") where embeddings shine. Reciprocal Rank Fusion is
score-scale-free, so no calibration between the two retrievers is needed.

Dense backend: sentence-transformers (bge-small) if installed, otherwise a
TF-IDF + truncated SVD (LSA) embedding - weaker, but dependency-free and
deterministic, which keeps CI/offline evaluation reproducible.

Usage:
    python -m sciagent.index          # builds data/index.pkl from chunks.jsonl
"""
from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter, defaultdict

import numpy as np

from . import config

STOP = set("""a an the of and or in on for to with by from as at is are was were be been
this that these those it its we our their they which can using use used via than into
not no do does how what why when where who whom whose vs versus about over under between
paper study show shows shown results result approach method methods""".split())
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    toks = []
    for t in TOKEN_RE.findall(text.lower()):
        if t in STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # crude plural folding: networks -> network
        toks.append(t)
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if self.N else 0.0
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)

    def scores(self, q: list[str]) -> np.ndarray:
        s = np.zeros(self.N)
        for t in set(q):
            if t not in self.idf:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                s[i] += self.idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class DenseEncoder:
    """sentence-transformers if available, else TF-IDF+SVD (LSA)."""

    def __init__(self, texts: list[str], prefer_st: bool = True):
        self.kind = "lsa"
        self.model = None
        if prefer_st:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(config.EMBED_MODEL)
                self.kind = f"st:{config.EMBED_MODEL}"
            except Exception:
                self.model = None
        if self.model is None:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                       sublinear_tf=True, min_df=1)
            X = self.vec.fit_transform(texts)
            k = max(2, min(256, X.shape[1] - 1, X.shape[0] - 1))
            self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        self.emb = self.encode(texts, is_query=False)

    def encode(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        if self.model is not None:
            if is_query and "bge" in config.EMBED_MODEL:
                texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
            return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=32))
        Z = self.svd.transform(self.vec.transform(texts))
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def __getstate__(self):  # don't pickle the torch model
        d = self.__dict__.copy()
        d["model"] = None
        return d


class HybridIndex:
    def __init__(self, chunks: list[dict], prefer_st: bool = True):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        texts = [self._doc_text(c) for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.dense = DenseEncoder(texts, prefer_st=prefer_st)
        self.reranker = None

    @staticmethod
    def _doc_text(c: dict) -> str:
        # title is prepended: gives every chunk paper-level context
        return f"{c['title']}. {c.get('section', '')}. {c['text']}"

    # ------------------------------------------------------------ search
    def _ranked(self, q: str, mode: str, mask: np.ndarray) -> list[tuple[int, float]]:
        if mode == "bm25":
            s = self.bm25.scores(tokenize(q))
        elif mode == "dense":
            s = self.dense.emb @ self.dense.encode([q])[0]
        else:
            raise ValueError(mode)
        s = np.where(mask, s, -np.inf)
        order = np.argsort(-s)
        return [(int(i), float(s[i])) for i in order if np.isfinite(s[i]) and (mode != "bm25" or s[i] > 0)]

    def search(self, query: str, k: int = 8, mode: str = "hybrid", chunk_type: str | None = None,
               paper_id: str | None = None, max_per_paper: int | None = config.MAX_CHUNKS_PER_PAPER,
               rrf_k: int = 60) -> list[dict]:
        mask = np.ones(len(self.chunks), bool)
        if chunk_type:
            types = set(chunk_type.split(","))
            mask &= np.array([c["type"] in types for c in self.chunks])
        if paper_id:
            mask &= np.array([c["paper_id"] == paper_id for c in self.chunks])

        if mode == "hybrid":
            fused: dict[int, float] = defaultdict(float)
            raw: dict[int, dict] = defaultdict(dict)
            for m in ("bm25", "dense"):
                for r, (i, s) in enumerate(self._ranked(query, m, mask)[:200]):
                    fused[i] += 1.0 / (rrf_k + r + 1)
                    raw[i][m] = s
            ranked = sorted(fused.items(), key=lambda x: -x[1])
        else:
            ranked = self._ranked(query, mode, mask)
            raw = {i: {mode: s} for i, s in ranked}

        if self.reranker is not None and ranked:
            cand = ranked[:30]
            pairs = [(query, self._doc_text(self.chunks[i])) for i, _ in cand]
            rs = self.reranker.predict(pairs)
            ranked = sorted(((i, float(s)) for (i, _), s in zip(cand, rs)), key=lambda x: -x[1])

        out, per_paper = [], Counter()
        for i, s in ranked:
            c = self.chunks[i]
            if max_per_paper and per_paper[c["paper_id"]] >= max_per_paper:
                continue
            per_paper[c["paper_id"]] += 1
            out.append({**c, "score": round(s, 5), "bm25": round(raw[i].get("bm25", 0.0), 3),
                        "dense": round(raw[i].get("dense", 0.0), 3)})
            if len(out) >= k:
                break
        return out

    def get(self, chunk_id: str) -> dict | None:
        i = self.by_id.get(chunk_id)
        return self.chunks[i] if i is not None else None

    def neighbors(self, chunk_id: str, window: int = 1) -> list[dict]:
        c = self.get(chunk_id)
        if not c or c["type"] != "text":
            return [c] if c else []
        n = int(chunk_id.split("#t")[1])
        ids = [f"{c['paper_id']}#t{j}" for j in range(n - window, n + window + 1)]
        return [self.get(x) for x in ids if self.get(x)]

    def papers(self) -> list[dict]:
        seen = {}
        for c in self.chunks:
            p = seen.setdefault(c["paper_id"], {"paper_id": c["paper_id"], "title": c["title"],
                                                "year": c.get("year"), "n_chunks": 0,
                                                "n_figures": 0, "n_tables": 0})
            p["n_chunks"] += 1
            p["n_figures"] += c["type"] == "figure"
            p["n_tables"] += c["type"] == "table"
        return list(seen.values())

    # ------------------------------------------------------------ io
    def save(self, path=config.INDEX_FILE):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=config.INDEX_FILE) -> "HybridIndex":
        with open(path, "rb") as f:
            idx: HybridIndex = _IndexUnpickler(f).load()
        if idx.dense.kind.startswith("st:"):
            from sentence_transformers import SentenceTransformer
            idx.dense.model = SentenceTransformer(idx.dense.kind[3:])
        if config.RERANK_MODEL:
            try:
                from sentence_transformers import CrossEncoder
                idx.reranker = CrossEncoder(config.RERANK_MODEL)
            except Exception:
                pass
        return idx


class _IndexUnpickler(pickle.Unpickler):
    """Also loads indexes pickled from `python -m sciagent.index` (classes saved under __main__)."""
    def find_class(self, module, name):
        if module == "__main__" and name in ("HybridIndex", "BM25", "DenseEncoder"):
            module = "sciagent.index"
        return super().find_class(module, name)


def load_chunks(path=config.CHUNKS_FILE) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsa", action="store_true", help="force LSA dense encoder (no torch)")
    a = ap.parse_args(argv)
    if not config.CHUNKS_FILE.exists():
        raise SystemExit(f"{config.CHUNKS_FILE} not found - run `python -m sciagent.ingest` first "
                         "(which needs data/papers.jsonl from `python -m sciagent.collect`).")
    chunks = [c for c in load_chunks() if c["text"].strip()]
    # Build with the classes from the *importable* module, not __main__, so the pickle
    # refers to sciagent.index.BM25 etc. and can be loaded from any entry point.
    from sciagent import index as _mod
    idx = _mod.HybridIndex(chunks, prefer_st=not a.lsa)
    idx.save()
    print(f"indexed {len(chunks)} chunks from {len(idx.papers())} papers "
          f"(dense={idx.dense.kind}) -> {config.INDEX_FILE}")


if __name__ == "__main__":
    main()
