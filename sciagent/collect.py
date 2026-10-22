"""Automated paper collection from arXiv.

Pipeline:
  1. run every topic query against the arXiv API (+ fetch seed IDs)
  2. de-duplicate (arXiv base id, normalised title)
  3. score each candidate's relevance to the topic description (TF-IDF cosine
     of title+abstract) and keep the top-N above a threshold
  4. download PDFs (polite rate limiting, retries, resumable)

Usage:
    python -m sciagent.collect --max-per-query 25 --keep 40
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from . import config

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
UA = "sciagent-research-prototype/0.1 (interview task)"


# ---------------------------------------------------------------- arXiv API
def _get(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network errors, 503 rate limit ...
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {last}")


def _parse_feed(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        id_url = e.findtext(f"{ATOM}id", "")
        m = re.search(r"abs/([^v]+)(v\d+)?$", id_url)
        if not m:
            continue
        title = " ".join((e.findtext(f"{ATOM}title") or "").split())
        if title.lower() == "error":
            continue
        pdf = next((l.get("href") for l in e.findall(f"{ATOM}link")
                    if l.get("title") == "pdf"), f"https://arxiv.org/pdf/{m.group(1)}")
        out.append({
            "paper_id": m.group(1),
            "title": title,
            "abstract": " ".join((e.findtext(f"{ATOM}summary") or "").split()),
            "authors": [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")],
            "year": int((e.findtext(f"{ATOM}published") or "0000")[:4]),
            "categories": [c.get("term") for c in e.findall(f"{ATOM}category")],
            "pdf_url": pdf,
            "source": "arxiv",
        })
    return out


def arxiv_search(query: str, max_results: int = 10, sort: str = "relevance") -> list[dict]:
    params = {"search_query": query, "start": 0, "max_results": max_results,
              "sortBy": sort, "sortOrder": "descending"}
    return _parse_feed(_get(f"{ARXIV_API}?{urllib.parse.urlencode(params)}"))


def arxiv_by_ids(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    return _parse_feed(_get(f"{ARXIV_API}?{urllib.parse.urlencode(params)}"))


# ---------------------------------------------------------------- curation
def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def dedupe(papers: list[dict]) -> list[dict]:
    seen_id, seen_title, out = set(), set(), []
    for p in papers:
        nt = _norm_title(p["title"])
        if p["paper_id"] in seen_id or nt in seen_title:
            continue
        seen_id.add(p["paper_id"]); seen_title.add(nt); out.append(p)
    return out


def relevance_scores(papers: list[dict], topic: str) -> list[float]:
    """TF-IDF cosine between topic description and title+abstract."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [topic] + [f"{p['title']} {p['title']} {p['abstract']}" for p in papers]
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
    return (X[1:] @ X[0].T).toarray().ravel().tolist()


def curate(papers: list[dict], keep: int, min_score: float, seeds: set[str]) -> list[dict]:
    scores = relevance_scores(papers, config.TOPIC_DESCRIPTION)
    for p, s in zip(papers, scores):
        p["relevance"] = round(float(s), 4)
        p["is_seed"] = p["paper_id"] in seeds
    # seeds always kept; others ranked by relevance
    ranked = sorted(papers, key=lambda p: (not p["is_seed"], -p["relevance"]))
    kept = [p for p in ranked if p["is_seed"] or p["relevance"] >= min_score]
    return kept[:max(keep, len(seeds))]


# ---------------------------------------------------------------- download
def download_pdf(p: dict, outdir: Path) -> Path | None:
    path = outdir / f"{p['paper_id'].replace('/', '_')}.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    try:
        data = _get(p["pdf_url"], timeout=60)
        if not data.startswith(b"%PDF"):
            raise ValueError("not a PDF")
        path.write_bytes(data)
        time.sleep(3)  # arXiv asks for >= 3 s between requests
        return path
    except Exception as e:
        print(f"  ! download failed {p['paper_id']}: {e}")
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-per-query", type=int, default=25)
    ap.add_argument("--keep", type=int, default=40, help="max papers kept after curation")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args(argv)
    config.ensure_dirs()

    cands: list[dict] = []
    print(f"Fetching {len(config.SEED_ARXIV_IDS)} seed papers ...")
    try:
        cands += arxiv_by_ids(config.SEED_ARXIV_IDS)
    except RuntimeError as e:
        print(f"  ! seed fetch failed: {e}")
    for q in config.ARXIV_QUERIES:
        time.sleep(3)
        try:
            res = arxiv_search(q, a.max_per_query)
            print(f"  {len(res):3d} hits  {q}")
            cands += res
        except RuntimeError as e:
            print(f"  ! query failed: {e}")
    if not cands:
        raise SystemExit("arXiv returned no results (network blocked or rate-limited). "
                         "Wait 1-2 minutes and re-run.")
    cands = dedupe(cands)
    kept = curate(cands, a.keep, a.min_score, set(config.SEED_ARXIV_IDS))
    print(f"{len(cands)} unique candidates -> kept {len(kept)}")

    if not a.no_download:
        for i, p in enumerate(kept, 1):
            print(f"[{i}/{len(kept)}] {p['paper_id']} {p['title'][:70]}")
            path = download_pdf(p, config.PDF_DIR)
            p["pdf_path"] = str(path) if path else None

    with open(config.PAPERS_FILE, "w") as f:
        for p in kept:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {config.PAPERS_FILE}")


if __name__ == "__main__":
    main()
