"""Agent tools. Each tool = JSON schema (shown to the LLM) + a Python function.

The ToolBox also records every chunk the agent has *seen*, which the citation
verifier later uses: an answer may only cite evidence that was actually
retrieved in this episode (blocks fabricated citations).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import config

MAX_TOOL_CHARS = 7000


def _short(c: dict, n: int = 900) -> dict:
    d = {"chunk_id": c["chunk_id"], "paper": f"{c['title']} ({c.get('year')})",
         "type": c["type"], "section": c.get("section"), "page": c.get("page"),
         "text": c["text"][:n] + ("..." if len(c["text"]) > n else "")}
    if "score" in c:
        d["score"] = c["score"]
    if c.get("image_path"):
        d["has_image"] = True
    return d


NUM_CLAIM_RE = re.compile(
    r"([^.]*?(?:\d+(?:\.\d+)?\s*(?:x|×|-fold|fold|times|%|orders? of magnitude)|"
    r"orders? of magnitude|speed-?up|faster|slower|outperform\w*|fail\w*)[^.]*\.)", re.I)


class ToolBox:
    def __init__(self, index, llm=None, allow_network: bool = True):
        self.index = index
        self.llm = llm
        self.allow_network = allow_network
        self.seen: dict[str, dict] = {}
        self.calls: list[dict] = []

    # ------------------------------------------------------------- registry
    @property
    def specs(self) -> list[dict]:
        return [
            {"name": "search_papers",
             "description": "Hybrid (BM25+semantic) search over the indexed corpus of full-text papers "
                            "on ML for CFD/FEM. Returns evidence chunks with chunk_id for citation. "
                            "Use specific technical terms; call several times with reformulated queries "
                            "for multi-part questions. Results are capped per paper for diversity.",
             "parameters": {"type": "object", "properties": {
                 "query": {"type": "string"},
                 "k": {"type": "integer", "default": 6},
                 "chunk_type": {"type": "string", "description": "optional filter: text, abstract, figure, table (comma-separated)"},
                 "paper_id": {"type": "string", "description": "optional: restrict to one paper"}},
                 "required": ["query"]}},
            {"name": "read_context",
             "description": "Read a chunk plus its neighbouring chunks (more context around a hit).",
             "parameters": {"type": "object", "properties": {
                 "chunk_id": {"type": "string"}, "window": {"type": "integer", "default": 1}},
                 "required": ["chunk_id"]}},
            {"name": "list_papers",
             "description": "List all papers in the corpus (id, title, year, #figures, #tables). "
                            "Use to check whether a topic/paper is covered at all.",
             "parameters": {"type": "object", "properties": {}}},
            {"name": "search_arxiv",
             "description": "Live arXiv search for papers NOT necessarily in the corpus (recent work, "
                            "coverage gaps). Returns titles+abstracts only; evidence from it is "
                            "abstract-level and must be labelled as such.",
             "parameters": {"type": "object", "properties": {
                 "query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}},
                 "required": ["query"]}},
            {"name": "find_figures",
             "description": "Search figure and table captions (and extracted table cells). Returns "
                            "figure/table ids; figures with has_image=true can be inspected visually.",
             "parameters": {"type": "object", "properties": {
                 "query": {"type": "string"}, "paper_id": {"type": "string"},
                 "k": {"type": "integer", "default": 5}}, "required": ["query"]}},
            {"name": "inspect_figure",
             "description": "Look at a figure image with a vision model and answer a question about it "
                            "(trends, axes, which curve is lower, approximate values).",
             "parameters": {"type": "object", "properties": {
                 "figure_id": {"type": "string"}, "question": {"type": "string"}},
                 "required": ["figure_id", "question"]}},
            {"name": "extract_numeric_claims",
             "description": "Collect quantitative claims (speed-ups, errors in %, 'orders of magnitude', "
                            "outperform/fail statements) about a topic, grouped by paper. Use for "
                            "comparing or reconciling claims across papers and detecting conflicts.",
             "parameters": {"type": "object", "properties": {
                 "topic": {"type": "string"}, "k": {"type": "integer", "default": 12}},
                 "required": ["topic"]}},
            {"name": "run_python",
             "description": "Run Python (numpy, scipy, pandas, sklearn, matplotlib available) for data "
                            "analysis or calculations. Print results. Files saved to ./ are reported. "
                            "`from sciagent.index import load_chunks` gives the corpus chunks.",
             "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                            "required": ["code"]}},
            {"name": "run_surrogate_experiment",
             "description": "Run our own ML experiment: 1D viscous Burgers equation; compares a "
                            "pseudo-spectral reference solver, a cheap coarse classical solver, and "
                            "POD+ridge / POD+MLP surrogates (5-fold CV) on accuracy (in-distribution and "
                            "out-of-distribution viscosity), cost per query and break-even point. "
                            "Takes ~30-60 s. Use to test claims about surrogates vs solvers empirically.",
             "parameters": {"type": "object", "properties": {
                 "n_train": {"type": "integer", "default": 400},
                 "nu_train_min": {"type": "number", "default": 0.05},
                 "nu_train_max": {"type": "number", "default": 0.3},
                 "nu_ood_min": {"type": "number", "default": 0.015},
                 "nu_ood_max": {"type": "number", "default": 0.04},
                 "coarse_modes": {"type": "integer", "default": 32},
                 "use_tensorflow": {"type": "boolean", "default": False,
                                    "description": "also train a TensorFlow/Keras MLP surrogate"}}}},
        ]

    def call(self, name: str, args: dict) -> str:
        t0 = time.perf_counter()
        fn = getattr(self, f"t_{name}", None)
        try:
            out = fn(**args) if fn else {"error": f"unknown tool {name}"}
        except TypeError as e:
            out = {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # tools must never crash the agent loop
            out = {"error": f"{type(e).__name__}: {e}"}
        s = json.dumps(out, ensure_ascii=False, default=str)
        if len(s) > MAX_TOOL_CHARS:
            s = s[:MAX_TOOL_CHARS] + '... [truncated]"'
        self.calls.append({"tool": name, "args": args, "seconds": round(time.perf_counter() - t0, 2),
                           "result_preview": s[:300]})
        return s

    def _remember(self, chunks):
        for c in chunks:
            self.seen[c["chunk_id"]] = c

    # ------------------------------------------------------------- tools
    def t_search_papers(self, query, k=6, chunk_type=None, paper_id=None):
        res = self.index.search(query, k=int(k), chunk_type=chunk_type, paper_id=paper_id)
        self._remember(res)
        if not res:
            return {"results": [], "note": "no matching evidence in corpus"}
        return {"results": [_short(c) for c in res]}

    def t_read_context(self, chunk_id, window=1):
        res = self.index.neighbors(chunk_id, int(window))
        if not res:
            return {"error": f"unknown chunk_id {chunk_id}"}
        self._remember(res)
        return {"chunks": [_short(c, 1500) for c in res]}

    def t_list_papers(self):
        return {"n_papers": len(self.index.papers()), "papers": self.index.papers()}

    def t_search_arxiv(self, query, max_results=5):
        if not self.allow_network:
            return {"error": "network disabled in this run"}
        from .collect import arxiv_search
        try:
            res = arxiv_search(query, int(max_results))
        except Exception as e:
            return {"error": f"arXiv unavailable: {e}"}
        in_corpus = {p["paper_id"] for p in self.index.papers()}
        out = []
        for p in res:
            cid = f"arxiv:{p['paper_id']}"
            self.seen[cid] = {"chunk_id": cid, "paper_id": p["paper_id"], "title": p["title"],
                              "year": p["year"], "type": "external_abstract", "text": p["abstract"]}
            out.append({"chunk_id": cid, "title": p["title"], "year": p["year"],
                        "in_corpus": p["paper_id"] in in_corpus, "abstract": p["abstract"][:700]})
        return {"results": out, "note": "abstract-level evidence only"}

    def t_find_figures(self, query, paper_id=None, k=5):
        res = self.index.search(query, k=int(k), chunk_type="figure,table", paper_id=paper_id,
                                max_per_paper=2)
        self._remember(res)
        return {"results": [_short(c, 600) for c in res]}

    def t_inspect_figure(self, figure_id, question):
        c = self.index.get(figure_id)
        if not c or c["type"] != "figure":
            return {"error": f"{figure_id} is not a figure id (use find_figures)"}
        self._remember([c])
        img = c.get("image_path")
        if not img or not Path(img).exists():
            return {"figure_id": figure_id, "caption": c["text"],
                    "note": "no rendered image available; only the caption can be used"}
        if self.llm is None or not hasattr(self.llm, "vision"):
            return {"figure_id": figure_id, "caption": c["text"], "image_path": img,
                    "note": "no vision model available in offline mode; caption only"}
        prompt = (f"This is a figure from the paper '{c['title']}'. Caption: {c['text']}\n\n"
                  f"Question: {question}\nDescribe only what is visible (axes, legend, trends, "
                  "approximate values). Say explicitly if the figure cannot answer the question.")
        return {"figure_id": figure_id, "caption": c["text"], "vision_answer": self.llm.vision(img, prompt)}

    def t_extract_numeric_claims(self, topic, k=12):
        res = self.index.search(topic, k=int(k), chunk_type="text,abstract,table", max_per_paper=3)
        self._remember(res)
        by_paper: dict[str, list] = {}
        for c in res:
            for m in NUM_CLAIM_RE.findall(c["text"]):
                s = " ".join(m.split())
                if 30 < len(s) < 400:
                    by_paper.setdefault(f"{c['title']} ({c.get('year')})", []).append(
                        {"chunk_id": c["chunk_id"], "claim": s})
        return {"claims_by_paper": {p: v[:4] for p, v in by_paper.items()},
                "note": "claims are extracted verbatim; check conditions (problem, resolution, "
                        "baseline) before comparing numbers across papers"}

    def t_run_python(self, code, timeout=60):
        work = Path(tempfile.mkdtemp(prefix="sciagent_py_"))
        before = set()
        env = {**os.environ, "PYTHONPATH": str(config.ROOT), "MPLBACKEND": "Agg"}
        try:
            r = subprocess.run([sys.executable, "-c", code], cwd=work, capture_output=True,
                               text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout}s"}
        files = [str(p) for p in work.iterdir() if p.name not in before]
        return {"returncode": r.returncode, "stdout": r.stdout[-3000:], "stderr": r.stderr[-1500:],
                "files": files}

    def t_run_surrogate_experiment(self, n_train=400, nu_train_min=0.05, nu_train_max=0.3,
                                   nu_ood_min=0.015, nu_ood_max=0.04, coarse_modes=32,
                                   use_tensorflow=None):
        from .ml.burgers import format_table, run_experiment
        config.ensure_dirs()
        n_train = int(min(max(n_train, 50), 1500))
        plot = str(config.RESULTS_DIR / f"burgers_n{n_train}.png")
        res = run_experiment(n_train=n_train, nu_train=(nu_train_min, nu_train_max),
                             nu_ood=(nu_ood_min, nu_ood_max), coarse_modes=int(coarse_modes),
                             plot_path=plot, use_tf=use_tensorflow, cache=str(config.DATA_DIR / f"burgers_cache_{n_train}.npz"))
        cid = f"experiment:burgers_n{n_train}"
        self.seen[cid] = {"chunk_id": cid, "paper_id": "own-experiment", "title": "Own Burgers experiment",
                          "type": "experiment", "text": res["summary"]}
        return {"chunk_id": cid, "table": format_table(res), "summary": res["summary"], "plot": plot}
