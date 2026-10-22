"""The research agent: an LLM tool-use loop + citation verification.

Two policies share the same tools, verifier and output format:
  * LLMAgent      - the model decides which tools to call, in which order, and
                    when it has enough evidence (true agentic behaviour).
  * OfflineAgent  - deterministic heuristic router + extractive answerer. Runs
                    with no API key; used as a baseline in the evaluation and
                    in CI, so retrieval/abstention logic is testable cheaply.

Usage:
    python -m sciagent.agent "Do PINNs beat the finite element method?"
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from . import config
from .index import HybridIndex, tokenize
from .tools import ToolBox

STATUSES = ("SUPPORTED", "CONFLICTING", "PARTIAL", "INSUFFICIENT")

SYSTEM_PROMPT = f"""You are a careful research assistant for the topic: {config.TOPIC_NAME}.
You answer questions using evidence from a corpus of scientific papers, via tools.

How to work
- Decide which tools you need. Typical: search_papers (possibly several reformulated queries),
  read_context for more detail, find_figures + inspect_figure for questions about plots/tables,
  extract_numeric_claims to compare quantitative claims across papers, run_python for calculations,
  run_surrogate_experiment to test a claim empirically, search_arxiv if the corpus lacks coverage.
- Do not call tools you do not need. Stop searching once the evidence answers the question.

Evidence rules (strict)
- Every factual sentence must cite the chunk(s) it comes from as [chunk_id], e.g. [2010.08895#t3].
  Only cite chunk_ids returned by tools in THIS conversation. Never invent ids.
- Distinguish what a paper claims from what you infer. Results from our own experiment must be
  cited with its experiment chunk_id and labelled as our own small-scale result.
- If papers disagree, do not average or pick a side silently: report each position, with the
  conditions under which it holds (problem, baseline, resolution, metric), and explain the likely
  reason for the disagreement.
- If the corpus does not contain the answer, say so plainly. A partial answer must say what is missing.
  Abstract-only evidence (search_arxiv) must be labelled as such.

Output format
Answer concisely (<= 250 words) with citations, then end with exactly these lines:
STATUS: <SUPPORTED | CONFLICTING | PARTIAL | INSUFFICIENT>
CONFIDENCE: <low | medium | high>
"""

CITE_RE = re.compile(r"\[([^\[\]]+?)\]")


@dataclass
class AgentResult:
    question: str
    answer: str
    status: str
    confidence: str
    citations: list[str]
    invalid_citations: list[str]
    trace: list[dict] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    policy: str = ""
    usage: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- verifier
def verify(answer: str, seen: dict) -> tuple[str, list[str], list[str], str, str]:
    """Strip citations that were never retrieved; parse STATUS/CONFIDENCE."""
    valid, invalid = [], []

    def repl(m):
        ids = [x.strip() for x in re.split(r"[;,]", m.group(1))]
        if not all(re.search(r"[#:]", x) for x in ids):  # not a citation (e.g. [0, 1])
            return m.group(0)
        keep = [x for x in ids if x in seen]
        for x in ids:
            (valid if x in seen else invalid).append(x)
        return f"[{'; '.join(keep)}]" if keep else "[citation removed: not in retrieved evidence]"

    cleaned = CITE_RE.sub(repl, answer)
    st = re.search(r"STATUS:\s*(\w+)", answer)
    status = st.group(1).upper() if st and st.group(1).upper() in STATUSES else "UNSPECIFIED"
    cf = re.search(r"CONFIDENCE:\s*(\w+)", answer)
    conf = cf.group(1).lower() if cf else "unspecified"
    if not valid and status not in ("INSUFFICIENT",):
        status, conf = "UNVERIFIED", "low"   # claims without any verifiable evidence
    if invalid and conf == "high":
        conf = "medium"
    return cleaned, list(dict.fromkeys(valid)), list(dict.fromkeys(invalid)), status, conf


# ---------------------------------------------------------------- LLM agent
class LLMAgent:
    policy = "llm"

    def __init__(self, index: HybridIndex, llm, max_steps: int = 8, allow_network: bool = True):
        self.index, self.llm, self.max_steps, self.allow_network = index, llm, max_steps, allow_network

    def run(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        tb = ToolBox(self.index, self.llm, self.allow_network)
        msgs = [{"role": "user", "content": question}]
        usage = {"in": 0, "out": 0}
        final, steps = "", 0
        for steps in range(1, self.max_steps + 1):
            last = steps == self.max_steps
            if last:
                msgs.append({"role": "user", "content": "Tool budget exhausted. Answer now with the "
                             "evidence gathered, following the output format."})
            r = self.llm.chat(SYSTEM_PROMPT, msgs, [] if last else tb.specs)
            for k in usage:
                usage[k] += r.usage.get(k, 0)
            msgs.append({"role": "assistant", "text": r.text, "tool_calls": r.tool_calls})
            if not r.tool_calls:
                final = r.text
                break
            for tc in r.tool_calls:
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"],
                             "content": tb.call(tc["name"], tc["args"])})
        answer, valid, invalid, status, conf = verify(final, tb.seen)
        return AgentResult(question, answer, status, conf, valid, invalid, tb.calls, steps,
                           round(time.perf_counter() - t0, 2), self.policy, usage)


# ---------------------------------------------------------------- offline agent
POS = re.compile(r"\b(outperform\w*|faster|speed-?ups?|accelerat\w*|orders? of magnitude|"
                 r"significant(ly)? (better|improv\w*)|accurate|superior|beats?)\b", re.I)
NEG = re.compile(r"\b(fail\w*|not (?:competitive|faster|better)|slower|weak baselines?|worse|"
                 r"cannot|overoptimis\w*|unable|inferior|do(?:es)? not (?:beat|outperform))\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")


class OfflineAgent:
    """Heuristic tool router + extractive, citation-backed answerer."""
    policy = "offline"

    def __init__(self, index: HybridIndex, coverage_threshold: float = 0.6,
                 run_experiments: bool = True, allow_network: bool = False):
        self.index = index
        self.thr = coverage_threshold
        self.run_experiments = run_experiments
        self.allow_network = allow_network

    def coverage(self, question: str, chunks: list[dict]) -> float:
        """IDF-weighted fraction of query terms present in the retrieved evidence."""
        q = set(tokenize(question))
        if not q:
            return 0.0
        idf = self.index.bm25.idf
        mx = max(idf.values()) if idf else 1.0
        text = set(tokenize(" ".join(c["title"] + " " + c["text"] for c in chunks)))
        w = {t: idf.get(t, mx) for t in q}
        return sum(v for t, v in w.items() if t in text) / sum(w.values())

    def route(self, q: str) -> list[str]:
        ql = q.lower()
        plan = []
        if re.search(r"\b(figure|fig\.|plot|curve|visuali[sz]\w*)\b|\bgraph\b(?! (?:neural )?network)", ql):
            plan.append("figures")
        if re.search(r"\btable\b", ql):
            plan.append("tables")
        if re.search(r"\b(our own|own experiment|empiric\w*|run an? |simulate|reproduce)\b", ql):
            plan.append("experiment")
        if re.search(r"\b(recent|latest|newest|2025|2026|arxiv)\b", ql):
            plan.append("arxiv")
        if re.search(r"\b(compare|comparison|conflict\w*|disagree\w*|outperform\w*|beat|faster|"
                     r"speed\w*|versus|vs\.?|better|overoptimis\w*|hype)\b", ql):
            plan.append("claims")
        return plan + ["search"]

    def run(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        tb = ToolBox(self.index, None, self.allow_network)
        plan = self.route(question)
        evidence: list[dict] = []
        extra_lines = []

        for step in plan:
            if step == "figures" or step == "tables":
                r = json.loads(tb.call("find_figures", {"query": question, "k": 3}))
                ids = [x["chunk_id"] for x in r.get("results", [])]
                evidence += [tb.seen[i] for i in ids]
                figs = [i for i in ids if tb.seen[i]["type"] == "figure"]
                if figs:
                    tb.call("inspect_figure", {"figure_id": figs[0], "question": question})
            elif step == "experiment" and self.run_experiments:
                r = json.loads(tb.call("run_surrogate_experiment", {}))
                if "chunk_id" in r:
                    evidence.append(tb.seen[r["chunk_id"]])
                    extra_lines.append(f"Our own experiment: {r['summary']} [{r['chunk_id']}]")
            elif step == "arxiv":
                tb.call("search_arxiv", {"query": question, "max_results": 3})
            elif step == "claims":
                tb.call("extract_numeric_claims", {"topic": question})
            elif step == "search":
                r = json.loads(tb.call("search_papers", {"query": question, "k": 6}))
                evidence += [tb.seen[x["chunk_id"]] for x in r.get("results", [])]

        corpus_ev = [c for c in evidence if c["type"] not in ("experiment",)]
        exp_ev = [c for c in evidence if c["type"] == "experiment"]
        cov = self.coverage(question, corpus_ev[:4] + exp_ev)
        qt = set(tokenize(question))
        idf = self.index.bm25.idf
        scored = []
        for rank, c in enumerate(corpus_ev):
            for s in SENT.split(re.sub(r"^\[[^\]]*\]\s*", "", c["text"])):
                st = set(tokenize(s))
                if len(s) < 40 or len(s) > 450:
                    continue
                sc = sum(idf.get(t, 0) for t in qt & st) / (1 + 0.15 * rank)
                if sc > 0:
                    scored.append((sc, s.strip(), c))
        scored.sort(key=lambda x: -x[0])
        picked, per_chunk = [], {}
        for sc, s, c in scored:
            if per_chunk.get(c["chunk_id"], 0) >= 2 or any(s == x for x, _ in picked):
                continue  # at most 2 sentences per chunk, no duplicates
            per_chunk[c["chunk_id"]] = per_chunk.get(c["chunk_id"], 0) + 1
            picked.append((s, c))
            if len(picked) >= 5:
                break

        if cov < self.thr or not (picked or extra_lines):
            answer = ("The indexed corpus does not contain sufficient evidence to answer this question "
                      f"(query-term coverage of retrieved evidence: {cov:.2f}).")
            if picked:
                s, c = picked[0]
                answer += f" Closest related evidence: \"{s}\" [{c['chunk_id']}]"
            status, conf = "INSUFFICIENT", "medium"
        else:
            lines = [f"- {s} [{c['chunk_id']}]" for s, c in picked] + extra_lines
            # sentence polarity: a negative/critical marker dominates ("do not beat ... faster")
            pol = [("neg" if NEG.search(s) else "pos" if POS.search(s) else None, c["paper_id"])
                   for s, c in picked]
            pos_p = {p for l, p in pol if l == "pos"}
            neg_p = {p for l, p in pol if l == "neg"}
            conflict = bool(pos_p - neg_p and neg_p - pos_p)  # different papers take opposite sides
            status = "CONFLICTING" if conflict else ("SUPPORTED" if cov >= 0.8 else "PARTIAL")
            conf = "medium" if status == "SUPPORTED" else "low"
            head = ("Evidence from the corpus points in different directions:" if conflict
                    else "Relevant evidence from the corpus:")
            answer = head + "\n" + "\n".join(lines)
        answer += f"\nSTATUS: {status}\nCONFIDENCE: {conf}"
        answer, valid, invalid, status, conf = verify(answer, tb.seen)
        return AgentResult(question, answer, status, conf, valid, invalid, tb.calls, len(tb.calls),
                           round(time.perf_counter() - t0, 2), self.policy)


def make_agent(index: HybridIndex, provider: str = config.LLM_PROVIDER, **kw):
    from .llm import get_llm
    llm = get_llm(provider)
    if llm is None:
        return OfflineAgent(index, **{k: v for k, v in kw.items() if k in ("coverage_threshold", "run_experiments", "allow_network")})
    return LLMAgent(index, llm, **{k: v for k, v in kw.items() if k in ("max_steps", "allow_network")})


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--provider", default=config.LLM_PROVIDER)
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args(argv)
    if not config.INDEX_FILE.exists():
        raise SystemExit("No index found: run `python -m sciagent.collect`, `python -m sciagent.ingest`, "
                         "`python -m sciagent.index` first.")
    idx = HybridIndex.load()
    agent = make_agent(idx, a.provider)
    res = agent.run(a.question)
    print(f"\n[{res.policy}] {res.steps} steps, {res.seconds}s\n")
    if a.trace:
        for t in res.trace:
            print(f"  -> {t['tool']}({json.dumps(t['args'])[:120]})  {t['seconds']}s")
        print()
    print(res.answer)
    if res.invalid_citations:
        print(f"\n[verifier] removed fabricated citations: {res.invalid_citations}")


if __name__ == "__main__":
    main()
