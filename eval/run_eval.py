"""Evaluation harness.

Three layers, each answering a different question:
 1. Retrieval    - does the retriever surface the gold papers?  (no LLM needed)
                   paper-level Recall@{3,5,10} and MRR, ablated over bm25 / dense / hybrid.
 2. End-to-end   - does the agent answer correctly, cite valid evidence, pick sensible
                   tools and abstain when it should?
 3. Robustness   - (a) missing evidence: remove the gold paper and re-ask -> the agent
                       should abstain / hedge rather than answer confidently;
                   (b) injected conflict: add a contradicting chunk -> the agent should
                       report the conflict.
Optional: LLM-as-judge faithfulness (are answer sentences supported by cited text?).

Usage:
    python eval/run_eval.py                       # real corpus, auto provider
    python eval/run_eval.py --provider offline    # deterministic baseline agent
    python eval/run_eval.py --fixture             # synthetic corpus (CI)
    python eval/run_eval.py --retrieval-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sciagent import config  # noqa: E402
from sciagent.agent import make_agent  # noqa: E402
from sciagent.index import HybridIndex, load_chunks  # noqa: E402
from sciagent.llm import get_llm  # noqa: E402


def load_q(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------------------------------------------------- retrieval
def retrieval_eval(idx: HybridIndex, qs: list[dict], ks=(3, 5, 10)) -> dict:
    out = {}
    qs = [q for q in qs if q["gold_papers"] and q["type"] != "unanswerable"]
    for mode in ("bm25", "dense", "hybrid"):
        rec = defaultdict(list); rr = []
        for q in qs:
            res = idx.search(q["question"], k=max(ks) * 3, mode=mode, max_per_paper=1)
            papers = [r["paper_id"] for r in res]
            gold = set(q["gold_papers"])
            for k in ks:
                rec[k].append(len(gold & set(papers[:k])) / len(gold))
            first = next((i for i, p in enumerate(papers) if p in gold), None)
            rr.append(0 if first is None else 1 / (first + 1))
        out[mode] = {**{f"recall@{k}": round(sum(v) / len(v), 3) for k, v in rec.items()},
                     "mrr": round(sum(rr) / len(rr), 3), "n": len(qs)}
    return out


# ---------------------------------------------------------------- end-to-end
def score_answer(q: dict, r) -> dict:
    ans = r.answer.lower()
    groups = q.get("must_include") or []
    kw = (sum(any(s.lower() in ans for s in g) for g in groups) / len(groups)) if groups else None
    cited_papers = [c.split("#")[0].replace("arxiv:", "") for c in r.citations]
    gold = set(q["gold_papers"])
    n_cit = len(r.citations) + len(r.invalid_citations)
    tools_used = {t["tool"] for t in r.trace}
    return {
        "id": q["id"], "type": q["type"], "status": r.status, "confidence": r.confidence,
        "status_ok": r.status in q["expected_status"],
        "keyword_score": kw,
        "citation_validity": (len(r.citations) / n_cit) if n_cit else None,
        "gold_cited": (bool(gold & set(cited_papers)) if gold and q["type"] != "unanswerable" else None),
        "citation_precision": (sum(p in gold for p in cited_papers) / len(cited_papers)
                               if gold and cited_papers and q["type"] in ("factual", "figure") else None),
        "tools_ok": set(q.get("expected_tools", [])) <= tools_used,
        "tools": sorted(tools_used), "steps": r.steps, "seconds": r.seconds,
        "answer": r.answer,
    }


def judge_faithfulness(llm, idx, r) -> float | None:
    """LLM-as-judge: fraction of answer claims supported by the cited chunks."""
    if llm is None or not r.citations:
        return None
    ev = "\n\n".join(f"[{c}] {(idx.get(c) or {}).get('text', '')[:1200]}" for c in r.citations[:8])
    prompt = ("You are grading faithfulness. EVIDENCE:\n" + ev + "\n\nANSWER:\n" + r.answer +
              "\n\nSplit the answer into factual claims. For each, decide if it is supported by the cited "
              "evidence. Reply with only JSON: {\"supported\": <int>, \"total\": <int>}")
    try:
        resp = llm.chat("You are a strict grader.", [{"role": "user", "content": prompt}], [])
        j = json.loads(resp.text[resp.text.find("{"): resp.text.rfind("}") + 1])
        return j["supported"] / max(1, j["total"])
    except Exception:
        return None


def agg(rows: list[dict]) -> dict:
    def mean(key, rs):
        v = [r[key] for r in rs if r.get(key) is not None]
        return round(sum(v) / len(v), 3) if v else None
    out = {"overall": {k: mean(k, rows) for k in ("status_ok", "keyword_score", "citation_validity",
                                                  "gold_cited", "citation_precision", "tools_ok",
                                                  "faithfulness", "steps", "seconds")}}
    out["overall"]["n"] = len(rows)
    for t in sorted({r["type"] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        out[t] = {k: mean(k, rs) for k in ("status_ok", "keyword_score", "gold_cited", "tools_ok")}
        out[t]["n"] = len(rs)
    return out


# ---------------------------------------------------------------- robustness
def robustness_eval(chunks, qs, provider, prefer_st, agent_kw) -> dict:
    res = {"missing_evidence": [], "injected_conflict": []}
    for q in qs:
        if q["type"] != "factual" or len(q["gold_papers"]) != 1:
            continue
        gold = q["gold_papers"][0]
        # (a) remove the gold paper entirely
        idx = HybridIndex([c for c in chunks if c["paper_id"] != gold], prefer_st=prefer_st)
        r = make_agent(idx, provider, **agent_kw).run(q["question"])
        hedged = r.status in ("INSUFFICIENT", "PARTIAL", "UNVERIFIED") or r.confidence == "low"
        res["missing_evidence"].append({"id": q["id"], "status": r.status, "confidence": r.confidence,
                                        "hedged": hedged})
        # (b) inject a contradicting chunk
        if q.get("counter_claim"):
            fake = {"paper_id": "inj-counter", "title": "Injected counter-evidence (synthetic)",
                    "year": 2025, "authors": [], "chunk_id": "inj-counter#abs", "type": "abstract",
                    "section": "Abstract", "page": 1, "text": q["counter_claim"]}
            idx = HybridIndex(chunks + [fake], prefer_st=prefer_st)
            r = make_agent(idx, provider, **agent_kw).run(q["question"])
            res["injected_conflict"].append({
                "id": q["id"], "status": r.status,
                "detected": r.status == "CONFLICTING" or ("inj-counter#abs" in r.citations and
                                                          any(w in r.answer.lower() for w in
                                                              ("conflict", "disagree", "contradict", "however"))),
                "cited_counter": "inj-counter#abs" in r.citations})
    summ = {}
    if res["missing_evidence"]:
        summ["abstain_or_hedge_when_gold_removed"] = round(
            sum(x["hedged"] for x in res["missing_evidence"]) / len(res["missing_evidence"]), 3)
    if res["injected_conflict"]:
        summ["conflict_detected_when_injected"] = round(
            sum(x["detected"] for x in res["injected_conflict"]) / len(res["injected_conflict"]), 3)
    return {"summary": summ, **res}


# ---------------------------------------------------------------- report
def to_markdown(report: dict) -> str:
    L = [f"# Evaluation report ({report['corpus']}, policy={report['policy']}"
         + (f", model={report['model']}" if report.get('model') else "") + ")", ""]
    L += ["## Retrieval (paper-level)", "", "| mode | recall@3 | recall@5 | recall@10 | MRR |", "|---|---|---|---|---|"]
    for m, v in report["retrieval"].items():
        L.append(f"| {m} | {v['recall@3']} | {v['recall@5']} | {v['recall@10']} | {v['mrr']} |")
    if "e2e" in report:
        o = report["e2e"]["overall"]
        L += ["", "## End-to-end", "", "| metric | value |", "|---|---|"]
        for k, v in o.items():
            L.append(f"| {k} | {v} |")
        L += ["", "| type | n | status_ok | keyword_score | gold_cited | tools_ok |", "|---|---|---|---|---|---|"]
        for t, v in report["e2e"].items():
            if t != "overall":
                L.append(f"| {t} | {v['n']} | {v['status_ok']} | {v['keyword_score']} | {v['gold_cited']} | {v['tools_ok']} |")
        L += ["", "### Per question", "", "| id | status | ok | kw | tools |", "|---|---|---|---|---|"]
        for r in report["rows"]:
            L.append(f"| {r['id']} | {r['status']} | {'✓' if r['status_ok'] else '✗'} | "
                     f"{r['keyword_score'] if r['keyword_score'] is not None else '-'} | {', '.join(r['tools'])} |")
    if "robustness" in report:
        L += ["", "## Robustness", ""]
        for k, v in report["robustness"]["summary"].items():
            L.append(f"- **{k}**: {v}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true", help="use the synthetic test corpus")
    ap.add_argument("--provider", default=config.LLM_PROVIDER)
    ap.add_argument("--questions")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--no-robustness", action="store_true")
    ap.add_argument("--no-experiments", action="store_true", help="skip analysis questions (slow)")
    ap.add_argument("--lsa", action="store_true", help="force LSA dense encoder")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness (needs API key)")
    ap.add_argument("--out", default=str(config.RESULTS_DIR))
    a = ap.parse_args(argv)

    if a.fixture:
        sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
        from make_fixture import build
        chunks = load_chunks(build(ROOT / "tests" / "fixtures" / "corpus"))
        qpath = a.questions or ROOT / "eval" / "fixture_questions.jsonl"
        corpus = "fixture"
    else:
        if not config.CHUNKS_FILE.exists():
            raise SystemExit("No real corpus yet: run collect -> ingest -> index first "
                             "(or use --fixture for the synthetic test corpus).")
        chunks = load_chunks()
        qpath = a.questions or ROOT / "eval" / "questions.jsonl"
        corpus = "arxiv"
    qs = load_q(qpath)
    if a.no_experiments:
        qs = [q for q in qs if q["type"] != "analysis"]
    prefer_st = not a.lsa
    idx = HybridIndex(chunks, prefer_st=prefer_st)
    llm = get_llm(a.provider)
    policy = "offline" if llm is None else f"llm:{llm.name}"
    model = None if llm is None else getattr(llm, "model", None)
    agent_kw = {"allow_network": False}
    report = {"corpus": corpus, "policy": policy, "model": model, "dense": idx.dense.kind, "n_chunks": len(chunks),
              "n_papers": len(idx.papers()), "time": time.strftime("%Y-%m-%d %H:%M")}

    print(f"corpus={corpus} papers={report['n_papers']} chunks={len(chunks)} dense={idx.dense.kind} policy={policy}")
    report["retrieval"] = retrieval_eval(idx, qs)
    print(json.dumps(report["retrieval"], indent=1))

    if not a.retrieval_only:
        agent = make_agent(idx, a.provider, **agent_kw)
        rows = []
        for q in qs:
            r = agent.run(q["question"])
            row = score_answer(q, r)
            if a.judge:
                row["faithfulness"] = judge_faithfulness(llm, idx, r)
            rows.append(row)
            print(f"  {q['id']:4s} {r.status:12s} ok={row['status_ok']!s:5s} kw={row['keyword_score']} "
                  f"tools={row['tools']}")
        report["rows"] = rows
        report["e2e"] = agg(rows)
        if not a.no_robustness:
            report["robustness"] = robustness_eval(chunks, qs, a.provider, prefer_st, agent_kw)
            print("robustness:", report["robustness"]["summary"])

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{corpus}_{policy.replace(':', '-')}"
    json.dump(report, open(out / f"{stem}.json", "w"), indent=2, default=str)
    (out / f"{stem}.md").write_text(to_markdown(report))
    print(f"wrote {out / stem}.json/.md")


if __name__ == "__main__":
    main()
