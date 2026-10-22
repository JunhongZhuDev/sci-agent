"""Run the example questions and write a markdown file with answers + tool traces.

    python examples/run_examples.py                       # real corpus, auto provider
    python examples/run_examples.py --fixture --provider offline
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from sciagent import config  # noqa: E402
from sciagent.agent import make_agent  # noqa: E402
from sciagent.index import HybridIndex, load_chunks  # noqa: E402

REAL_QS = [
    "Do machine-learning solvers for fluid PDEs really deliver the large speedups over numerical solvers that are often reported?",
    "Can physics-informed neural networks replace the finite element method for forward problems?",
    "Why do physics-informed neural networks fail on convection problems, and what fixes are proposed?",
    "In the Fourier Neural Operator paper, what does the figure comparing models on the Navier-Stokes equation show?",
    "What drag coefficient do the papers report for a hypersonic re-entry capsule at Mach 25?",
    "Run our own experiment: does a POD-MLP surrogate beat a coarse classical solver for the 1D Burgers equation, "
    "in and out of distribution? Relate the result to the literature.",
]
FIXTURE_QS = [
    "Are learned neural operators really faster than numerical solvers? Compare the claims.",
    "Do physics-informed neural networks beat the finite element method?",
    "What does the figure of relative error versus runtime show for the spectral operator and solver?",
    "What drag coefficient is reported for a hypersonic re-entry capsule at Mach 25?",
    "Run our own experiment: does a POD-MLP surrogate beat a coarse classical solver for the 1D Burgers "
    "equation, in and out of distribution?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--provider", default=config.LLM_PROVIDER)
    a = ap.parse_args()
    if a.fixture:
        from make_fixture import build
        idx = HybridIndex(load_chunks(build(ROOT / "tests" / "fixtures" / "corpus")), prefer_st=False)
        qs, tag = FIXTURE_QS, "fixture"
    else:
        idx, qs, tag = HybridIndex.load(), REAL_QS, "arxiv"
    agent = make_agent(idx, a.provider)
    out = [f"# Example questions and outputs (policy={agent.policy}, corpus={tag})\n"]
    if a.fixture:
        out.append("> The fixture papers are fictional (synthetic test corpus). Re-run without `--fixture` "
                   "on the real arXiv corpus, with an API key for the LLM agent.\n")
    for q in qs:
        r = agent.run(q)
        trace = " → ".join(f"`{t['tool']}`" for t in r.trace) or "(no tools)"
        out.append(f"## Q: {q}\n\n**Tool trace:** {trace}  ({r.steps} steps, {r.seconds}s)\n\n"
                   f"{r.answer}\n")
        if r.invalid_citations:
            out.append(f"*Verifier removed fabricated citations: {r.invalid_citations}*\n")
        print(f"done: {q[:70]}  [{r.status}]")
    path = ROOT / "examples" / f"examples_{tag}_{agent.policy}.md"
    path.write_text("\n".join(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
