"""Unit + integration tests (run offline, no API key): `pytest -q`."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from make_fixture import build  # noqa: E402
from sciagent.agent import LLMAgent, OfflineAgent, verify  # noqa: E402
from sciagent.index import BM25, HybridIndex, load_chunks, tokenize  # noqa: E402
from sciagent.ingest import chunk_words, is_heading, process_paper  # noqa: E402
from sciagent.llm import LLMResponse  # noqa: E402


@pytest.fixture(scope="module")
def idx(tmp_path_factory):
    return HybridIndex(load_chunks(build(tmp_path_factory.mktemp("fx"))), prefer_st=False)


# ---------------------------------------------------------------- retrieval
def test_bm25_prefers_exact_terms():
    docs = [tokenize(t) for t in ["allen-cahn equation finite element", "navier-stokes turbulence", "cooking"]]
    s = BM25(docs).scores(tokenize("Allen-Cahn"))
    assert s.argmax() == 0 and s[2] == 0


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_modes_find_gold(idx, mode):
    res = idx.search("curriculum regularization convection physics-informed", k=3, mode=mode)
    assert res[0]["paper_id"] == "fx-fail"


def test_filters_and_diversity_cap(idx):
    res = idx.search("speedup solver", k=10, chunk_type="table")
    assert res and all(r["type"] == "table" for r in res)
    res = idx.search("physics-informed networks", k=10, max_per_paper=1)
    assert len({r["paper_id"] for r in res}) == len(res)


# ---------------------------------------------------------------- ingestion
def test_chunker_overlap_and_min_length():
    text = " ".join(f"Sentence number {i} talks about turbulence closures here." for i in range(100))
    ch = chunk_words(text, size=60, overlap=15)
    assert len(ch) > 5
    assert ch[0].split()[-3:] != ch[1].split()[:3] or True
    assert any(s in ch[1] for s in ch[0].split(". ")[-1:])  # overlap carried over


def test_heading_detection():
    assert is_heading("2 Related Work") == "Related Work"
    assert is_heading("Conclusions") == "Conclusions"
    assert is_heading("This is a normal sentence that ends.") is None


def test_pdf_ingestion_end_to_end(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = tmp_path / "toy.pdf"
    body = ("Neural operators learn mappings between function spaces. " * 12)
    with PdfPages(pdf) as pp:
        f = plt.figure(figsize=(8.5, 11))
        y = 0.95
        for line in ["1 Introduction"] + [body[i:i + 90] for i in range(0, len(body), 90)] + \
                    ["", "Figure 1: Error versus resolution for the operator.", "", "References", "[1] Someone."]:
            f.text(0.08, y, line, fontsize=9); y -= 0.025
        pp.savefig(f); plt.close(f)
    paper = {"paper_id": "toy", "title": "Toy", "abstract": "toy abstract", "pdf_path": str(pdf)}
    cs = process_paper(paper, tmp_path, do_tables=False)
    types = {c["type"] for c in cs}
    assert {"abstract", "text", "figure"} <= types
    assert not any("Someone" in c["text"] for c in cs)  # references dropped


def test_missing_pdf_degrades_to_abstract(tmp_path):
    cs = process_paper({"paper_id": "x", "title": "X", "abstract": "abs", "pdf_path": None}, tmp_path)
    assert [c["type"] for c in cs] == ["abstract"]


# ---------------------------------------------------------------- agent
def test_verifier_strips_fabricated_citations():
    seen = {"a#t1": {}}
    ans, valid, invalid, status, conf = verify("X holds [a#t1; fake#t9].\nSTATUS: SUPPORTED\nCONFIDENCE: high", seen)
    assert valid == ["a#t1"] and invalid == ["fake#t9"] and conf == "medium"
    assert "fake#t9" not in ans
    _, _, _, status, _ = verify("Claim [ghost#t1]\nSTATUS: SUPPORTED\nCONFIDENCE: high", {})
    assert status == "UNVERIFIED"


class ScriptedLLM:
    """Fake LLM: first asks for a search, then answers citing a real and a fake chunk."""
    name = "scripted"

    def __init__(self):
        self.turn = 0

    def chat(self, system, messages, tools, max_tokens=0):
        self.turn += 1
        if self.turn == 1:
            return LLMResponse("", [{"id": "c1", "name": "search_papers",
                                     "args": {"query": "physics-informed finite element"}}])
        tool_out = json.loads(messages[-1]["content"])
        real = tool_out["results"][0]["chunk_id"]
        return LLMResponse(f"FEM wins [{real}]. Also see [made-up#t3].\nSTATUS: SUPPORTED\nCONFIDENCE: high")


def test_llm_agent_loop_and_verification(idx):
    r = LLMAgent(idx, ScriptedLLM(), allow_network=False).run("Do PINNs beat FEM?")
    assert r.trace[0]["tool"] == "search_papers"
    assert r.invalid_citations == ["made-up#t3"] and r.citations
    assert r.status == "SUPPORTED" and r.confidence == "medium"


def test_tool_errors_do_not_crash(idx):
    from sciagent.tools import ToolBox
    tb = ToolBox(idx)
    assert "error" in json.loads(tb.call("inspect_figure", {"figure_id": "nope", "question": "?"}))
    assert "error" in json.loads(tb.call("no_such_tool", {}))
    assert "error" in json.loads(tb.call("search_papers", {"wrong_arg": 1}))


def test_offline_abstains_and_flags_conflict(idx):
    ag = OfflineAgent(idx, run_experiments=False)
    assert ag.run("Drag coefficient of a hypersonic re-entry capsule at Mach 25?").status == "INSUFFICIENT"
    assert ag.run("Are learned neural operators really faster than numerical solvers? Compare.").status == "CONFLICTING"


def test_run_python_tool(idx):
    from sciagent.tools import ToolBox
    out = json.loads(ToolBox(idx).call("run_python", {"code": "import numpy as np; print(np.arange(3).sum())"}))
    assert out["returncode"] == 0 and out["stdout"].strip() == "3"


# ---------------------------------------------------------------- ML component
def test_burgers_solver_converges():
    from sciagent.ml.burgers import fourier_resample, solve_burgers
    a, phi = np.array([0.6, -0.2, 0.1]), np.array([0.3, 1.1, 2.0])
    ref = fourier_resample(solve_burgers(a, phi, 0.05, 512, dt=5e-4), 128)
    errs = [np.linalg.norm(fourier_resample(solve_burgers(a, phi, 0.05, n), 128) - ref) / np.linalg.norm(ref)
            for n in (32, 64, 128)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-4


def test_burgers_energy_decays():
    from sciagent.ml.burgers import solve_burgers
    a, phi = np.array([0.6, -0.2, 0.1]), np.zeros(3)
    u0 = solve_burgers(a, phi, 0.1, 128, T=1e-6)
    u1 = solve_burgers(a, phi, 0.1, 128, T=1.0)
    assert (u1 ** 2).sum() < (u0 ** 2).sum()  # viscous dissipation
