"""Builds a small SYNTHETIC corpus (fictional papers, ids fx-*) for tests and CI.

It deliberately contains: agreeing evidence, a direct conflict (speed-up claim
vs. weak-baseline critique), a figure with an image, a table, and topics that
are NOT covered (for abstention tests). Content is invented - do not cite it.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

P = {
    "fx-fno": ("Fast Spectral Operators for Turbulent Flow (fictional)", 2021),
    "fx-crit": ("Fair Baselines for Learned PDE Solvers (fictional)", 2024),
    "fx-pinn": ("Physics-Informed Networks versus Finite Elements (fictional)", 2023),
    "fx-fail": ("Training Pathologies of Physics-Informed Networks (fictional)", 2022),
    "fx-gnn": ("Graph Networks for Mesh-Based Simulation (fictional)", 2021),
    "fx-rans": ("Data-Driven Closures for RANS Turbulence Models (fictional)", 2019),
}

TEXT = {
    "fx-fno": [
        ("Abstract", "We propose a spectral neural operator that learns the solution operator of the "
         "Navier-Stokes equations. On two-dimensional turbulent flow at viscosity 1e-4 the operator is "
         "about 1000 times faster than the pseudo-spectral solver used to generate training data, "
         "while reaching a relative error of 2 percent."),
        ("Results", "The operator generalises to higher resolution without retraining (zero-shot "
         "super-resolution). Training required 10000 solver trajectories and 12 GPU hours. The speedup "
         "is measured against the high-resolution solver at its native resolution."),
    ],
    "fx-crit": [
        ("Abstract", "We review published comparisons between machine learning PDE solvers and "
         "numerical methods for fluid problems. Most studies compare against weak baselines: when the "
         "numerical solver is run at a resolution that matches the learned model's accuracy, reported "
         "speedups shrink by one to three orders of magnitude, and several learned solvers are slower."),
        ("Discussion", "Reporting biases such as omitting negative results lead to overoptimism. We "
         "recommend comparing at equal accuracy or equal runtime and including the cost of generating "
         "training data."),
    ],
    "fx-pinn": [
        ("Abstract", "We compare physics-informed neural networks with the finite element method for "
         "the Poisson and Allen-Cahn equations. The finite element method is faster and more accurate "
         "in solving time; physics-informed networks do not beat finite elements in our tests, but a "
         "trained network evaluates new points quickly."),
        ("Conclusion", "For forward problems in low dimension classical solvers remain preferable. "
         "Physics-informed networks may be useful for inverse problems or very high dimension."),
    ],
    "fx-fail": [
        ("Abstract", "Physics-informed networks fail to train for convection problems with large "
         "convection coefficients. The loss landscape becomes ill-conditioned because the PDE residual "
         "term dominates the gradient."),
        ("Method", "Curriculum regularization, starting from small convection coefficients, and "
         "sequence-to-sequence training in time reduce the error by one to two orders of magnitude."),
    ],
    "fx-gnn": [
        ("Abstract", "A message-passing graph neural network learns mesh-based simulation for cloth, "
         "aerodynamics and structural mechanics. It runs 10 to 100 times faster than the simulator it "
         "was trained on and supports adaptive remeshing."),
    ],
    "fx-rans": [
        ("Introduction", "Reynolds-averaged Navier-Stokes (RANS) models rely on closures for the "
         "Reynolds stress. Data-driven closures trained on high-fidelity data can correct model-form "
         "errors, but must respect Galilean invariance and realizability."),
    ],
}


def build(out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for pid, (title, year) in P.items():
        base = {"paper_id": pid, "title": title, "year": year, "authors": ["Fixture A."]}
        for n, (sec, txt) in enumerate(TEXT[pid]):
            ctype = "abstract" if sec == "Abstract" else "text"
            cid = f"{pid}#abs" if ctype == "abstract" else f"{pid}#t{n}"
            chunks.append({**base, "chunk_id": cid, "type": ctype, "section": sec, "page": 1 + n,
                           "text": txt if ctype == "abstract" else f"[{sec}] {txt}"})
    # figure with image
    img = out_dir / "fx-fno_p3_fig2.png"
    x = np.logspace(0, 4, 20)
    plt.figure(figsize=(4, 3)); plt.loglog(x, 1 / x, label="solver"); plt.loglog(x, 0.02 + 0 * x, label="operator")
    plt.xlabel("runtime budget"); plt.ylabel("relative error"); plt.legend(); plt.tight_layout()
    plt.savefig(img); plt.close()
    chunks.append({"paper_id": "fx-fno", "title": P["fx-fno"][0], "year": 2021, "authors": [],
                   "chunk_id": "fx-fno#fig2", "type": "figure", "section": "Results", "page": 3,
                   "text": "Figure 2: Relative error versus runtime for the spectral operator and the "
                           "pseudo-spectral solver on 2D turbulence.", "image_path": str(img)})
    chunks.append({"paper_id": "fx-crit", "title": P["fx-crit"][0], "year": 2024, "authors": [],
                   "chunk_id": "fx-crit#tab1", "type": "table", "section": "Results", "page": 4,
                   "text": "Table 1: Reported versus baseline-corrected speedup of learned solvers.\n"
                           "method | reported speedup | corrected speedup\n"
                           "operator A | 1000x | 7x\nPINN B | 100x | 0.1x\nGNN C | 50x | 4x"})
    path = out_dir / "chunks.jsonl"
    with open(path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return path


if __name__ == "__main__":
    print(build(HERE / "corpus"))
