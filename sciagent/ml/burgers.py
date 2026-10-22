"""ML component: does a learned surrogate beat a classical solver? (1D Burgers)

This mirrors the central debate in the collected literature (FNO/PINN speedup
claims vs. "weak baselines" critiques) with a small, honest experiment that
the agent can run as a tool.

Problem: viscous Burgers, u_t + u u_x = nu u_xx on [0, 2pi), periodic,
         u0(x) = sum_{m=1..3} a_m sin(m x + phi_m), predict u(x, T=1).
Ground truth: pseudo-spectral solver, 256 modes, 2/3 de-aliasing, RK45.

Contestants (all evaluated on the same 128-point grid):
  * coarse classical solver (32 modes)       <- the *fair* cheap baseline
  * POD + ridge regression (poly-2 features)
  * POD + MLP (sklearn), hyper-parameters chosen by 5-fold CV
Reported: relative L2 error in-distribution and out-of-distribution
(viscosity outside the training range), cost per query, and the break-even
number of queries once data generation + training are amortised.

ML fundamentals shown: train/val/test split, k-fold CV for model selection,
dimensionality reduction (POD = PCA on snapshots), input standardisation,
OOD generalisation, and an equal-footing baseline comparison.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

L = 2 * np.pi
N_EVAL = 128


# ---------------------------------------------------------------- solver
def solve_burgers(a: np.ndarray, phi: np.ndarray, nu: float, n: int = 256, T: float = 1.0,
                  dt: float | None = None) -> np.ndarray:
    """Pseudo-spectral Burgers solver, integrating-factor RK4 (diffusion treated
    exactly, so the time step is limited only by advection CFL)."""
    x = np.arange(n) * L / n
    u0 = sum(a[m] * np.sin((m + 1) * x + phi[m]) for m in range(len(a)))
    k = np.fft.rfftfreq(n, d=L / (2 * np.pi * n))
    dealias = k < (2 / 3) * k.max()
    if dt is None:  # CFL: max|u| <= ~1.8 for our ICs
        dt = min(0.01, 0.5 * (L / n) / 2.0)
    steps = int(np.ceil(T / dt)); dt = T / steps
    E, E2 = np.exp(-nu * k ** 2 * dt), np.exp(-nu * k ** 2 * dt / 2)

    def N(vh):  # nonlinear term -(u^2/2)_x in Fourier space
        u = np.fft.irfft(vh * dealias, n)
        return -0.5j * k * np.fft.rfft(u * u) * dealias

    vh = np.fft.rfft(u0)
    for _ in range(steps):
        k1 = N(vh)
        k2 = N(E2 * (vh + 0.5 * dt * k1))
        k3 = N(E2 * vh + 0.5 * dt * k2)
        k4 = N(E * vh + dt * E2 * k3)
        vh = E * vh + dt / 6 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
    return np.fft.irfft(vh, n)


def fourier_resample(u: np.ndarray, m: int) -> np.ndarray:
    """Spectral interpolation / restriction of a periodic signal to m points."""
    uh = np.fft.rfft(u) / len(u)
    out = np.zeros(m // 2 + 1, complex)
    nk = min(len(uh), len(out))
    out[:nk] = uh[:nk]
    return np.fft.irfft(out * m, m)


# ---------------------------------------------------------------- data
def sample_params(rng, n: int, nu_range: tuple[float, float]):
    a = rng.uniform(-1, 1, (n, 3)) / np.array([1, 2, 3])
    phi = rng.uniform(0, 2 * np.pi, (n, 3))
    nu = np.exp(rng.uniform(np.log(nu_range[0]), np.log(nu_range[1]), n))
    return a, phi, nu


def features(a, phi, nu) -> np.ndarray:
    # IC Fourier coefficients (linear in a) + viscosity
    return np.column_stack([a * np.cos(phi), a * np.sin(phi), np.log(nu), nu])


def make_dataset(rng, n, nu_range, n_ref=256):
    a, phi, nu = sample_params(rng, n, nu_range)
    t0 = time.perf_counter()
    Y = np.stack([fourier_resample(solve_burgers(a[i], phi[i], nu[i], n_ref), N_EVAL) for i in range(n)])
    dt = (time.perf_counter() - t0) / n
    return features(a, phi, nu), Y, (a, phi, nu), dt


def rel_l2(pred, true) -> np.ndarray:
    return np.linalg.norm(pred - true, axis=1) / np.linalg.norm(true, axis=1)


@dataclass
class Row:
    method: str
    err_in_mean: float
    err_in_std: float
    err_ood_mean: float
    cost_per_query_ms: float
    offline_cost_s: float
    note: str = ""


def run_experiment(n_train: int = 400, n_test: int = 50, n_ood: int = 40,
                   nu_train=(0.05, 0.3), nu_ood=(0.015, 0.04), seed: int = 0,
                   coarse_modes: int = 32, plot_path: str | None = None,
                   cache: str | None = None, use_tf: bool | None = None) -> dict:
    if use_tf is None:
        import os
        use_tf = os.environ.get("SCIAGENT_TF", "0") == "1"
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    rng = np.random.default_rng(seed)
    t_gen = time.perf_counter()
    cache_p = Path(cache) if cache else None
    key = f"{n_train}_{n_test}_{n_ood}_{nu_train}_{nu_ood}_{seed}"
    if cache_p and cache_p.exists() and str(np.load(cache_p, allow_pickle=True)["key"]) == key:
        z = np.load(cache_p, allow_pickle=True)
        Xtr, Ytr, Xte, Yte, Xood, Yood = (z[k] for k in ("Xtr", "Ytr", "Xte", "Yte", "Xood", "Yood"))
        Pte, Pood = z["Pte"].item(), z["Pood"].item()
        ref_cost = float(z["ref_cost"])
    else:
        Xtr, Ytr, _, ref_cost = make_dataset(rng, n_train, nu_train)
        Xte, Yte, Pte_t, _ = make_dataset(rng, n_test, nu_train)
        Xood, Yood, Pood_t, _ = make_dataset(rng, n_ood, nu_ood)
        Pte = dict(zip(("a", "phi", "nu"), Pte_t)); Pood = dict(zip(("a", "phi", "nu"), Pood_t))
        if cache_p:
            np.savez(cache_p, key=key, Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, Xood=Xood, Yood=Yood,
                     Pte=np.array(Pte, dtype=object), Pood=np.array(Pood, dtype=object), ref_cost=ref_cost)
    gen_time = n_train * ref_cost  # what data generation costs (cache-independent)

    rows: list[Row] = []
    rows.append(Row("reference solver (256 modes)", 0.0, 0.0, 0.0, ref_cost * 1e3, 0.0, "ground truth"))

    # -- coarse classical baseline
    def coarse(P):
        t = time.perf_counter()
        out = np.stack([fourier_resample(solve_burgers(P["a"][i], P["phi"][i], P["nu"][i], coarse_modes), N_EVAL)
                        for i in range(len(P["nu"]))])
        return out, (time.perf_counter() - t) / len(P["nu"])
    c_in, c_cost = coarse(Pte)
    c_ood, _ = coarse(Pood)
    e_in, e_ood = rel_l2(c_in, Yte), rel_l2(c_ood, Yood)
    rows.append(Row(f"coarse solver ({coarse_modes} modes)", e_in.mean(), e_in.std(), e_ood.mean(),
                    c_cost * 1e3, 0.0, "no training data needed"))

    # -- POD of training snapshots
    pca = PCA(n_components=0.9999, svd_solver="full").fit(Ytr)
    r = pca.n_components_
    Ztr = pca.transform(Ytr)
    cv = KFold(5, shuffle=True, random_state=seed)

    models = {
        "POD + ridge (poly-2)": GridSearchCV(
            make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge()),
            {"ridge__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]}, cv=cv,
            scoring="neg_mean_squared_error"),
        "POD + MLP": GridSearchCV(
            make_pipeline(StandardScaler(), MLPRegressor(max_iter=3000, early_stopping=True,
                                                         random_state=seed)),
            {"mlpregressor__hidden_layer_sizes": [(64, 64), (128, 128)],
             "mlpregressor__alpha": [1e-4, 1e-3]}, cv=cv, scoring="neg_mean_squared_error"),
    }
    preds = {}
    for name, gs in models.items():
        t = time.perf_counter(); gs.fit(Xtr, Ztr); train_t = time.perf_counter() - t
        t = time.perf_counter(); p_in = pca.inverse_transform(gs.predict(Xte))
        q_cost = (time.perf_counter() - t) / len(Xte)
        p_ood = pca.inverse_transform(gs.predict(Xood))
        e_in, e_ood = rel_l2(p_in, Yte), rel_l2(p_ood, Yood)
        preds[name] = (p_in, p_ood)
        rows.append(Row(name, e_in.mean(), e_in.std(), e_ood.mean(), q_cost * 1e3,
                        gen_time + train_t, f"best={gs.best_params_}"))

    # -- optional TensorFlow/Keras MLP on the same POD coefficients
    if use_tf:
        try:
            t = time.perf_counter(); predict, info = keras_surrogate(Xtr, Ztr, seed)
            train_t = time.perf_counter() - t
            t = time.perf_counter(); p_in = pca.inverse_transform(predict(Xte))
            q_cost = (time.perf_counter() - t) / len(Xte)
            p_ood = pca.inverse_transform(predict(Xood))
            e_in, e_ood = rel_l2(p_in, Yte), rel_l2(p_ood, Yood)
            preds["POD + MLP (Keras)"] = (p_in, p_ood)
            rows.append(Row("POD + MLP (Keras)", e_in.mean(), e_in.std(), e_ood.mean(), q_cost * 1e3,
                            gen_time + train_t, f"best={info}"))
        except ImportError:
            print("TensorFlow not installed - skipping Keras surrogate")

    # break-even: queries N where  offline + N*c_s  <  N*c_ref
    for rw in rows:
        if rw.offline_cost_s > 0:
            saving = ref_cost - rw.cost_per_query_ms / 1e3
            rw.note += f"; break-even after ~{int(rw.offline_cost_s / saving) if saving > 0 else 'never'} queries"

    best_sur = min((r_ for r_ in rows if r_.method.startswith("POD")), key=lambda r_: r_.err_in_mean)
    coarse_row = rows[1]
    summary = (
        f"1D viscous Burgers equation experiment (surrogate vs classical solver). POD rank r={r}. Best surrogate '{best_sur.method}': in-dist error {best_sur.err_in_mean:.2%}, "
        f"OOD error {best_sur.err_ood_mean:.2%}. Coarse {coarse_modes}-mode solver: in-dist "
        f"{coarse_row.err_in_mean:.2%}, OOD {coarse_row.err_ood_mean:.2%}, at "
        f"{coarse_row.cost_per_query_ms:.1f} ms/query vs surrogate {best_sur.cost_per_query_ms:.3f} ms. "
        "Surrogate errors grow sharply outside the training viscosity range, while the classical "
        "solver's do not depend on any training distribution; the speed-up must be weighed against "
        "offline data-generation cost and the accuracy of an equally cheap classical baseline."
    )

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(N_EVAL) * L / N_EVAL
        i_ood = int(np.argmin(Pood["nu"]))
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        for a_, idx, Y, tag, ci in ((ax[0], 0, Yte, "in-distribution", c_in), (ax[1], i_ood, Yood, "OOD (low nu)", c_ood)):
            a_.plot(x, Y[idx], "k", lw=2, label="reference")
            a_.plot(x, ci[idx], "--", label=f"coarse {coarse_modes}")
            for name, (pi, po) in preds.items():
                a_.plot(x, (pi if tag.startswith("in") else po)[idx], label=name)
            a_.set_title(f"u(x, T=1), {tag}"); a_.set_xlabel("x")
        ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(plot_path, dpi=110); plt.close(fig)

    return {"rows": [asdict(r_) for r_ in rows], "pod_rank": int(r), "summary": summary,
            "plot": plot_path, "config": {"n_train": n_train, "n_test": n_test, "n_ood": n_ood,
                                          "nu_train": nu_train, "nu_ood": nu_ood, "seed": seed}}


def keras_surrogate(Xtr: np.ndarray, Ztr: np.ndarray, seed: int = 0):
    """TensorFlow/Keras MLP: standardised inputs AND outputs (POD coefficients have very
    different scales), Adam, early stopping + LR decay on a 20 % validation split, and a
    small width sweep selected on validation loss. Returns (predict_fn, info)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    zm, zs = Ztr.mean(0), Ztr.std(0) + 1e-8
    Xn, Zn = (Xtr - xm) / xs, (Ztr - zm) / zs
    best = None
    for width in (64, 128):
        model = tf.keras.Sequential([
            tf.keras.layers.Input((Xn.shape[1],)),
            tf.keras.layers.Dense(width, activation="gelu"),
            tf.keras.layers.Dense(width, activation="gelu"),
            tf.keras.layers.Dense(Zn.shape[1]),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        h = model.fit(Xn, Zn, validation_split=0.2, epochs=3000, batch_size=32, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=150, restore_best_weights=True),
                                 tf.keras.callbacks.ReduceLROnPlateau(patience=50, factor=0.5, min_lr=1e-5)])
        v = float(min(h.history["val_loss"]))
        if best is None or v < best[0]:
            best = (v, model, width, len(h.history["loss"]))
    v, model, width, epochs = best

    def predict(X):
        return model((X - xm) / xs, training=False).numpy() * zs + zm
    return predict, {"width": width, "val_mse_std": round(v, 4), "epochs": epochs}


def format_table(res: dict) -> str:
    lines = ["| method | rel. L2 in-dist | rel. L2 OOD | ms/query | offline cost (s) | note |",
             "|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['method']} | {r['err_in_mean']:.3%} ± {r['err_in_std']:.3%} | "
                     f"{r['err_ood_mean']:.3%} | {r['cost_per_query_ms']:.3f} | {r['offline_cost_s']:.1f} | {r['note']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import json, sys
    import os; os.makedirs("results", exist_ok=True)
    res = run_experiment(plot_path="results/burgers_surrogate.png", use_tf="--tf" in sys.argv or None)
    print(format_table(res)); print(res["summary"])
    json.dump(res, open("results/burgers_surrogate.json", "w"), indent=2, default=str)
