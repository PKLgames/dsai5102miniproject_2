"""E2 - How few indicators are enough? Two independent selection criteria.

Criterion A (label-blind): elastic-net sparse PCA. The income labels are never
    seen, so the resulting set can be evaluated by plain cross-validation
    without leakage. Confidence comes from bootstrapping the countries.

Criterion B (label-aware): forward selection on balanced accuracy, with the
    whole selection repeated inside every outer fold (nested CV). This gives an
    honest estimate of what a supervised shortlist can do -- and a second,
    independent ranking of the indicators.

Writes results/metrics/E2_summary.json, results/metrics/sets.json,
results/tables/T3_selection, results/figures/F3_selection.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from sparse_solver import sparse_pca, subspace_fev

ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 1.00, 1.50]
BOOT = 200
KMAX = 14
TREES = 120


def rf():
    return RandomForestClassifier(n_estimators=TREES, random_state=0, n_jobs=1)


# ------------------------------------------------------- criterion A -------
def sparse_curve(Z: np.ndarray) -> tuple[pd.DataFrame, dict]:
    Zc = Z - Z.mean(axis=0)
    cov = np.cov(Z.T)
    rows, loadings = [], {}
    for a in ALPHAS:
        W, _ = sparse_pca(Zc, a, k=2)
        nz = (np.abs(W) > 1e-8).sum(axis=0)
        fev = subspace_fev(W, cov)
        rows.append({"alpha": a, "n_nonzero_pc1": int(nz[0]),
                     "n_nonzero_pc2": int(nz[1]), "fev": round(fev, 4)})
        loadings[a] = W
    df = pd.DataFrame(rows)

    # knee: the point furthest from the chord joining the two extremes
    pts = df.sort_values("n_nonzero_pc1")[["n_nonzero_pc1", "fev", "alpha"]].to_numpy()
    xs = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min())
    ys = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min())
    p0, p1 = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
    chord = p1 - p0
    dist = np.abs(np.cross(np.broadcast_to(chord, (len(xs), 2)),
                           np.c_[xs, ys] - p0)) / np.linalg.norm(chord)
    astar = float(pts[int(np.argmax(dist)), 2])
    return df, {"astar": astar, "loadings": loadings}


def sparse_bootstrap(Z: np.ndarray, alpha: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    hits = np.zeros(Z.shape[1])
    for _ in range(BOOT):
        take = rng.integers(0, len(Z), len(Z))
        Zb = Z[take]
        Zc = Zb - Zb.mean(axis=0)
        W, _ = sparse_pca(Zc, alpha, k=1)
        hits += (np.abs(W[:, 0]) > 1e-8).astype(float)
    return hits / BOOT


# ------------------------------------------------------- criterion B -------
def forward_select(X: np.ndarray, y: np.ndarray, cv) -> list[tuple[int, float, list[int]]]:
    """Greedy forward selection; returns [(k, mean balanced acc, chosen cols)]."""
    remaining = list(range(X.shape[1]))
    chosen: list[int] = []
    curve = []
    for _ in range(min(KMAX, X.shape[1])):
        scores = []
        for j in remaining:
            cols = chosen + [j]
            s = cross_val_score(rf(), X[:, cols], y, cv=cv,
                                scoring="balanced_accuracy").mean()
            scores.append(s)
        b = int(np.argmax(scores))
        chosen.append(remaining.pop(b))
        curve.append((len(chosen), float(np.max(scores)), chosen.copy()))
    return curve


def forward_curve_global(X, y) -> list[tuple[int, float, list[int]]]:
    return forward_select(X, y, StratifiedKFold(5, shuffle=True, random_state=42))


def nested_forward(X, y) -> dict:
    from sklearn.metrics import f1_score, balanced_accuracy_score
    outer = StratifiedKFold(5, shuffle=True, random_state=7)
    per_k_pred = {k: np.full(len(y), "", dtype=object) for k in range(1, KMAX + 1)}
    freq = np.zeros((X.shape[1], KMAX + 1))
    for tr, te in outer.split(X, y):
        cv_in = StratifiedKFold(3, shuffle=True, random_state=11)
        curve = forward_select(X[tr], y[tr], cv_in)
        for k, _, cols in curve:
            m = rf().fit(X[tr][:, cols], y[tr])
            pred = m.predict(X[te][:, cols])
            per_k_pred[k][te] = pred
            for c in cols:
                freq[c, k] += 1
    out = []
    for k in range(1, KMAX + 1):
        pred = per_k_pred[k]
        ok = pred != ""
        out.append({"k": k,
                    "macro_f1": round(float(f1_score(y[ok], pred[ok], average="macro")), 4),
                    "balanced_acc": round(float(balanced_accuracy_score(y[ok], pred[ok])), 4)})
    return {"curve": out, "freq": freq}


# ------------------------------------------------------------- figure ------
def figure(curve, df_sparse, astar, fev_dense, boot_freq, sup_freq, chosen_sup, flags):
    plt = C.setup_matplotlib()
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(7.3, 2.85))
    gs = GridSpec(1, 2, width_ratios=[0.92, 1.35], figure=fig, wspace=0.34)

    # (a) sparsity / fidelity frontier
    ax = fig.add_subplot(gs[0, 0])
    d = df_sparse.sort_values("n_nonzero_pc1")
    ax.plot(d.n_nonzero_pc1, d.fev, "-o", ms=3.2, lw=1.2, color=C.PALETTE["blue"])
    ax.axhline(fev_dense, color="0.55", lw=0.7, ls="--")
    ax.text(0.97, fev_dense + 0.004, f"dense PC1+2: {fev_dense:.2f}",
            fontsize=6.2, color="0.4", ha="right")
    sel = df_sparse.loc[df_sparse.alpha == astar].iloc[0]
    ax.scatter([sel.n_nonzero_pc1], [sel.fev], s=44, facecolor="none",
               edgecolor=C.PALETTE["vermilion"], lw=1.3, zorder=5)
    ax.annotate(f"$\\alpha^*$={astar:g}\n{int(sel.n_nonzero_pc1)} indicators",
                xy=(sel.n_nonzero_pc1, sel.fev), xytext=(0.06, 0.10),
                textcoords="axes fraction", fontsize=6.4,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4",
                                shrinkA=2, shrinkB=4))
    ax.set_xlabel("indicators retained in PC1")
    ax.set_ylabel("variance explained (PC1+2)")
    ax.set_title("(a) How much structure a shortlist keeps", loc="left")
    ax.invert_xaxis()
    ax.set_ylim(0.05, fev_dense + 0.035)

    # (b) selection frequency of the two criteria
    ax = fig.add_subplot(gs[0, 1])
    order = np.argsort(np.maximum(boot_freq, sup_freq))
    y = np.arange(len(order))
    for i, j in enumerate(order):
        col = ("both" if flags[j] == 2 else "blind" if flags[j] == 1 else
               "sup" if flags[j] == 3 else "none")
        ax.barh(i, boot_freq[j], height=0.44, color=C.PALETTE["blue"] if col in ("both", "blind")
                else "0.80", lw=0)
        ax.barh(i, sup_freq[j], height=0.26, left=0.0,
                color=C.PALETTE["orange"] if col in ("both", "sup") else "0.80", lw=0)
    ax.set_yticks(y, [C.LABELS[j] for j in order], fontsize=6.2)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("selection frequency over 200 bootstraps / 5 outer folds")
    ax.set_title("(b) Which indicators survive resampling", loc="left")
    ax.grid(axis="y", visible=False)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
         (C.PALETTE["blue"], C.PALETTE["orange"])]
    ax.legend(h, ["label-blind (sparse PCA, bootstrap)", "label-aware (forward, outer folds)"],
              frameon=False, fontsize=6.2, loc="lower right")

    C.save(fig, "F3_selection")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    Z = pd.read_csv(C.DATA_PROC / "country_level.csv", index_col="iso3")
    meta = C.load_country_meta().reindex(Z.index)
    y = C.income_short(meta["income"]).to_numpy()
    X = Z[C.LABELS].to_numpy()

    # ---- criterion A
    df_sparse, sp = sparse_curve(X)
    astar = sp["astar"]
    Wstar = sp["loadings"][astar]
    sparse_set = [l for l, w in zip(C.LABELS, Wstar[:, 0]) if abs(w) > 1e-8]
    fev_star = float(df_sparse.loc[df_sparse.alpha == astar, "fev"].iloc[0])
    fev_dense = float(df_sparse.fev.max())
    print(f"[A] alpha*={astar}  keeps {len(sparse_set)} indicators  "
          f"FEV {fev_star:.3f} (dense {fev_dense:.3f})  {time.time()-t0:.0f}s")

    boot_freq = sparse_bootstrap(X, astar)
    print(f"[A] bootstrap done, {time.time()-t0:.0f}s")

    # ---- criterion B
    sup_curve = forward_curve_global(X, y)
    sup12 = sup_curve[11][2] if len(sup_curve) >= 12 else sup_curve[-1][2]
    sup3 = sup_curve[2][2]
    sup_labels = [C.LABELS[j] for j in sup12]
    print(f"[B] global curve done, {time.time()-t0:.0f}s; 12-set={len(sup12)}")

    nested = nested_forward(X, y)
    sup_freq = (nested["freq"][:, 12] / 5.0) if nested["freq"].shape[1] > 12 else np.zeros(len(C.LABELS))
    print(f"[B] nested done, {time.time()-t0:.0f}s")

    jacc = len(set(sparse_set) & set(sup_labels)) / len(set(sparse_set) | set(sup_labels))

    # ---- flags for the figure: 1 blind only, 3 supervised only, 2 both
    flags = np.zeros(len(C.LABELS))
    for j, l in enumerate(C.LABELS):
        in_a, in_b = l in sparse_set, l in sup_labels
        flags[j] = 2 if (in_a and in_b) else 1 if in_a else 3 if in_b else 0

    figure(sup_curve, df_sparse, astar, fev_dense, boot_freq, sup_freq, sup12, flags)

    # ---- tables
    rows = []
    for j, l in enumerate(C.LABELS):
        rows.append({"indicator": l, "domain": C.DOMAIN[l],
                     "sparse_pca": "yes" if l in sparse_set else "",
                     "forward": "yes" if l in sup_labels else "",
                     "boot_freq": round(float(boot_freq[j]), 3),
                     "sup_freq": round(float(sup_freq[j]), 3),
                     "pc1_loading": round(float(Wstar[j, 0]), 3)})
    t = pd.DataFrame(rows).sort_values(["boot_freq", "sup_freq"], ascending=False)
    C.write_table(C.x_escape(t), "T3_selection",
                  "Indicator selection under the two criteria ($n=198$).",
                  "tab:selection", align="llccrr", note=(
                      "\\texttt{boot\\_freq}: share of 200 country-bootstrap resamples in which the "
                      "indicator keeps a non-zero PC1 loading at $\\alpha^*$. "
                      "\\texttt{sup\\_freq}: share of the 5 nested outer folds in which the "
                      "indicator is chosen among the first 12 by forward selection."))

    C.save_json({
        "alpha_star": astar,
        "sparse_n": len(sparse_set),
        "sparse_set": sparse_set,
        "sparse_fev": round(fev_star, 4),
        "dense_fev": round(fev_dense, 4),
        "sparse_table": df_sparse.round(4).to_dict("records"),
        "boot_freq": {l: round(float(f), 3) for l, f in zip(C.LABELS, boot_freq)},
        "boot_n": BOOT,
        "nested_curve": nested["curve"],
        "sup_freq12": {l: round(float(f), 2) for l, f in zip(C.LABELS, sup_freq)},
        "sup12": sup_labels,
        "sup3": [C.LABELS[j] for j in sup3],
        "global_curve": [{"k": k, "bal_acc": round(s, 4), "set": [C.LABELS[j] for j in c]}
                         for k, s, c in sup_curve],
        "jaccard": round(jacc, 4),
        "runtime_s": round(time.time() - t0, 1),
    }, C.METRICS / "E2_summary.json")

    C.save_json({"all28": C.LABELS, "sparse": sparse_set, "top12": sup_labels,
                 "top3": [C.LABELS[j] for j in sup3]},
                C.METRICS / "sets.json")
    print(f"Jaccard(sparse, forward12) = {jacc:.2f}; total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
