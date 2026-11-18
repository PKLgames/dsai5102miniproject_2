"""F3 (redraw) + ranking fidelity of the shortlists.

Reads results/metrics/E2_summary.json and results/metrics/sets.json, so it can be
re-run without repeating the expensive selection. Adds one number the report needs
and the selection pass does not compute: the Spearman correlation between the index
built from a shortlist alone and the index built from all 28 indicators.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import country as K

NAMES = {"sparse": "label-blind 12", "top12": "label-aware 12", "top3": "label-aware 3"}


def shortlist_index(Z: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = Z[cols].to_numpy()
    X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    sc = PCA(n_components=1, random_state=0).fit_transform(X)[:, 0]
    j = C.LABELS.index(K.ORIENT)
    if cols[j] in cols:                                  # same sign convention
        v = PCA(n_components=1, random_state=0).fit(X).components_[0]
        if v[cols.index(K.ORIENT)] < 0:
            sc = -sc
    return sc


def main() -> None:
    Z = pd.read_csv(C.DATA_PROC / "country_level.csv", index_col="iso3")
    sets = json.loads((C.METRICS / "sets.json").read_text())
    e2 = json.loads((C.METRICS / "E2_summary.json").read_text())
    full = K.fit_index(Z).scores[:, 0]

    fid = {}
    for key, name in NAMES.items():
        cols = [c for c in sets[key] if c in C.LABELS]
        sc = shortlist_index(Z, cols)
        fid[key] = {"name": name, "k": len(cols),
                    "spearman": round(float(spearmanr(sc, full).statistic), 4)}
        print(f"{name:16s} k={len(cols):2d}  rho with full index {fid[key]['spearman']:.4f}")

    C.save_json({"fidelity": fid,
                 "spearman_sparse_vs_top12": round(float(spearmanr(
                     shortlist_index(Z, [c for c in sets["sparse"] if c in C.LABELS]),
                     shortlist_index(Z, [c for c in sets["top12"] if c in C.LABELS])).statistic), 4)},
                C.METRICS / "E2b_fidelity.json")

    figure(e2, sets)


def figure(e2: dict, sets: dict) -> None:
    plt = C.setup_matplotlib()

    st = pd.DataFrame(e2["sparse_table"])
    boot = e2["boot_freq"]
    sup = e2["sup_freq12"]
    labels = C.LABELS
    both = {l for l in labels if boot.get(l, 0) > 0.5 and sup.get(l, 0) > 0.5}
    score = {l: max(boot.get(l, 0), sup.get(l, 0)) for l in labels}
    order = sorted(labels, key=lambda l: -score[l])
    halves = [order[:14], order[14:]]

    fig = plt.figure(figsize=(7.35, 3.35))

    # (a) sparsity - fidelity frontier
    ax = fig.add_axes([0.072, 0.235, 0.175, 0.60])
    ax.plot(st.n_nonzero_pc1, st.fev, "-o", ms=3.0, lw=1.2, color=C.PALETTE["blue"])
    astar = e2["alpha_star"]
    row = st.loc[st.alpha == astar].iloc[0]
    ax.scatter([row.n_nonzero_pc1], [row.fev], s=46, facecolor="none",
               edgecolor=C.PALETTE["vermilion"], lw=1.4, zorder=5)
    ax.annotate(f"$\\alpha^*$={astar}\n{int(row.n_nonzero_pc1)} indicators",
                (row.n_nonzero_pc1, row.fev), xytext=(8, -22),
                textcoords="offset points", fontsize=6.2, color="0.2")
    ax.axhline(e2["dense_fev"], color="0.55", lw=0.8, ls="--")
    ax.text(16.0, e2["dense_fev"] + 0.014, f"all 28: {e2['dense_fev']:.3f}",
            fontsize=5.8, color="0.35")
    ax.set_xlabel("indicators retained", fontsize=7.4)
    ax.set_ylabel("variance explained (PC1+2)", fontsize=7.4)
    ax.set_title("(a) Sparsity costs structure", loc="left", fontsize=8.2, pad=4)
    ax.tick_params(labelsize=6.6)

    # (b) selection frequency, split over two columns of 14 rows
    cols = [[0.440, 0.235, 0.183, 0.60], [0.777, 0.235, 0.183, 0.60]]
    for h, (block, rect) in enumerate(zip(halves, cols)):
        ax = fig.add_axes(rect)
        y = np.arange(len(block))
        bf = np.array([boot.get(l, 0) for l in block])
        sf = np.array([sup.get(l, 0) for l in block])
        ax.barh(y + 0.20, bf, height=0.38, color=C.PALETTE["blue"], lw=0)
        ax.barh(y - 0.20, sf, height=0.38, color=C.PALETTE["orange"], lw=0)
        ax.set_yticks(y, block, fontsize=5.95)
        for lbl in ax.get_yticklabels():
            if lbl.get_text() in both:
                lbl.set_fontweight("bold")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.65, len(block) - 0.35)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=6.4)
        if h == 0:
            ax.set_title("(b) Which indicators survive resampling", loc="left", fontsize=8.2, pad=4)
        ax.set_xlabel("selection frequency", fontsize=6.9)

    h1 = plt.Rectangle((0, 0), 1, 1, color=C.PALETTE["blue"])
    h2 = plt.Rectangle((0, 0), 1, 1, color=C.PALETTE["orange"])
    fig.legend([h1, h2], ["label-blind (sparse PCA, 200 bootstraps)",
                          "label-aware (forward selection, 5 outer folds)"],
               frameon=False, fontsize=6.4, loc="lower center",
               ncol=2, bbox_to_anchor=(0.60, 0.005), handlelength=1.0,
               columnspacing=1.4)

    C.save(fig, "F3_selection")
    plt.close(fig)


if __name__ == "__main__":
    main()
