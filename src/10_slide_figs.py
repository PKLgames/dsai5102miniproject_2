"""Slide-specific graphics: the same evidence as Figures 1-2, but sized for a room.

The report figures carry 28 stacked labels, which is right for print and wrong for
a projected slide. These two panels keep the message and drop the clutter.

Writes results/figures/S1_missing.png and S2_index.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import country as K

ORDER = ["High", "Upper middle", "Lower middle", "Low"]
DOMAIN_COL = C.DOMAIN_COLORS


def fig_missing() -> None:
    C.setup_matplotlib()
    dd = pd.read_csv(C.DATA_PROC / "data_dictionary.csv")
    dd = dd.sort_values("missing_before", ascending=False).head(12)
    dd = dd.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.9, 3.5))
    fig.subplots_adjust(left=0.42, right=0.97, top=0.86, bottom=0.16)
    y = np.arange(len(dd))
    cols = [DOMAIN_COL[d] for d in dd["domain"]]
    ax.barh(y, dd["missing_before"] * 100, color=cols, height=0.68,
            edgecolor="white", linewidth=0.5)
    for i, v in enumerate(dd["missing_before"] * 100):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=7.6, color="0.25")
    ax.set_yticks(y, dd["indicator"], fontsize=8.4)
    ax.set_xlim(0, 40)
    ax.set_xlabel("country-years missing before imputation (%)", fontsize=8.6)
    ax.tick_params(axis="x", labelsize=7.8)
    ax.set_title("The twelve sparsest of the 28 indicators", loc="left", fontsize=9.6)
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    C.save(fig, "S1_missing")
    plt.close(fig)


def fig_index() -> None:
    C.setup_matplotlib()
    Z, _ = K.build_country_level(range(C.YEAR_MIN, C.YEAR_MAX + 1))
    idx = K.fit_index(Z, k=6)
    meta = C.load_country_meta().reindex(Z.index)
    fr = idx.frame()
    inc = C.income_short(meta["income"]).reindex(fr.index)

    fig = plt.figure(figsize=(7.7, 4.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.72, 1.34], wspace=0.62,
                          left=0.115, right=0.985, top=0.90, bottom=0.20)
    ax = fig.add_subplot(gs[0, 0])
    data = [fr["index"].values[inc.values == g] for g in ORDER]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True,
                    medianprops=dict(color="0.15", lw=1.1),
                    flierprops=dict(marker="o", ms=2.6, mfc="0.45", mec="none"))
    for patch, g in zip(bp["boxes"], ORDER):
        patch.set_facecolor(C.INCOME_COLORS["Low income"] if g == "Low" else "white")
        patch.set_alpha(0.55)
        patch.set_edgecolor("0.25")
    ax.set_xlim(-7.8, 9.4)
    ax.set_yticks(range(1, 5), ORDER, fontsize=8.4)
    ax.set_xlabel("development index (PC1)", fontsize=8.6)
    ax.tick_params(axis="x", labelsize=7.8)
    ax.set_title("(a) It reproduces the income ordering", loc="left", fontsize=9.2)
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for i, g in enumerate(ORDER):
        ax.annotate(f"n={int((inc.values == g).sum())}", xy=(0.995, i + 1.30),
                    xycoords=("axes fraction", "data"), fontsize=7.2, color="0.45",
                    va="center", ha="right")

    ax2 = fig.add_subplot(gs[0, 1])
    lo = idx.loadings[:, 0]
    order = np.argsort(lo)
    keep = np.concatenate([order[:8], order[-10:]])
    keep = keep[np.argsort(lo[keep])]
    y = np.arange(len(keep))
    cols = [DOMAIN_COL[C.DOMAIN[C.LABELS[k]]] for k in keep]
    ax2.barh(y, lo[keep], color=cols, height=0.7, edgecolor="white", linewidth=0.4)
    ax2.set_yticks(y, [C.LABELS[k] for k in keep], fontsize=8.4)
    ax2.axvline(0, color="0.3", lw=0.8)
    ax2.set_xlabel("loading on PC1", fontsize=8.6)
    ax2.tick_params(axis="x", labelsize=7.8)
    ax2.set_xlim(-0.36, 0.30)
    ax2.set_title("(b) Wealth, health and connectivity load together",
                  loc="left", fontsize=9.2)
    ax2.grid(axis="x", color="0.9", linewidth=0.6)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in DOMAIN_COL.values()]
    fig.legend(handles, list(DOMAIN_COL), frameon=False, fontsize=7.4, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, 0.02), handlelength=1.0,
               columnspacing=1.1)
    C.save(fig, "S2_index")
    plt.close(fig)


if __name__ == "__main__":
    fig_missing()
    fig_index()
    print("ok")
