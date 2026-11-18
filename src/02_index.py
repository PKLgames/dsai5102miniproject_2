"""E1 - A one-dimensional development index from 28 indicators, and how stable it is.

Steps
  1. aggregate the 2005-2022 panel to one profile per country (n=198)
  2. PCA -> PC1 = the development index; PC1+PC2 = the map used later
  3. stability: bootstrap the country set, recompute on four sub-windows,
     and leave one indicator out (28 refits)
Writes data/processed/country_level.csv, results/metrics/E1_summary.json,
results/tables/T{1,2}, results/figures/F2_index.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import country as K

WINDOWS = ["2005-09", "2010-14", "2015-19", "2015-22", "2005-22"]
BOOT = 300


def main() -> None:
    Z, Mi = K.build_country_level(range(C.YEAR_MIN, C.YEAR_MAX + 1))
    Z.to_csv(C.DATA_PROC / "country_level.csv")
    idx = K.fit_index(Z, k=6)
    meta = C.load_country_meta().reindex(Z.index)

    evr = idx.evr
    print(f"n={len(Z)} countries, d={Z.shape[1]} indicators")
    print("EVR:", np.round(evr, 4), f"PC1 {evr[0]:.4f}  PC1+2 {evr[:2].sum():.4f}")

    # ---- country ranking ---------------------------------------------------
    fr = idx.frame()
    fr["country"] = meta["country"]
    fr["region"] = meta["region"]
    fr["income"] = meta["income"]
    fr = fr.sort_values("index", ascending=False)
    fr.to_csv(C.TABLES / "T1_index_ranking.csv", index_label="iso3")

    # ---- loadings table ----------------------------------------------------
    load = pd.DataFrame({
        "indicator": C.LABELS,
        "domain": [C.DOMAIN[l] for l in C.LABELS],
        "pc1": idx.loadings[:, 0],
        "pc2": idx.loadings[:, 1],
    })
    load["abs_pc1"] = load.pc1.abs()
    load = load.sort_values("abs_pc1", ascending=False)
    C.write_table(C.x_escape(load[["indicator", "domain", "pc1", "pc2"]]),
                  "T2_loadings",
                  "PC1 and PC2 loadings ($n=198$ countries, 2005--2022 means).",
                  "tab:loadings", align="llrr", float_fmt="%.3f")

    # ---- stability: bootstrap the countries --------------------------------
    rng = np.random.default_rng(0)
    boot_evr, boot_rho = [], []
    X = Z.to_numpy()
    base = idx.scores[:, 0]
    for _ in range(BOOT):
        take = rng.integers(0, len(X), len(X))
        p = K.fit_index(Z.iloc[take].reset_index(drop=True))
        boot_evr.append(p.evr[0])
        # correlation with the full-sample index on the duplicated rows
        boot_rho.append(spearmanr(p.scores[:, 0], base[take]).statistic)
    boot_evr = np.array(boot_evr)
    boot_rho = np.array(boot_rho)

    # ---- stability: sub-windows -------------------------------------------
    win_idx = {w: K.window_index(w, ref=idx) for w in WINDOWS}
    rho_win = np.ones((len(WINDOWS), len(WINDOWS)))
    for i, a in enumerate(WINDOWS):
        for j, b in enumerate(WINDOWS):
            if j <= i:
                continue
            ca = win_idx[a].Z.index.intersection(win_idx[b].Z.index)
            x = win_idx[a].frame().loc[ca, "index"]
            y = win_idx[b].frame().loc[ca, "index"]
            r = spearmanr(x, y).statistic
            rho_win[i, j] = rho_win[j, i] = r
    off = rho_win[~np.eye(len(WINDOWS), dtype=bool)]
    win_evr = {w: float(win_idx[w].evr[0]) for w in WINDOWS}

    # ---- stability: leave one indicator out --------------------------------
    loio = []
    for j, lab in enumerate(C.LABELS):
        cols = [c for c in C.LABELS if c != lab]
        Xs = Z[cols].to_numpy()
        # re-standardise (a dropped column changes nothing else) then PCA
        p = K.fit_index(pd.DataFrame(Xs, index=Z.index, columns=cols))
        loio.append(spearmanr(p.scores[:, 0], base).statistic)
    loio = np.array(loio)

    # ---- Figure F2 ---------------------------------------------------------
    figure(fr, load, evr, win_evr, rho_win)

    C.save_json({
        "n_countries": int(len(Z)),
        "n_indicators": int(Z.shape[1]),
        "evr_6": [round(float(v), 4) for v in evr],
        "pc1_share": round(float(evr[0]), 4),
        "pc1_pc2_share": round(float(evr[:2].sum()), 4),
        "eigenvalues": [round(float(v), 4) for v in idx.pca.explained_variance_],
        "pc1_share_ci": [round(float(np.percentile(boot_evr, 2.5)), 4),
                         round(float(np.percentile(boot_evr, 97.5)), 4)],
        "pc1_share_boot_sd": round(float(boot_evr.std()), 4),
        "bootstrap_spearman_median": round(float(np.median(boot_rho)), 4),
        "bootstrap_spearman_min": round(float(boot_rho.min()), 4),
        "window_pc1_share": {k: round(v, 4) for k, v in win_evr.items()},
        "window_spearman_min": round(float(off.min()), 4),
        "window_spearman_median": round(float(np.median(off)), 4),
        "window_spearman_matrix": {"windows": WINDOWS,
                                   "values": np.round(rho_win, 4).tolist()},
        "loio_spearman_min": round(float(loio.min()), 4),
        "loio_spearman_median": round(float(np.median(loio)), 4),
        "loio_worst_indicator": C.LABELS[int(np.argmin(loio))],
        "index_range": [round(float(base.min()), 3), round(float(base.max()), 3)],
        "top5": fr.head(5)[["country", "index"]].round(3).values.tolist(),
        "bottom5": fr.tail(5)[["country", "index"]].round(3).values.tolist(),
        "spearman_index_gdp": round(float(spearmanr(
            fr["index"], Mi["GDP per capita"].reindex(fr.index)).statistic), 4),
    }, C.METRICS / "E1_summary.json")
    print(f"PC1 95% CI [{np.percentile(boot_evr,2.5):.3f}, {np.percentile(boot_evr,97.5):.3f}]"
          f"  window rho min {off.min():.4f}  LOIO rho min {loio.min():.4f}")


def figure(fr, load, evr, win_evr, rho_win):
    plt = C.setup_matplotlib()
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(7.3, 4.95))
    gs = GridSpec(2, 3, height_ratios=[0.80, 1.95], figure=fig,
                  hspace=0.50, wspace=0.62)

    # (a) scree
    ax = fig.add_subplot(gs[0, 0])
    k = 6
    ax.bar(np.arange(1, k + 1), evr[:k] * 100, color=C.PALETTE["blue"], width=0.62)
    ax.plot(np.arange(1, k + 1), np.cumsum(evr[:k]) * 100, "-o", ms=2.6, lw=0.9,
            color=C.PALETTE["vermilion"])
    ax.axhline(100 / 28, color="0.5", lw=0.6, ls=":")
    ax.set_xticks(range(1, k + 1))
    ax.set_xlabel("component")
    ax.set_ylabel("variance explained (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Scree", loc="left")
    ax.annotate(f"{evr[0]*100:.1f}%", xy=(1.28, evr[0] * 100),
                xytext=(2.0, evr[0] * 100 - 20), fontsize=7,
                color=C.PALETTE["blue"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45"))

    # (b) index by income group
    ax = fig.add_subplot(gs[0, 1:])
    groups = [g for g in C.INCOME_ORDER if g in set(fr.income)][::-1]
    data = [fr.loc[fr.income == g, "index"].values for g in groups]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", lw=0.8),
                    flierprops=dict(ms=2.2, mfc="0.4", mec="none"),
                    boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5),
                    capprops=dict(lw=0.5))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(C.INCOME_COLORS[g])
        patch.set_alpha(0.55)
    ax.set_yticks(range(1, len(groups) + 1),
                  [C.income_short(g) for g in groups], fontsize=7)
    for i, g in enumerate(groups, start=1):
        ax.annotate(f"n={int((fr.income == g).sum())}", xy=(1.005, i),
                    xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", fontsize=6.2, color="0.35",
                    ha="left", va="center", annotation_clip=False)
    ax.set_xlabel("development index (PC1)")
    ax.set_title("(b) Does the index reproduce the income groups?", loc="left")
    ax.grid(axis="y", visible=False)

    # (c) PC1 loadings by domain
    ax = fig.add_subplot(gs[1, :])
    order = np.argsort(load.pc1.values)
    vals = load.pc1.values[order]
    labs = load.indicator.values[order]
    doms = load.domain.values[order]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=[C.DOMAIN_COLORS[d] for d in doms], height=0.72, lw=0)
    ax.axvline(0, color="0.3", lw=0.6)
    ax.set_yticks(y, labs, fontsize=6.4)
    ax.set_xlabel("PC1 loading")
    ax.set_xlim(-0.34, 0.30)
    ax.set_title("(c) What the index is made of", loc="left")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C.DOMAIN_COLORS[d]) for d in C.DOMAINS]
    ax.legend(handles, C.DOMAINS, frameon=False, fontsize=6.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.19), ncol=6, handlelength=0.9,
              handleheight=0.7, columnspacing=1.1)

    C.save(fig, "F2_index")
    plt.close(fig)


if __name__ == "__main__":
    main()
