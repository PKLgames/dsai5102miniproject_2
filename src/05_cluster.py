"""E4 - Do the countries fall into policy archetypes?

K-means on the (PC1, PC2) map, k=2..6. We do not choose k by silhouette alone:
we also ask whether the partition survives resampling the countries (bootstrap
ARI), because a typology that is not stable cannot be used for targeting.

Writes results/metrics/E4_summary.json, results/tables/T5_archetypes,
results/figures/F5_archetypes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import country as K

KS = [2, 3, 4, 5, 6]
BOOT = 200
K_MAIN = 3


def kmeans(Z2: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z2)


def bootstrap_ari(Z2: np.ndarray, k: int) -> np.ndarray:
    rng = np.random.default_rng(2)
    base = kmeans(Z2, k)
    out = []
    for _ in range(BOOT):
        take = rng.integers(0, len(Z2), len(Z2))
        lab = kmeans(Z2[take], k, seed=int(rng.integers(1e6)))
        out.append(adjusted_rand_score(base[take], lab))
    return np.array(out)


def main() -> None:
    Z = pd.read_csv(C.DATA_PROC / "country_level.csv", index_col="iso3")
    idx = K.fit_index(Z)
    meta = C.load_country_meta().reindex(Z.index)
    y = C.income_short(meta["income"]).to_numpy()
    P2 = idx.scores[:, :2]

    stats = []
    for k in KS:
        lab = kmeans(P2, k)
        bs = bootstrap_ari(P2, k)
        stats.append({"k": k,
                      "silhouette": float(silhouette_score(P2, lab)),
                      "ari_income": float(adjusted_rand_score(lab, y)),
                      "boot_ari_mean": float(bs.mean()),
                      "boot_ari_p05": float(np.percentile(bs, 5)),
                      "boot_ari_min": float(bs.min())})
        print(f"k={k}  sil={stats[-1]['silhouette']:.3f}  "
              f"ARI_income={stats[-1]['ari_income']:.3f}  "
              f"boot ARI {bs.mean():.3f} [{np.percentile(bs,5):.3f}]")

    lab = kmeans(P2, K_MAIN)
    # order clusters by mean index so the naming is stable
    fr = idx.frame()
    means = pd.Series(fr["index"].values).groupby(lab).mean().sort_values(ascending=False)
    remap = {old: new for new, old in enumerate(means.index)}
    lab = np.array([remap[l] for l in lab])
    names = {0: "Advanced", 1: "Mature middle", 2: "Emerging frontier"}

    # ---- profile tables ----------------------------------------------------
    prof = pd.DataFrame({"iso3": Z.index, "cluster": lab, "income": y,
                         "index": fr["index"].values})
    prof["country"] = [meta.loc[i, "country"] for i in prof.iso3]
    prof["region"] = [meta.loc[i, "region"] for i in prof.iso3]

    Zz = Z.to_numpy()
    # a low z is only "bad" for indicators that load positively on the index
    sign = np.sign(idx.loadings[:, 0])
    rows = []
    for c in range(K_MAIN):
        m = lab == c
        inc = prof.loc[m, "income"].value_counts()
        adj = pd.Series(Zz[m].mean(axis=0) * sign, index=C.LABELS).sort_values()
        rows.append({
            "archetype": names[c],
            "n": int(m.sum()),
            "mean_index": float(prof.loc[m, "index"].mean()),
            "high": int(inc.get("High", 0)),
            "upper_mid": int(inc.get("Upper middle", 0)),
            "lower_mid": int(inc.get("Lower middle", 0)),
            "low": int(inc.get("Low", 0)),
            "gaps": ", ".join(adj.index[:3]),
            "strengths": ", ".join(adj.index[-2:]),
            "examples": ", ".join(prof.loc[m].sort_values("index").iloc[
                np.linspace(0, m.sum() - 1, min(5, m.sum())).astype(int)]["country"].tolist()),
        })
    t5 = pd.DataFrame(rows)
    C.write_table(C.x_escape(t5), "T5_archetypes",
                  "The three development archetypes ($k=3$, bootstrapped from $B=200$ resamples).",
                  "tab:archetypes", align="lrrrrrrrll",
                  note="\\texttt{gaps}: the three indicators on which the archetype sits "
                       "furthest below the global mean after orienting every indicator by the "
                       "sign of its index loading, so that higher means more developed; "
                       "\\texttt{strengths}: the two on which it sits furthest above. The "
                       "orientation is statistical, not normative: CO$_2$ emissions per capita "
                       "and renewable energy share load with the income gradient, so a "
                       "'gap' there is not a policy deficit. \\texttt{examples}: members "
                       "sampled across the index range.")

    prof.to_csv(C.TABLES / "T5a_membership.csv", index=False)
    pd.DataFrame(stats).to_csv(C.TABLES / "T4a_cluster_selection.csv", index=False)

    figure(fr, lab, P2, stats, names, meta)

    C.save_json({
        "k_selection": stats,
        "k_main": K_MAIN,
        "archetypes": t5.round(3).to_dict("records"),
        "ari_income_k3": stats[1]["ari_income"],
        "boot_ari_k3": [stats[1]["boot_ari_mean"], stats[1]["boot_ari_p05"]],
        "crosstab_k3": pd.crosstab(pd.Series(lab, name="cluster"),
                                   pd.Series(y, name="income")).to_dict(),
        "membership_k3": prof.groupby("cluster")["iso3"].apply(list).to_dict(),
        "cluster_mean_index": prof.groupby("cluster")["index"].mean().round(3).to_dict(),
    }, C.METRICS / "E4_summary.json")


def figure(fr, lab, P2, stats, names, meta):
    plt = C.setup_matplotlib()
    from matplotlib.gridspec import GridSpec

    cols = [C.PALETTE["blue"], C.PALETTE["green"], C.PALETTE["orange"],
            C.PALETTE["purple"], C.PALETTE["vermilion"]]

    fig = plt.figure(figsize=(7.3, 4.75))
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.42], width_ratios=[1.5, 1.0],
                  figure=fig, hspace=0.50, wspace=0.30)

    # (a) the map
    ax = fig.add_subplot(gs[0, 0])
    for c in range(len(names)):
        m = lab == c
        ax.scatter(P2[m, 0], P2[m, 1], s=15, color=cols[c], alpha=0.82,
                   lw=0.3, edgecolor="white", label=names[c])
    ax.set_xlabel("development index (PC1)", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.set_title("(a) Three archetypes in the index plane", loc="left", pad=4)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", handletextpad=0.2,
              borderaxespad=0.2, markerscale=1.3)
    ax.tick_params(labelsize=7)

    off = {"USA": (3, 4), "CHN": (4, -7), "IND": (5, -2), "NGA": (4, -6),
           "BRA": (4, 3), "JPN": (3, 5), "ETH": (-9, 4)}
    for iso, dxdy in off.items():
        if iso in fr.index:
            i = list(fr.index).index(iso)
            ax.annotate(iso, (P2[i, 0], P2[i, 1]), fontsize=6.0, color="0.2",
                        xytext=dxdy, textcoords="offset points")

    # (c) stability of k
    ax = fig.add_subplot(gs[0, 1])
    st = pd.DataFrame(stats)
    ax.plot(st.k, st.boot_ari_mean, "-o", ms=3, lw=1.3, color=C.PALETTE["blue"],
            label="bootstrap ARI")
    ax.plot(st.k, st.silhouette, "-s", ms=2.6, lw=1.0, color=C.PALETTE["orange"],
            label="silhouette")
    ax.axvline(K_MAIN, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("k", fontsize=8)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(KS)
    ax.set_title("(c) Which k to trust", loc="left", pad=4)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", handletextpad=0.3)
    ax.tick_params(labelsize=7)

    # (b) indicator profile of each archetype, full width
    ax = fig.add_subplot(gs[1, :])
    Z = pd.read_csv(C.DATA_PROC / "country_level.csv", index_col="iso3")
    Zz = Z.to_numpy()
    M = np.vstack([Zz[lab == c].mean(axis=0) for c in range(len(names))])
    order = np.argsort(-np.ptp(M, axis=0))
    im = ax.imshow(M[:, order], cmap="RdBu_r", vmin=-1.15, vmax=1.15, aspect="auto")
    ax.set_yticks(range(len(names)), [names[c] for c in range(len(names))], fontsize=7.6)
    ax.set_xticks(range(len(order)), [C.LABELS[j] for j in order], rotation=90,
                  fontsize=7.0)
    ax.set_title("(b) What separates the archetypes (mean z-score)", loc="left", pad=4)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.010)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("mean z", fontsize=6.5)

    C.save(fig, "F5_archetypes")
    plt.close(fig)


if __name__ == "__main__":
    main()
