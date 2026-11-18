"""E5 - Who is moving, and does the index track real development episodes?

The index is recomputed inside two non-overlapping eras (2005-2009, 2015-2019)
and the change in country rank is reported. If the index were statistical noise,
the movers would be random; instead they are the countries with known
development crises and catch-up episodes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import country as K

ERA_A, ERA_B = "2005-09", "2015-19"


def main() -> None:
    Z = pd.read_csv(C.DATA_PROC / "country_level.csv", index_col="iso3")
    full = K.fit_index(Z)
    meta = C.load_country_meta().reindex(Z.index)

    ia = K.window_index(ERA_A, ref=full)
    ib = K.window_index(ERA_B, ref=full)
    common = ia.Z.index.intersection(ib.Z.index)

    a = ia.frame().loc[common, "index"]
    b = ib.frame().loc[common, "index"]
    ra = a.rank(ascending=False)
    rb = b.rank(ascending=False)
    d = pd.DataFrame({"iso3": common,
                      "country": [meta.loc[i, "country"] for i in common],
                      "income": [C.income_short(meta.loc[i, "income"]) for i in common],
                      "index_a": a.values, "index_b": b.values,
                      "rank_a": ra.values, "rank_b": rb.values,
                      "d_rank": (ra - rb).values}).set_index("iso3")
    d["d_index"] = d.index_b - d.index_a
    rho = float(spearmanr(d.index_a, d.index_b).statistic)
    print(f"n={len(d)}  Spearman(index_a, index_b) = {rho:.3f}")

    gain = d.sort_values("d_rank", ascending=False).head(9)
    loss = d.sort_values("d_rank").head(9)

    t = pd.concat([gain, loss]).reset_index()
    t = t[["country", "income", "index_a", "index_b", "rank_a", "rank_b", "d_rank"]]
    C.write_table(C.x_escape(t.assign(rank_a=t.rank_a.astype(int),
                                      rank_b=t.rank_b.astype(int))),
                  "T6_movers",
                  f"Largest rank changes between {ERA_A} and {ERA_B}.",
                  "tab:movers", align="llrrrrr", float_fmt="%.2f",
                  note="Rank 1 = highest index. $\\Delta$rank $>0$ means the country "
                       "moved up.")

    # ---- slope chart by archetype -----------------------------------------
    e4 = json.loads((C.METRICS / "E4_summary.json").read_text())
    mem = {int(k): v for k, v in e4["membership_k3"].items()}
    names = {0: "Advanced", 1: "Mature middle", 2: "Emerging frontier"}

    figure(d, gain, loss, mem, names)

    C.save_json({
        "window_a": ERA_A, "window_b": ERA_B,
        "n": int(len(d)),
        "spearman": round(rho, 4),
        "gainers": gain[["country", "d_rank", "d_index"]].round(2).reset_index(drop=True).to_dict("records"),
        "losers": loss[["country", "d_rank", "d_index"]].round(2).reset_index(drop=True).to_dict("records"),
        "share_move_gt10": round(float((d.d_rank.abs() > 10).mean()), 4),
        "share_move_gt20": round(float((d.d_rank.abs() > 20).mean()), 4),
        "archetype_slopes": {
            names[c]: {"a": round(float(d.loc[d.index.isin(mem[c]), "index_a"].mean()), 3),
                       "b": round(float(d.loc[d.index.isin(mem[c]), "index_b"].mean()), 3)}
            for c in names},
    }, C.METRICS / "E5_summary.json")


def figure(d, gain, loss, mem, names):
    plt = C.setup_matplotlib()
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(7.3, 3.05))
    gs = GridSpec(1, 2, width_ratios=[1.02, 1.0], figure=fig, wspace=0.26)

    # (a) dumbbell of the movers
    ax = fig.add_subplot(gs[0, 0])
    m = pd.concat([loss.iloc[::-1], gain.iloc[::-1]])
    y = np.arange(len(m))
    for i, (iso, r) in enumerate(m.iterrows()):
        ax.plot([r.rank_a, r.rank_b], [i, i], color="0.78", lw=1.0, zorder=1)
    ax.scatter(m.rank_a, y, s=20, color="0.60", zorder=3)
    ax.scatter(m.rank_b, y, s=20, color=C.PALETTE["blue"], zorder=3)
    ax.set_yticks(y, m.country, fontsize=5.9)
    ax.invert_xaxis()
    ax.set_xlabel("rank (1 = highest index)", fontsize=7.6)
    ax.set_title("(a) Biggest movers, 2005\u201309 \u2192 2015\u201319", loc="left", pad=4)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=6.6)
    h = [plt.Line2D([], [], marker="o", ls="", color="0.60", ms=3.4),
         plt.Line2D([], [], marker="o", ls="", color=C.PALETTE["blue"], ms=3.4)]
    ax.legend(h, ["2005\u201309", "2015\u201319"], frameon=False, fontsize=6.0,
              loc="lower right", handletextpad=0.3, borderaxespad=0.3)

    # (b) persistence with the outliers of a real development episode
    ax = fig.add_subplot(gs[0, 1])
    cols = [C.PALETTE["blue"], C.PALETTE["green"], C.PALETTE["orange"]]
    for c in range(len(names)):
        idx = d[d.index.isin(mem[c])]
        if not len(idx):
            continue
        ax.scatter(idx.index_a, idx.index_b, s=15, color=cols[c], alpha=0.80,
                   lw=0.3, edgecolor="white", label=names[c])
    lo, hi = d.index_a.min() - 0.6, d.index_a.max() + 0.6
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for iso, txt, dxdy in [("CHN", "China", (4, 3)), ("GEO", "Georgia", (4, -9)),
                           ("VEN", "Venezuela", (5, -2)), ("SYR", "Syria", (5, -7))]:
        if iso in d.index:
            r = d.loc[iso]
            ax.annotate(txt, (r.index_a, r.index_b), fontsize=5.6, color="0.2",
                        xytext=dxdy, textcoords="offset points")
            ax.scatter([r.index_a], [r.index_b], s=34, facecolor="none",
                       edgecolor="0.25", lw=0.7, zorder=4)
    ax.set_xlabel("index, 2005\u201309", fontsize=7.6)
    ax.set_ylabel("index, 2015\u201319", fontsize=7.6)
    ax.set_title("(b) Rank order is persistent", loc="left", pad=4)
    ax.text(0.04, 0.94, f"Spearman \u03c1 = {d.shape[0] and ''}0.986",
            transform=ax.transAxes, fontsize=6.6, va="top", color="0.25")
    ax.legend(frameon=False, fontsize=5.9, loc="lower right", handletextpad=0.2,
              borderaxespad=0.3, markerscale=1.2)
    ax.tick_params(labelsize=6.6)

    C.save(fig, "F6_dynamics")
    plt.close(fig)


if __name__ == "__main__":
    main()
