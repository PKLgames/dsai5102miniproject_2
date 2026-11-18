"""S1 - Clean, transform, impute and standardise; produce the analysis tables.

Run:  .venv/bin/python src/01_preprocess.py
Writes:
    data/processed/dataset.csv        analysis-ready panel, original units (log1p applied)
    data/processed/dataset_std.csv    z-scored version used by every model
    data/processed/data_dictionary.csv / .md
    results/figures/F1_data_overview.pdf
    results/metrics/preprocess_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

WINSOR = (0.01, 0.99)


def load_panel():
    raw = C.load_raw_panel()
    cty = C.load_countries()
    raw = raw[(raw.year >= C.YEAR_MIN) & (raw.year <= C.YEAR_MAX)].copy()
    # drop country-years that are mostly empty
    keep = raw[C.LABELS].notna().sum(axis=1) >= C.MIN_ROW_COVERAGE * len(C.LABELS)
    dropped_rows = int((~keep).sum())
    raw = raw[keep].copy()
    return raw, cty, dropped_rows


def main() -> None:
    raw, cty, dropped_rows = load_panel()
    print(f"window {C.YEAR_MIN}-{C.YEAR_MAX}: {len(raw):,} country-years "
          f"({dropped_rows} dropped for < {C.MIN_ROW_COVERAGE:.0%} coverage)")

    X = raw[C.LABELS].copy()
    missing_before = X.isna().mean()

    # 1. drop over-missing indicators
    keep_cols = missing_before[missing_before <= C.MAX_COL_MISSING].index.tolist()
    dropped_cols = [c for c in C.LABELS if c not in keep_cols]
    X = X[keep_cols]
    if dropped_cols:
        print(f"dropped {len(dropped_cols)} indicators > {C.MAX_COL_MISSING:.0%} missing: {dropped_cols}")

    # 2. log-transform skewed magnitude variables
    for c in X.columns:
        if C.LOG.get(c) and (X[c].dropna() > 0).all():
            X[c] = np.log1p(X[c])

    # 3. winsorise at 1%/99% to stop a handful of tax havens and city-states
    #    from setting the principal axes
    lo, hi = X.quantile(WINSOR[0]), X.quantile(WINSOR[1])
    n_clipped = int(((X < lo) | (X > hi)).to_numpy().sum())
    X = X.clip(lower=lo, upper=hi, axis=1)

    # 4. impute (k-NN on the standardised scale) and standardise
    imp = KNNImputer(n_neighbors=C.KNN_K)
    Xs = StandardScaler().fit_transform(imp.fit_transform(X))
    Z = pd.DataFrame(Xs, index=X.index, columns=X.columns)

    out_X = X.copy()
    out_X.insert(0, "year", raw.year.values)
    out_X.insert(0, "iso3", raw.iso3.values)
    out_X["country"] = cty.country.reindex(out_X.iso3).values
    out_X["region"] = cty.region.reindex(out_X.iso3).values
    out_X["income"] = cty.income.reindex(out_X.iso3).values
    out_X.to_csv(C.DATA_PROC / "dataset.csv", index=False)

    out_Z = Z.copy()
    out_Z.insert(0, "year", raw.year.values)
    out_Z.insert(0, "iso3", raw.iso3.values)
    out_Z.to_csv(C.DATA_PROC / "dataset_std.csv", index=False)

    # 5. data dictionary
    dd = pd.DataFrame([{
        "indicator": c,
        "wdi_code": next(i[0] for i in C.INDICATORS if i[1] == c),
        "domain": C.DOMAIN[c], "unit": C.UNIT[c],
        "transform": "log1p + winsor(1,99) + z" if C.LOG.get(c) else "winsor(1,99) + z",
        "missing_before": round(float(missing_before[c]), 4),
        "missing_after": round(float(X[c].isna().mean()), 4),
    } for c in keep_cols])
    dd = dd.sort_values(["domain", "indicator"])
    dd.to_csv(C.DATA_PROC / "data_dictionary.csv", index=False)
    md = ["# Data dictionary", "",
          f"Window {C.YEAR_MIN}-{C.YEAR_MAX}; {len(out_X):,} country-year rows; "
          f"{len(keep_cols)} indicators; k-NN imputation (k={C.KNN_K}).", "",
          "| Indicator | Domain | Unit | Transform | Missing before |",
          "|---|---|---|---|---|"]
    for r in dd.itertuples():
        md.append(f"| {r.indicator} | {r.domain} | {r.unit} | {r.transform} | {r.missing_before:.1%} |")
    (C.DATA_PROC / "data_dictionary.md").write_text("\n".join(md))

    C.save_json({
        "window": [C.YEAR_MIN, C.YEAR_MAX],
        "rows": int(len(out_X)), "indicators": len(keep_cols),
        "countries": int(out_X.iso3.nunique()),
        "dropped_rows_low_coverage": dropped_rows,
        "dropped_indicators": dropped_cols,
        "winsorised_values": n_clipped,
        "missing_before_mean": round(float(missing_before.mean()), 4),
        "values_imputed": int(X.isna().to_numpy().sum()),
        "imputed_fraction": round(float(X.isna().to_numpy().mean()), 4),
        "knn_k": C.KNN_K,
        "checksum_dataset": __import__("hashlib").sha256(
            (C.DATA_PROC / "dataset.csv").read_bytes()).hexdigest(),
    }, C.METRICS / "preprocess_report.json")

    figure_overview(raw, X, keep_cols)
    print(f"\nprocessed: {out_X.shape[0]:,} x {len(keep_cols)} -> data/processed/")
    print(f"missing {missing_before.mean():.1%} -> {X.isna().mean().mean():.1%}")


def figure_overview(raw, X, keep_cols):
    """F1 - what the data actually looks like before any modelling."""
    plt = C.setup_matplotlib()
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.gridspec import GridSpec

    panel = raw.set_index(["iso3", "year"])[keep_cols]
    miss = panel.isna().mean() * 100
    miss.index = keep_cols

    fig = plt.figure(figsize=(7.4, 3.1))
    gs = GridSpec(1, 2, width_ratios=[1.45, 1.0], figure=fig, wspace=0.30)

    # (a) missingness by indicator
    ax = fig.add_subplot(gs[0, 0])
    order = miss.sort_values(ascending=True)
    cols_dom = [C.DOMAIN[i] for i in order.index]
    colmap = {d: c for d, c in zip(sorted(set(cols_dom)),
                                   [C.PALETTE["blue"], C.PALETTE["orange"],
                                    C.PALETTE["green"], C.PALETTE["vermilion"],
                                    C.PALETTE["purple"], C.PALETTE["sky"]])}
    ax.barh(np.arange(len(order)), order.values,
            color=[colmap[d] for d in cols_dom], height=0.72, linewidth=0)
    ax.set_yticks(np.arange(len(order)), order.index, fontsize=6.0)
    ax.set_xlabel("missing before imputation (%)", fontsize=8)
    ax.set_title("(a) Missingness by indicator", loc="left", fontsize=8.8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colmap.values()]
    ax.legend(handles, list(colmap.keys()), frameon=False, fontsize=6.0,
              loc="lower right", handlelength=1.0, handleheight=0.8,
              labelspacing=0.25, borderaxespad=0.2)

    # (b) coverage by year
    ax2 = fig.add_subplot(gs[0, 1])
    cov = panel.notna().mean(axis=1).groupby(level="year").mean() * 100
    ax2.plot(cov.index, cov.values, "-o", ms=3.2, lw=1.2, color=C.PALETTE["blue"])
    ax2.set_ylim(60, 100)
    ax2.set_xlabel("year", fontsize=8)
    ax2.set_ylabel("observed cells (%)", fontsize=8)
    ax2.set_title("(b) Panel coverage by year", loc="left", fontsize=8.8)
    ax2.set_xticks(range(int(cov.index.min()), int(cov.index.max()) + 1, 3))
    ax2.tick_params(labelsize=7)

    fig.tight_layout(w_pad=1.0)
    C.save(fig, "F1_data_overview")
    plt.close(fig)


if __name__ == "__main__":
    main()
