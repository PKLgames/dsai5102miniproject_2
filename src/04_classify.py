"""E3 - Does a short list of indicators still identify the income group?

Four specifications of the same question:
    all28     everything we have
    sparse    the 12 label-blind indicators from E2 (no label leakage)
    top12     the 12 indicators a supervised search would pick
    top3      a three-indicator rapid screen

Three classifiers: multinomial logistic regression, random forest, gradient
boosting. Repeated stratified 5-fold CV, balanced accuracy + macro-F1 because
the smallest class (low income, n=25) is 2.7x smaller than the largest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SCORING = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
           "f1_macro": "f1_macro"}
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
ORDER = ["High", "Upper middle", "Lower middle", "Low"]


def models():
    return {
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, C=1.0)),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=0,
                                               n_jobs=1),
        "GradBoost": HistGradientBoostingClassifier(max_iter=300, random_state=0),
    }


def oof_confusion(model, X, y) -> np.ndarray:
    """Out-of-fold confusion matrix, summed over the 25 repeated folds."""
    cm = np.zeros((len(ORDER), len(ORDER)), dtype=int)
    for tr, te in CV.split(X, y):
        m = clone(model).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        cm += confusion_matrix(y[te], pred, labels=ORDER)
    return cm


def main() -> None:
    Z = pd.read_csv(C.DATA_PROC / "country_level.csv", index_col="iso3")
    meta = C.load_country_meta().reindex(Z.index)
    y = C.income_short(meta["income"]).to_numpy()
    sets = json.loads((C.METRICS / "sets.json").read_text())
    specs = {"all28": sets["all28"], "sparse": sets["sparse"],
             "top12": sets["top12"], "top3": sets["top3"]}
    pretty = {"all28": "All 28", "sparse": "Shortlist 12 (blind)",
              "top12": "Shortlist 12 (supervised)", "top3": "Rapid screen 3"}

    rows, detail = [], {}
    for key, cols in specs.items():
        X = Z[cols].to_numpy()
        for name, clf in models().items():
            res = cross_validate(clf, X, y, cv=CV, scoring=SCORING, n_jobs=1)
            r = {m: (float(res[f"test_{m}"].mean()), float(res[f"test_{m}"].std()))
                 for m in SCORING}
            rows.append({"set": pretty[key], "d": len(cols), "model": name,
                         "acc": r["accuracy"][0], "acc_sd": r["accuracy"][1],
                         "bal_acc": r["balanced_accuracy"][0],
                         "bal_acc_sd": r["balanced_accuracy"][1],
                         "f1": r["f1_macro"][0], "f1_sd": r["f1_macro"][1]})
            detail[f"{key}|{name}"] = r
            print(f"  {key:7s} {name:12s} acc={r['accuracy'][0]:.3f} "
                  f"bal={r['balanced_accuracy'][0]:.3f} f1={r['f1_macro'][0]:.3f}")

    t = pd.DataFrame(rows)
    C.write_table(
        t.assign(f1_sd=t.f1_sd), "T4_classification",
        "Income-group classification, repeated stratified 5-fold CV (5 repeats).",
        "tab:classification", align="llcrrrr", float_fmt="%.3f",
        note="\\texttt{f1} is macro-averaged; standard deviations are over the 25 folds.")

    # ---- per-class recall + confusion matrix for the headline model --------
    Xs = Z[specs["sparse"]].to_numpy()
    cm = oof_confusion(models()["RandomForest"], Xs, y)
    rec = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(ORDER)}

    X3 = Z[specs["top3"]].to_numpy()
    cm3 = oof_confusion(models()["RandomForest"], X3, y)

    figure(t, detail, cm, rec)

    C.save_json({
        "rows": t.round(4).to_dict("records"),
        "per_class_recall_sparse_rf": rec,
        "confusion_sparse_rf": cm.tolist(),
        "confusion_top3_rf": cm3.tolist(),
        "class_counts": {k: int(v) for k, v in pd.Series(y).value_counts().items()},
        "cv": {"splits": 5, "repeats": 5},
        "sets": {k: len(v) for k, v in specs.items()},
    }, C.METRICS / "E3_summary.json")


def figure(t, detail, cm, rec):
    plt = C.setup_matplotlib()
    from matplotlib.gridspec import GridSpec

    e2 = json.loads((C.METRICS / "E2_summary.json").read_text())
    nest = pd.DataFrame(e2["nested_curve"])
    glob = pd.DataFrame(e2["global_curve"])

    fig = plt.figure(figsize=(7.3, 2.95))
    gs = GridSpec(1, 3, width_ratios=[1.30, 1.22, 0.92], figure=fig, wspace=0.46)

    # (a) accuracy vs number of indicators
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(glob.k, glob.bal_acc, "-o", ms=3.0, lw=1.1, color="0.62",
            label="selected & scored on the same folds")
    ax.plot(nest.k, nest.balanced_acc, "-o", ms=3.0, lw=1.3,
            color=C.PALETTE["blue"], label="selection nested inside CV")
    ax.axhline(detail["all28|RandomForest"]["balanced_accuracy"][0],
               color=C.PALETTE["vermilion"], lw=0.9, ls="--")
    ax.text(14.3, detail["all28|RandomForest"]["balanced_accuracy"][0] + 0.006,
            "all 28 indicators", fontsize=6.2, color=C.PALETTE["vermilion"], ha="right")
    ax.set_xlabel("indicators used")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("(a) Accuracy against shortlist size", loc="left", fontsize=7.6)
    ax.set_xlim(0.5, 14.5)
    ax.set_ylim(0.755, 0.908)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left",
              handlelength=1.0, borderaxespad=0.25)

    # (b) grouped bars
    ax = fig.add_subplot(gs[0, 1])
    sets_order = ["All 28", "Shortlist 12 (blind)", "Shortlist 12 (supervised)",
                  "Rapid screen 3"]
    models_order = ["LogReg", "RandomForest", "GradBoost"]
    cols = {"LogReg": "0.45", "RandomForest": C.PALETTE["blue"],
            "GradBoost": C.PALETTE["sky"]}
    w = 0.26
    for i, m in enumerate(models_order):
        vals, errs = [], []
        for s in sets_order:
            row = t[(t.set == s) & (t.model == m)].iloc[0]
            vals.append(row.f1)
            errs.append(row.f1_sd)
        ax.bar(np.arange(len(sets_order)) + (i - 1) * w, vals, width=w,
               color=cols[m], label=m, yerr=errs, ecolor="0.3",
               capsize=1.2, error_kw=dict(lw=0.6))
    ax.set_xticks(range(len(sets_order)),
                  ["28", "12\nblind", "12\nsup.", "3"], fontsize=7)
    ax.set_ylim(0.60, 0.98)
    ax.set_ylabel("macro-F1")
    ax.set_title("(b) Macro-F1 by specification", loc="left", fontsize=7.6)
    ax.legend(frameon=False, fontsize=6.2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), handlelength=1.0, columnspacing=1.0)

    # (c) confusion matrix, sparse RF
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    labs = ["High", "Upper", "Lower", "Low"]
    ax.set_xticks(range(4), labs, fontsize=6.0)
    ax.set_yticks(range(4), labs, fontsize=6.0)
    ax.set_xlabel("predicted", fontsize=7.5)
    ax.set_ylabel("true", fontsize=7.5)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.4,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "0.2")
    ax.set_title("(c) 12-indicator RF", loc="left", fontsize=7.6)
    ax.grid(False)

    C.save(fig, "F4_classification")
    plt.close(fig)


if __name__ == "__main__":
    main()
