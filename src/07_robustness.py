"""E6 - Do the three answers survive reasonable changes of method?

Every check here is a single number that a referee would ask for:
  * is the typology an artefact of K-means?          (Gaussian mixture, 28-D k-means)
  * is the shortlist an artefact of one alpha?       (neighbouring penalties)
  * is it driven by one region?                      (drop East Asia & Pacific)
  * does a 3-component index classify better?        (RF on PC1..PC3)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import country as K
from sparse_solver import sparse_pca

CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)


def sparse_set(Z: np.ndarray, alpha: float) -> set[str]:
    Zc = Z - Z.mean(axis=0)
    W, _ = sparse_pca(Zc, alpha, k=1)
    return {l for l, w in zip(C.LABELS, W[:, 0]) if abs(w) > 1e-8}


def main() -> None:
    Z = pd.read_csv(C.DATA_PROC / "country_level.csv", index_col="iso3")
    meta = C.load_country_meta().reindex(Z.index)
    y = C.income_short(meta["income"]).to_numpy()
    idx = K.fit_index(Z)
    P2 = idx.scores[:, :2]
    Zv = Z.to_numpy()

    e2 = json.loads((C.METRICS / "E2_summary.json").read_text())
    base_set = set(e2["sparse_set"])
    km = KMeans(3, n_init=20, random_state=0).fit_predict(P2)

    checks = []

    # 1. mixture model instead of k-means
    gm = GaussianMixture(3, covariance_type="full", n_init=10, random_state=0)
    gm_lab = gm.fit_predict(P2)
    a = adjusted_rand_score(km, gm_lab)
    checks.append(("Typology: Gaussian mixture instead of K-means", f"ARI {a:.3f}",
                   "same partition" if a > 0.8 else "partial overlap"))

    # 2. cluster in the full 28-dimensional space
    km28 = KMeans(3, n_init=20, random_state=0).fit_predict(Zv)
    a = adjusted_rand_score(km, km28)
    checks.append(("Typology: full 28-D space instead of PC1-PC2", f"ARI {a:.3f}",
                   "same partition" if a > 0.8 else "partial overlap"))

    # 3. neighbouring penalties for the label-blind shortlist
    for al in (0.25, 0.50):
        s = sparse_set(Zv, al)
        j = len(base_set & s) / len(base_set | s)
        checks.append((f"Shortlist: $\\alpha$={al} instead of {e2['alpha_star']}",
                       f"{len(s)} kept, Jaccard {j:.2f}",
                       "stable" if j > 0.7 else "sensitive"))

    # 4. one region removed
    eap = meta["region"].str.contains("East Asia", na=False).to_numpy()
    s = sparse_set(Zv[~eap], e2["alpha_star"])
    j = len(base_set & s) / len(base_set | s)
    checks.append(("Shortlist: East Asia & Pacific removed", f"{len(s)} kept, Jaccard {j:.2f}",
                   "stable" if j > 0.7 else "sensitive"))

    # 5. a three-component index
    P3 = idx.scores[:, :3]
    f1_3pc = cross_val_score(RandomForestClassifier(400, random_state=0, n_jobs=1),
                             P3, y, cv=CV, scoring="f1_macro").mean()
    f1_28 = cross_val_score(RandomForestClassifier(400, random_state=0, n_jobs=1),
                            Zv, y, cv=CV, scoring="f1_macro").mean()
    checks.append(("Index: PC1--PC3 instead of 28 indicators",
                   f"macro-F1 {f1_3pc:.3f} vs {f1_28:.3f}",
                   "as good" if f1_3pc >= f1_28 - 0.01 else "worse"))

    t = pd.DataFrame(checks, columns=["Check", "Result", "Reading"])
    C.write_table(C.x_escape(t), "T7_robustness",
                  "Robustness of the three answers to reasonable changes of method.",
                  "tab:robustness", align="lll")

    for c in checks:
        print(" -", c[0], "->", c[1], "->", c[2])
    C.save_json({"checks": [{"check": c[0], "result": c[1], "reading": c[2]}
                            for c in checks],
                 "f1_3pc": round(float(f1_3pc), 4), "f1_28": round(float(f1_28), 4)},
                C.METRICS / "E6_summary.json")


if __name__ == "__main__":
    main()
