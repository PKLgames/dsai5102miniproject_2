"""Country-level analysis object: aggregation, imputation, PCA index.

Everything downstream (selection, classification, clustering, dynamics) works on
the country-level table produced here, so the unit of analysis is defined once.

    country profile = mean over the window's years of the log-transformed,
                      winsorised indicator values
                    -> k-NN imputation at country level (k=5)
                    -> z-score across countries
                    -> PCA
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

import common as C

ORIENT = "GDP per capita"      # sign convention: PC1 loads positively on this
EPS = 1e-8


def load_panel(years: list[int] | tuple[int, ...] | None = None) -> pd.DataFrame:
    X = pd.read_csv(C.DATA_PROC / "dataset.csv")
    if years is not None:
        X = X[X["year"].isin(list(years))]
    return X


def build_country_level(years=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (Z country-level z-scores, M imputed country means in log units)."""
    X = load_panel(years)
    M = X.groupby("iso3")[C.LABELS].mean()
    imp = KNNImputer(n_neighbors=C.KNN_K)
    Mi = pd.DataFrame(imp.fit_transform(M), index=M.index, columns=M.columns)
    Z = pd.DataFrame(StandardScaler().fit_transform(Mi),
                     index=Mi.index, columns=Mi.columns)
    return Z, Mi


def years_of(tag: str) -> list[int]:
    """'2005-09' -> [2005..2009]."""
    a, b = tag.split("-")
    a = int(a)
    b = int(b) if len(b) == 4 else int(a // 100 * 100 + int(b))
    return list(range(a, b + 1))


@dataclass
class Index:
    Z: pd.DataFrame          # standardised country matrix
    pca: PCA
    loadings: np.ndarray     # d x k, sign-aligned
    scores: np.ndarray       # n x k
    evr: np.ndarray

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "iso3": self.Z.index,
            "index": self.scores[:, 0],
        })
        if self.scores.shape[1] > 1:
            df["index2"] = self.scores[:, 1]
        df["rank"] = df["index"].rank(ascending=False).astype(int)
        return df.set_index("iso3")

    def loading_series(self, comp: int = 0) -> pd.Series:
        return pd.Series(self.loadings[:, comp], index=C.LABELS, name="loading")


def fit_index(Z: pd.DataFrame, orient_ref: np.ndarray | None = None,
              k: int = 2) -> Index:
    """PCA on the country matrix, with a deterministic sign convention."""
    X = Z.to_numpy()
    pca = PCA(n_components=k, random_state=0).fit(X)
    load, sc = pca.components_.T.copy(), pca.transform(X)
    if orient_ref is not None:
        # flip so the index agrees with the reference orientation
        if np.corrcoef(sc[:, 0], orient_ref)[0, 1] < 0:
            load[:, 0] *= -1
            sc[:, 0] *= -1
    else:
        j = C.LABELS.index(ORIENT)
        for c in range(k):
            if load[j, c] < 0:
                load[:, c] *= -1
                sc[:, c] *= -1
    return Index(Z=Z, pca=pca, loadings=load, scores=sc,
                 evr=pca.explained_variance_ratio_)


def window_index(tag: str, ref: "Index | None" = None) -> Index:
    """Index computed on one time window, sign-aligned with the reference."""
    Z, _ = build_country_level(years_of(tag))
    idx = fit_index(Z)
    if ref is not None:
        idx = align_to(idx, ref)
    return idx


def align_to(idx: Index, ref: Index) -> Index:
    """Sign-align idx's PC1 with ref's PC1 (they may have been fit separately)."""
    common = idx.Z.index.intersection(ref.Z.index)
    a = idx.frame().loc[common, "index"].to_numpy()
    b = ref.frame().loc[common, "index"].to_numpy()
    if np.corrcoef(a, b)[0, 1] < 0:
        idx.loadings[:, 0] *= -1
        idx.scores[:, 0] *= -1
        if idx.scores.shape[1] > 1:
            idx.scores[:, 1] *= -1
            idx.loadings[:, 1] *= -1
    return idx
