"""Sparse PCA by elastic-net deflation (shared by S4 and S5).

    max_W  tr(W' S W) - lam * ||W||_1     s.t.  W'W = I

is non-convex and non-smooth. Following Zou, Hastie & Tibshirani (2006) we
regress the current principal component score on the data with an elastic-net
penalty, deflate, and repeat.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet

L1_RATIO = 1.0


def sparse_pca(Zc: np.ndarray, alpha: float, k: int = 2, l1_ratio: float = L1_RATIO):
    """Return (loadings d x k with unit-norm columns, scores n x k)."""
    comps, scores = [], []
    R = Zc.copy()
    for _ in range(k):
        v = PCA(n_components=1).fit(R).components_[0]      # dense direction
        y = R @ v
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                        max_iter=20000, tol=1e-6).fit(R, y)
        b = en.coef_.astype(float)
        if np.linalg.norm(b) < 1e-12:                      # over-regularised
            b = v.copy()
        b /= np.linalg.norm(b)
        comps.append(b)
        scores.append(R @ b)
        R = R - np.outer(R @ b, b)                         # deflation
    return np.column_stack(comps), np.column_stack(scores)


def subspace_fev(W: np.ndarray, cov: np.ndarray) -> float:
    """Variance captured by span(W); sparse W is not orthogonal, so re-orthonormalise."""
    Q, _ = np.linalg.qr(W)
    return float(np.trace(Q.T @ cov @ Q) / np.trace(cov))
