"""Weighted alternating least squares biplot (criss-cross regression) and the
log-frequency biplot of MultBiplotR (``CrissCross``, ``LogFrequencyBiplot``)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classical import ContinuousBiplot
from .rrng import RRNG
from .transforms import initial_transform, transform_name


def _wcor(d1, d2, w):
    s = np.sum(w)
    m1 = np.sum(d1 * w) / s
    m2 = np.sum(d2 * w) / s
    return (np.sum(d1 * d2 * w) / s - m1 * m2) / np.sqrt((np.sum(d1 ** 2 * w) / s - m1 ** 2) *
                                                         (np.sum(d2 ** 2 * w) / s - m2 ** 2))


def criss_cross(X, W=None, dim=2, a0=None, b0=None, maxiter=100, tol=1e-4, addsvd=True, lam=0.0, seed=None):
    """Weighted low-rank approximation ``X ~ A B'`` by alternating weighted
    least squares (Gabriel & Zamir 1979; ``MultBiplotR::CrissCross``).

    ``W``: non-negative weights (zero = missing). Starting values ``a0``/``b0``
    default to normal random numbers drawn with R's generator after
    ``set.seed(seed)`` (``seed=None`` uses numpy). Iterates until the change of
    the relative weighted residual sum of squares is below ``tol``; with
    ``addsvd`` the fitted matrix is decomposed by SVD to give the biplot
    markers. Returns row/column markers, weighted goodness of fit of each row
    and column by dimension (%) and the global fit."""
    X = np.array(X, float)
    n, p = X.shape
    W = np.ones((n, p)) if W is None else np.asarray(W, float)
    X[W == 0] = 0
    if a0 is None or b0 is None:
        if seed is None:
            g = np.random.default_rng()
            ra = lambda k: g.standard_normal(k)
        else:
            r = RRNG(seed)
            ra = r.rnorm
        if a0 is None:
            a0 = ra(n * dim).reshape(dim, n).T          # matrix(rnorm(n*dim), n, dim): filled by columns
        if b0 is None:
            b0 = ra(p * dim).reshape(dim, p).T
    a0 = np.asarray(a0, float); b0 = np.asarray(b0, float)
    gfit0 = np.sum(((X - a0 @ b0.T) * W) ** 2) / np.sum((X * W) ** 2)
    a, b = a0.copy(), b0.copy()
    err, it = 0.1, 0
    while err > tol and it < maxiter:
        it += 1
        for i in range(n):
            a[i] = np.linalg.pinv(b0.T @ np.diag(W[i] + lam) @ b0) @ b0.T @ np.diag(W[i]) @ X[i]
        for j in range(p):
            b[j] = np.linalg.pinv(a.T @ np.diag(W[:, j] + lam) @ a) @ a.T @ np.diag(W[:, j]) @ X[:, j]
        esp = a @ b.T
        gfit = np.sum(((X - esp) * W) ** 2) / np.sum((X * W) ** 2)
        err = abs(gfit0 - gfit)
        a0, b0, gfit0 = a.copy(), b.copy(), gfit
    esp = a @ b.T
    if addsvd:
        U, d, Vt = np.linalg.svd(esp, full_matrices=False)
        a = U[:, :dim] * np.sqrt(d[:dim])
        b = Vt[:dim].T * np.sqrt(d[:dim])
    rowfit = np.zeros((n, dim)); colfit = np.zeros((p, dim))
    for k in range(dim):
        e = np.outer(a[:, k], b[:, k])
        for i in range(n):
            rowfit[i, k] = _wcor(X[i], e[i], W[i]) ** 2
        for j in range(p):
            colfit[j, k] = _wcor(X[:, j], e[:, j], W[:, j]) ** 2
    return {"A": a, "B": b, "fitted": esp, "row_fit": rowfit * 100, "col_fit": colfit * 100,
            "goodness_of_fit": 1 - gfit0, "iterations": it, "weights": W}


def log_frequency_biplot(N, dim=2, scaling=2, logoffset=1.0, freqoffset=None, seed=None, maxiter=100,
                         tol=1e-4, row_names=None, col_names=None) -> ContinuousBiplot:
    """Log-frequency biplot (``MultBiplotR::LogFrequencyBiplot``): the table of
    frequencies ``N`` is transformed to ``x = log(N + logoffset)``, scaled
    (``scaling``: 1 raw, 2 subtract the global mean, 4 column centering, 5
    standardize columns, 6 row centering, 7 standardize rows) and approximated
    by a weighted ALS biplot (``criss_cross``) with weights ``x + freqoffset``,
    so that larger counts weigh more. ``seed`` fixes R's random start."""
    if isinstance(N, pd.DataFrame):
        row_names = row_names or [str(i) for i in N.index]; col_names = col_names or [str(c) for c in N.columns]
    N = np.asarray(N, float)
    freqoffset = logoffset if freqoffset is None else freqoffset
    x = np.log(N + logoffset)
    Wt = x + freqoffset
    t = transform_name(scaling)
    if t in ("Double centering", "Divide by the column means and center", "Normalized residuals from independence"):
        raise ValueError(f"{t} is not compatible with log-frequencies")
    Xs = initial_transform(x, t)["X"]
    cc = criss_cross(Xs, Wt, dim=dim, seed=seed, maxiter=maxiter, tol=tol)
    n, p = N.shape
    row_names = row_names or [f"I{i + 1}" for i in range(n)]
    col_names = col_names or [f"V{j + 1}" for j in range(p)]
    ev = np.linalg.svd(cc["fitted"], compute_uv=False) ** 2
    bip = ContinuousBiplot(
        title="Log-frequency biplot (weighted ALS)", alpha=0.5, dim=dim, transformation=t, scale_factor=1.0,
        row_coordinates=cc["A"], col_coordinates=cc["B"], row_names=row_names, col_names=col_names,
        row_contributions=cc["row_fit"], col_contributions=cc["col_fit"], eigenvalues=ev,
        inertia=ev / ev.sum() * 100, cum_inertia=np.cumsum(ev / ev.sum() * 100),
        goodness_of_fit=cc["goodness_of_fit"], iterations=cc["iterations"], weights=Wt, data=Xs, log_data=x,
        means=x.mean(0), deviations=x.std(0, ddof=1), gmean=x.mean())
    return bip
