"""Numerical helpers shared by the biplot methods."""
from __future__ import annotations

import math

import numpy as np

DBL_EPS = np.finfo(float).eps


def matrix_sqrt(S, tol=math.sqrt(DBL_EPS), inverse=False):
    """Square root (or inverse square root) of a symmetric PSD matrix via SVD,
    dropping singular values below ``tol`` (as ``matrixsqrt``/``matrixsqrtinv``
    in MultBiplotR)."""
    u, d, vt = np.linalg.svd(np.asarray(S, float))
    nz = d > tol
    f = 1 / np.sqrt(d[nz]) if inverse else np.sqrt(d[nz])
    return u[:, nz] @ np.diag(f) @ vt[nz]


def r_pretty(x, n=5, min_n=None, shrink_sml=0.75, high_u_bias=1.5, u5_bias=None):
    """Port of R's ``pretty()`` (``R_pretty`` in src/appl/pretty.c, eps.correct = 0)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, up = float(x.min()), float(x.max())
    if min_n is None:
        min_n = n // 3
    h = high_u_bias
    h5 = 0.5 + 1.5 * h if u5_bias is None else u5_bias
    ndiv = n
    dx = up - lo
    if dx == 0 and up == 0:
        cell = 1.0
        i_small = True
    else:
        cell = max(abs(lo), abs(up))
        U = 1 + (1 / (1 + h) if h5 >= 1.5 * h + 0.5 else 1.5 / (1 + h5))
        U *= max(1, ndiv) * DBL_EPS
        i_small = dx < cell * U * 3
    if i_small:
        if cell > 10:
            cell = 9 + cell / 10
        cell *= shrink_sml
        if min_n > 1:
            cell /= min_n
    else:
        cell = dx
        if ndiv > 1:
            cell /= ndiv
    if cell < 20 * np.finfo(float).tiny:
        cell = 20 * np.finfo(float).tiny
    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    ns = 2 * base
    if ns - cell < h * (cell - unit):
        unit = ns
        ns = 5 * base
        if ns - cell < h5 * (cell - unit):
            unit = ns
            ns = 10 * base
            if ns - cell < h * (cell - unit):
                unit = ns
    ns = math.floor(lo / unit + 1e-7)
    nu = math.ceil(up / unit - 1e-7)
    rounding_eps = 1e-10
    while ns * unit > lo + rounding_eps * unit:
        ns -= 1
    while nu * unit < up - rounding_eps * unit:
        nu += 1
    k = int(0.5 + nu - ns)
    if k < min_n:
        k = min_n - k
        if ns >= 0:
            nu += k // 2
            ns -= k // 2 + k % 2
        else:
            ns -= k // 2
            nu += k // 2 + k % 2
    return np.arange(ns, nu + 1) * unit
