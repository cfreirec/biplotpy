"""PLS regression with binary responses (MultBiplotR).

* ``plsr1_bin``: PLS regression of one binary response on continuous
  predictors (``PLSR1Bin``, algorithm 1 of Bastien, Esposito Vinzi & Tenenhaus
  2005): each component is built from the slopes of univariate ridge logistic
  regressions and the response is finally regressed on the components.
* ``binary_plsr``: PLS regression for binary data (``BinaryPLSR``), with
  binary predictors and binary responses, both modelled by logistic biplots
  whose row scores are shared; fitted by alternating conjugate gradients
  (R's ``optim(method = "CG")``) from random starts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .distances import ridge_logistic
from .roptim import optim_cg
from .rrng import RRNG
from .transforms import initial_transform, transform_name


def plsr1_bin(y, X, S=2, scaling=5, penalization=0.0, row_names=None, col_names=None) -> dict:
    """PLS regression of a binary response ``y`` (0/1) on ``X``
    (``MultBiplotR::PLSR1Bin`` with ``Algorithm = 1``).

    Component ``s``: for each predictor, a ridge logistic regression (penalty
    0.2, as in R) of ``y`` on the previous components and the predictor gives
    a slope; the normalized slopes are the weights ``w``, the scores are
    ``t = X_(s-1) w`` (deflated ``X``) and the loadings ``p = X't / t't``. The
    response is finally regressed on the components with ``penalization``.

    Returns scores ``T``, weights ``W``, loadings ``P``, the final logistic fit
    and the structure correlations between the predictors and the scores."""
    if isinstance(X, pd.DataFrame):
        col_names = col_names or [str(c) for c in X.columns]
        row_names = row_names or [str(i) for i in X.index]
    Xo = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("The response must be binary (0 or 1)")
    Xs = initial_transform(Xo, transform_name(scaling))["X"]
    n, J = Xs.shape
    T = np.zeros((n, S)); W = np.zeros((J, S)); P = np.zeros((J, S))
    w = np.zeros(J)
    for j in range(J):
        w[j] = ridge_logistic(y, Xs[:, j], 0.2, intercept=True)["beta"][1]
    w = w / np.sqrt(np.sum(w ** 2))
    t = Xs @ w
    T[:, 0] = t; W[:, 0] = w
    p = Xs.T @ t / np.sum(t ** 2); P[:, 0] = p
    X1 = Xs - np.outer(t, p)
    for s in range(1, S):
        for j in range(J):
            x = np.column_stack([T[:, :s], Xs[:, j]])
            w[j] = ridge_logistic(y, x, 0.2, intercept=True)["beta"][s + 1]
        w = w / np.sqrt(np.sum(w ** 2))
        t = X1 @ w
        T[:, s] = t; W[:, s] = w
        p = Xs.T @ t / np.sum(t ** 2); P[:, s] = p
        X1 = X1 - np.outer(t, p)
    fit = ridge_logistic(y, T, penalization, intercept=True)
    comps = [f"Comp. {k + 1}" for k in range(S)]
    col_names = col_names or [f"X{j + 1}" for j in range(J)]
    row_names = row_names or [f"I{i + 1}" for i in range(n)]
    structure = np.corrcoef(np.column_stack([Xo, T]), rowvar=False)[:J, J:]
    return {"method": "PLSR1 Binary", "scores": pd.DataFrame(T, index=row_names, columns=comps),
            "weights": pd.DataFrame(W, index=col_names, columns=comps),
            "loadings": pd.DataFrame(P, index=col_names, columns=comps),
            "structure": pd.DataFrame(structure, index=col_names, columns=comps), "fit": fit,
            "y_weights": fit["beta"], "scaled_X": Xs}


# --------------------------------------------------------------------------- BinaryPLSR
def _rsum(v):
    """``sum(v, na.rm = TRUE)`` of R: sequential accumulation in long double,
    in column-major order, skipping NaN (keeps the CG path close to R's)."""
    v = np.asarray(v, float).ravel(order="F")
    v = v[~np.isnan(v)]
    return float(np.cumsum(v.astype(np.longdouble))[-1]) if v.size else 0.0


def _mm(A, B):
    """``A %*% B`` with the summation order of the reference BLAS used by R
    (sequential over the inner index, in double precision)."""
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    C = np.zeros((A.shape[0], B.shape[1]))
    for l in range(A.shape[1]):
        C += A[:, l:l + 1] * B[l:l + 1, :]
    return C


def _sig(z):
    return 1 / (1 + np.exp(-1 * z))


def _cost(H, X):
    with np.errstate(divide="ignore", invalid="ignore"):
        return _rsum(-1 * X * np.log(H) - (1 - X) * np.log(1 - H))


def _J_b0(par, X):
    H = _sig(_mm(np.ones((X.shape[0], 1)), par[None, :]))
    return _cost(H, X) / 2


def _g_b0(par, X):
    H = _sig(_mm(np.ones((X.shape[0], 1)), par[None, :]))
    return _mm((H - X).T, np.ones((X.shape[0], 1))).ravel()


def _J_B(par, X, A, B, lam):
    r = A.shape[1] - 1
    B = B.copy(); B[:, r] = par
    H = _sig(_mm(A, B.T))
    return _cost(H, X) / 2 + lam * _rsum(B[:, 1:] ** 2) / 2


def _g_B(par, X, A, B, lam):
    r = A.shape[1] - 1
    B = B.copy(); B[:, r] = par
    E = _sig(_mm(A, B.T)) - X
    E[np.isnan(X)] = 0
    G = _mm(E.T, A) + lam * np.column_stack([np.zeros(B.shape[0]), B[:, 1:]])
    return G[:, r]


def _J_A(par, X, A, B, lam):
    r = B.shape[1] - 1
    A = A.copy(); A[:, r] = par
    H = _sig(_mm(A, B.T))
    return _cost(H, X) / 2 + lam * _rsum(A ** 2) / 2


def _g_A(par, X, A, B, lam):
    r = B.shape[1] - 1
    A = A.copy(); A[:, r] = par
    E = _sig(_mm(A, B.T)) - X
    E[np.isnan(X)] = 0
    return (_mm(E, B) + lam * A)[:, r]


def _scale(v):
    """R's ``scale()`` of a vector: ``colMeans`` (long double sum / n) and the root
    of the sum of squares (long double sum) over n - 1."""
    v = np.asarray(v, float).ravel()
    n = v.size
    m = float(np.cumsum(v.astype(np.longdouble))[-1] / n)
    c = v - m
    ss = float(np.cumsum((c * c).astype(np.longdouble))[-1])
    return c / np.sqrt(ss / max(1, n - 1))


def binary_plsr(Y, X, S=2, penalization=0.1, tol=5e-5, maxiter=100, seed=0, row_names=None,
                x_names=None, y_names=None) -> dict:
    """PLS regression for binary data (``MultBiplotR::BinaryPLSR``).

    ``X`` (n x J) and ``Y`` (n x K) are binary. Both are modelled by logistic
    biplots with intercepts ``p0`` and ``q0``; component ``s`` alternates, until
    the scores stabilize, ridge-penalized conjugate-gradient updates of the
    ``X`` loadings, the ``X`` scores, the ``Y`` loadings and the ``Y`` scores,
    the scores being standardized after each update (R's ``optim`` CG, ported in
    ``biplotpy.roptim``). Starting values are normal random numbers drawn
    with R's generator after ``set.seed(seed)``, so a given seed gives R's
    result. Returns scores, loadings, intercepts, fitted probabilities and the
    percentages of correct classification of ``Y``."""
    if isinstance(X, pd.DataFrame):
        x_names = x_names or [str(c) for c in X.columns]; row_names = row_names or [str(i) for i in X.index]
    if isinstance(Y, pd.DataFrame):
        y_names = y_names or [str(c) for c in Y.columns]
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if not np.all(np.isin(Y[np.isfinite(Y)], (0, 1))):
        raise ValueError("The response must be binary (0 or 1)")
    I, J = X.shape
    K = Y.shape[1]
    rng = RRNG(seed)
    p0 = optim_cg(_J_b0, _g_b0, rng.rnorm(J), args=(X,))["par"]
    q0 = optim_cg(_J_b0, _g_b0, rng.rnorm(K), args=(Y,))["par"]
    TT = np.ones((I, 1)); U = np.ones((I, 1)); P = p0[:, None]; Q = q0[:, None]
    iters = []
    for k in range(S):
        tt = rng.rnorm(I); TT = np.column_stack([TT, tt])
        u = rng.rnorm(I); U = np.column_stack([U, u])
        parP = rng.rnorm(J); P = np.column_stack([P, parP])
        parQ = rng.rnorm(K); Q = np.column_stack([Q, parQ])
        t = u
        error, it = 1.0, 0
        while error > tol and it < maxiter:
            it += 1
            told = t
            parP = optim_cg(_J_B, _g_B, parP, args=(X, TT, P, penalization))["par"]
            P[:, k + 1] = parP
            t = optim_cg(_J_A, _g_A, t, args=(X, TT, P, penalization))["par"]
            t = _scale(t)
            tt = t
            u = t
            U[:, k + 1] = u
            parQ = optim_cg(_J_B, _g_B, parQ, args=(Y, U, Q, penalization))["par"]
            Q[:, k + 1] = parQ
            u = optim_cg(_J_A, _g_A, u, args=(Y, U, Q, penalization))["par"]
            U[:, k + 1] = _scale(u)
            t = _scale(u)
            TT[:, k + 1] = t
            error = np.sum((told - t) ** 2)
        TT[:, k + 1] = tt
        iters.append(it)
    U, Q, P = U[:, 1:], Q[:, 1:], P[:, 1:]
    lin = TT @ np.column_stack([q0, Q]).T
    E = expit(lin)
    pred = (E > 0.5).astype(float)
    right = Y == pred
    comps = [f"Comp. {k + 1}" for k in range(S)]
    row_names = row_names or [f"I{i + 1}" for i in range(I)]
    x_names = x_names or [f"X{j + 1}" for j in range(J)]
    y_names = y_names or [f"Y{j + 1}" for j in range(K)]
    return {"method": "PLSR for binary data", "x_scores": pd.DataFrame(TT[:, 1:], index=row_names, columns=comps),
            "x_loadings": pd.DataFrame(P, index=x_names, columns=comps),
            "y_scores": pd.DataFrame(U, index=row_names, columns=comps),
            "y_loadings": pd.DataFrame(Q, index=y_names, columns=comps), "x_intercepts": p0, "y_intercepts": q0,
            "linear_terms": lin, "expected": E, "predictions": pred, "percent_correct": right.mean(),
            "percent_correct_cols": right.mean(0), "iterations": iters, "seed": seed}
