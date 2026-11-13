"""Clustering and Disjoint Principal Component Analysis (CDPCA; Vichi &
Saporta, 2009; Macedo & Freitas, 2015). Port of ``biplotbootGUI::CDpca``.

Simultaneously partitions the I objects into P clusters (membership U) and
the J variables into Q disjoint groups (membership V), each group defining one
component (loadings A, non-zero only on its variables), maximizing the
between-cluster deviance in the reduced space, F = ||U Ybar||^2."""
from __future__ import annotations

import numpy as np


def _rand_mat(rng, d1, d2):
    U = np.zeros((d1, d2))
    U[:d2, :d2] = np.eye(d2)
    for j in range(d2, d1):
        U[j, rng.integers(d2)] = 1
    return U


def _lead(Xg, idx, I):
    """Loading vector of a group of variables (first eigenvector, sign with positive sum)."""
    if len(idx) == 1:
        return np.ones(1)
    if I >= len(idx):
        w, v = np.linalg.eigh(Xg[:, idx].T @ Xg[:, idx]); a = v[:, -1]
    else:
        w, v = np.linalg.eigh(Xg[:, idx] @ Xg[:, idx].T)
        a = Xg[:, idx].T @ v[:, -1] / np.sqrt(w[-1])
    return a if a.sum() >= 0 else -a


def _loadings(Xg, V, J, Q, I):
    A = np.zeros((J, Q))
    for j in range(Q):
        idx = np.where(V[:, j] == 1)[0]
        if len(idx):
            A[idx, j] = _lead(Xg, idx, I)
    return A


def _one_run(Xs, P, Q, U, V, tol, maxit):
    I, J = Xs.shape
    Xbar = (U.T @ Xs) / U.sum(0)[:, None]
    Xg = U @ Xbar
    A = _loadings(Xg, V, J, Q, I)
    A0 = A.copy()
    Ybar = Xbar @ A
    F0 = np.sum((U @ Ybar) ** 2)
    Fmax = F0
    it = 0
    while True:
        it += 1
        Y = Xs @ A
        U = np.zeros((I, P))
        U[np.arange(I), ((Y[:, None, :] - Ybar[None, :, :]) ** 2).sum(2).argmin(1)] = 1
        su = U.sum(0)
        while (su == 0).any():                    # empty cluster: move half of the largest
            imax, imin = su.argmax(), su.argmin()
            sel = np.where(U[:, imax] == 1)[0][:int(su[imax]) // 2]
            U[sel, imin] = 1; U[sel, imax] = 0
            su = U.sum(0)
        Xbar = (U.T @ Xs) / U.sum(0)[:, None]
        Ybar = Xbar @ A
        Xg = U @ Xbar
        for j in range(J):
            posmax = int(np.where(V[j] == 1)[0][0])
            for g in range(Q):
                V[j] = np.eye(Q)[g]
                if g != posmax:
                    for gg in (g, posmax):
                        idx = np.where(V[:, gg] == 1)[0]
                        A[:, gg] = 0
                        if len(idx):
                            A[idx, gg] = _lead(Xg, idx, I)
                Ybar = Xbar @ A
                F = np.sum((U @ Ybar) ** 2)
                if F > Fmax:
                    Fmax, posmax, A0 = F, g, A.copy()
                else:
                    A = A0.copy()
            V[j] = np.eye(Q)[posmax]
        Y = Xs @ A
        Ybar = Xbar @ A
        F = np.sum((U @ Ybar) ** 2)
        conv = F - F0
        if conv > tol:
            F0, A0 = F, A.copy()
        else:
            break
        if it == maxit:
            break
    return {"U": U, "V": V, "A": A, "F": F, "iter": it, "Xbar": Xbar}


def cdpca(X, P: int, Q: int, nstart: int = 20, tol: float = 1e-5, maxit: int = 100, seed=None, init=None) -> dict:
    """CDPCA with ``nstart`` random starts (``init`` = (U0, V0) for a single
    given start). Data are standardized with variance divided by I, as in the
    original. Components are ordered by explained variance."""
    X = np.asarray(X, float)
    I, J = X.shape
    Xs = (X - X.mean(0)) / X.std(0, ddof=1) * np.sqrt(I / (I - 1))
    rng = np.random.default_rng(seed)
    best = None
    starts = [init] if init is not None else [None] * nstart
    for st in starts:
        U0, V0 = (np.array(st[0], float), np.array(st[1], float)) if st is not None else \
            (_rand_mat(rng, I, P), _rand_mat(rng, J, Q))
        r = _one_run(Xs, P, Q, U0, V0, tol, maxit)
        if best is None or r["F"] > best["F"]:
            best = r
    U, V, A = best["U"], best["V"], best["A"]
    Y = Xs @ A
    Ybar = best["Xbar"] @ A
    var = np.var(Y, axis=0, ddof=1) * 100 / J
    o = np.argsort(-var, kind="stable")
    return {"U": U, "cluster": U.argmax(1) + 1, "V": V[:, o], "A": A[:, o], "Y": Y[:, o], "Ybar": Ybar[:, o],
            "F": best["F"], "iter": best["iter"], "var_percent": var[o],
            "bcdev_total": best["F"] / (I * J) * 100,
            "bcdev": np.sum((U @ Ybar) ** 2) / np.sum(Y ** 2) * 100,
            "Enorm": np.linalg.norm(Xs - U @ Ybar @ A.T) / I}
