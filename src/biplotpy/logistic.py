"""Logistic biplots for binary, ordinal and nominal data.

Binary (Vicente-Villardón, Galindo & Blázquez-Zaballos, 2006; Demey et al., 2008):
    logit P(x_ij = 1) = b_j0 + sum_k a_ik b_jk
Ordinal (Vicente-Villardón & Hernández-Sánchez, 2014), cumulative logits:
    logit P(x_ij <= c) = tau_jc - sum_k a_ik b_jk
Nominal (Hernández-Sánchez & Vicente-Villardón, 2013), baseline-category logits:
    log P(x_ij = c) / P(x_ij = 1) = b_jc0 + sum_k a_ik b_jck

All three are fitted by penalized joint maximum likelihood (ridge penalty
lambda/2 on the row and slope parameters, which removes the indeterminacy of
scale between A and B) with L-BFGS and analytic gradients. The solution is
unique up to an orthogonal rotation of the latent space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps
from scipy.special import expit, log_expit, logsumexp


def _init_rows(X, dim, init):
    if isinstance(init, np.ndarray):
        return np.array(init, float)
    Xc = np.nan_to_num(X - np.nanmean(X, 0))
    sd = Xc.std(0)
    Xc = Xc / np.where(sd > 0, sd, 1)
    U, d, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :dim] * d[:dim] / np.sqrt(X.shape[0])


class LogisticBiplot:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def plot(self, *args, **kwargs):
        from .plot import plot_logistic
        return plot_logistic(self, *args, **kwargs)

    def __repr__(self):
        return (f"{self.title}: {self.n} rows, {self.p} variables, dim = {self.dim}, penalization = {self.penalization}\n"
                f"Objective = {self.objective:.4f}; global % correct = {self.global_percent:.2f}\n"
                f"{self.columns.round(3).to_string()}")


# --------------------------------------------------------------------------- binary
def _bin_obj(par, X, M, n, p, r, lam):
    A = par[:n * r].reshape(n, r)
    B = par[n * r:].reshape(p, r + 1)
    eta = B[:, 0] + A @ B[:, 1:].T
    H = expit(eta)
    ll = np.where(M, X * log_expit(eta) + (1 - X) * log_expit(-eta), 0.0)
    f = -ll.sum() + lam / 2 * ((A ** 2).sum() + (B[:, 1:] ** 2).sum())
    E = np.where(M, H - np.nan_to_num(X), 0.0)
    gA = E @ B[:, 1:] + lam * A
    gB = np.column_stack([E.sum(0), E.T @ A + lam * B[:, 1:]])
    return f, np.concatenate([gA.ravel(), gB.ravel()])


def binary_logistic_biplot(X, dim=2, penalization=0.01, init="svd", maxiter=5000, tol=1e-10,
                           row_names=None, col_names=None) -> LogisticBiplot:
    """Binary logistic biplot by penalized joint maximum likelihood.

    Minimizes ``-loglik + penalization/2 (|A|^2 + |B_slopes|^2)``. Missing values
    (NaN) are ignored. Returns row coordinates ``A``, column parameters ``B``
    (intercept and slopes), fitted probabilities, deviance tests against the
    intercept-only model, percentages of correct classification and a pseudo R2.
    """
    if isinstance(X, pd.DataFrame):
        row_names = row_names or [str(i) for i in X.index]
        col_names = col_names or [str(c) for c in X.columns]
        X = X.to_numpy(float)
    X = np.asarray(X, float)
    n, p = X.shape
    M = np.isfinite(X)
    A0 = _init_rows(X, dim, init)
    pbar = np.clip(np.nanmean(X, 0), 0.01, 0.99)
    B0 = np.column_stack([np.log(pbar / (1 - pbar)), np.zeros((p, dim)) + 0.01])
    par0 = np.concatenate([A0.ravel(), B0.ravel()])
    res = optimize.minimize(_bin_obj, par0, args=(X, M, n, p, dim, penalization), jac=True, method="L-BFGS-B",
                            options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-9, "maxcor": 30})
    A = res.x[:n * dim].reshape(n, dim)
    B = res.x[n * dim:].reshape(p, dim + 1)
    # orthogonalize the latent space (rotation that leaves the fit unchanged)
    U, d, Vt = np.linalg.svd(A, full_matrices=False)
    A, B = A @ Vt.T, np.column_stack([B[:, 0], B[:, 1:] @ Vt.T])
    eta = B[:, 0] + A @ B[:, 1:].T
    P = expit(eta)
    Xz = np.nan_to_num(X)
    pred = (P > 0.5).astype(float)
    acc = np.where(M, pred == Xz, np.nan)
    p0 = np.clip(np.nanmean(X, 0), 1e-12, 1 - 1e-12)
    Pc = np.clip(P, 1e-12, 1 - 1e-12)
    d_null = -2 * np.nansum(np.where(M, Xz * np.log(p0) + (1 - Xz) * np.log(1 - p0), 0), 0)
    d_fit = -2 * np.nansum(np.where(M, Xz * np.log(Pc) + (1 - Xz) * np.log(1 - Pc), 0), 0)
    dev = d_null - d_fit
    pvals = sps.chi2.sf(dev, dim)
    r2 = 1 - np.nansum(np.where(M, (Xz - P) ** 2, 0)) / np.nansum(np.where(M, Xz ** 2, 0))
    col_names = col_names or [f"V{j + 1}" for j in range(p)]
    row_names = row_names or [f"I{i + 1}" for i in range(n)]
    cols = pd.DataFrame(np.column_stack([B, dev, pvals, np.nanmean(acc, 0) * 100,
                                         1 - d_fit / np.where(d_null > 0, d_null, np.nan)]),
                        index=col_names, columns=["b0"] + [f"b{k + 1}" for k in range(dim)] +
                        ["Deviance", "p-value", "% correct", "McFadden R2"])
    return LogisticBiplot(title="Binary Logistic Biplot", n=n, p=p, dim=dim, penalization=penalization,
                          A=A, B=B, fitted=P, objective=res.fun, converged=res.success, columns=cols,
                          global_percent=np.nanmean(acc) * 100, row_percent=np.nanmean(acc, 1) * 100,
                          R2=r2, row_names=row_names, col_names=col_names)


# --------------------------------------------------------------------------- ordinal
def _ord_unpack(par, n, p, r, ncat):
    A = par[:n * r].reshape(n, r)
    o = n * r
    B = par[o:o + p * r].reshape(p, r)
    o += p * r
    taus = []
    for j in range(p):
        k = ncat[j] - 1
        th = par[o:o + k]
        o += k
        taus.append(np.concatenate([[th[0]], th[0] + np.cumsum(np.exp(th[1:]))]))
    return A, B, taus


def _ord_obj(par, X, n, p, r, ncat, lam, fixed_A=None):
    if fixed_A is not None:
        par_full = np.concatenate([fixed_A.ravel(), par])
    else:
        par_full = par
    A, B, taus = _ord_unpack(par_full, n, p, r, ncat)
    f = lam / 2 * ((A ** 2).sum() + (B ** 2).sum()) if fixed_A is None else lam / 2 * (B ** 2).sum()
    gA = lam * A if fixed_A is None else np.zeros_like(A)
    gB = lam * B
    gth = []
    for j in range(p):
        x = X[:, j]
        m = np.isfinite(x)
        c = x[m].astype(int)                         # categories 1..C
        eta = A[m] @ B[j]
        tau = np.concatenate([[-np.inf], taus[j], [np.inf]])
        up, lo = tau[c] - eta, tau[c - 1] - eta
        Fu, Fl = expit(up), expit(lo)
        prob = np.clip(Fu - Fl, 1e-300, None)
        f -= np.log(prob).sum()
        fu, fl = Fu * (1 - Fu), Fl * (1 - Fl)        # densities
        deta = (fu - fl) / prob                       # d(-log p)/d eta
        gA[m] += np.outer(deta, B[j])
        gB[j] += A[m].T @ deta
        # gradient wrt thresholds tau_k (k = 1..C-1)
        C = ncat[j]
        gt = np.zeros(C - 1)
        for k in range(1, C):
            gt[k - 1] = -np.sum(np.where(c == k, fu, 0) / prob) + np.sum(np.where(c == k + 1, fl, 0) / prob)
        # chain rule to the free parameters (th0, log increments)
        J = np.zeros((C - 1, C - 1))
        J[:, 0] = 1
        th_inc = np.diff(taus[j])
        for k in range(1, C - 1):
            J[k:, k] = th_inc[k - 1]
        gth.append(gt @ J)
    g = np.concatenate([gA.ravel(), gB.ravel()] + gth)
    if fixed_A is not None:
        g = g[n * r:]
    return f, g


def _ord_start(X, p, ncat):
    th = []
    for j in range(p):
        x = X[:, j][np.isfinite(X[:, j])]
        cum = np.array([np.mean(x <= k) for k in range(1, ncat[j])])
        cum = np.clip(cum, 0.02, 0.98)
        t = np.log(cum / (1 - cum))
        t = np.maximum.accumulate(t + np.arange(len(t)) * 1e-3)
        th.append(np.concatenate([[t[0]], np.log(np.maximum(np.diff(t), 1e-3))]))
    return th


def ordinal_logistic_fit(y, A, penalization=0.0):
    """Cumulative-logit (proportional odds) regression of one ordinal variable on
    fixed row coordinates ``A``: logit P(y <= c) = tau_c - A b. Equivalent to
    ``MASS::polr`` when ``penalization = 0``. Returns thresholds and slopes."""
    y = np.asarray(y, float)
    n, r = A.shape
    C = int(np.nanmax(y))
    X = y[:, None]
    th0 = _ord_start(X, 1, [C])[0]
    par0 = np.concatenate([np.zeros(r), th0])
    res = optimize.minimize(_ord_obj, par0, args=(X, n, 1, r, [C], penalization, A), jac=True, method="BFGS",
                            options={"gtol": 1e-10, "maxiter": 10000})
    _, B, taus = _ord_unpack(np.concatenate([A.ravel(), res.x]), n, 1, r, [C])
    return {"thresholds": taus[0], "slopes": B[0], "deviance": 2 * res.fun, "converged": res.success}


def ordinal_logistic_biplot(X, dim=2, penalization=0.01, init="svd", maxiter=10000, row_names=None,
                            col_names=None) -> LogisticBiplot:
    """Ordinal logistic biplot (cumulative logits) by penalized joint ML.
    Categories must be coded 1..C_j for each variable (NaN = missing)."""
    if isinstance(X, pd.DataFrame):
        row_names = row_names or [str(i) for i in X.index]
        col_names = col_names or [str(c) for c in X.columns]
        X = X.to_numpy(float)
    X = np.asarray(X, float)
    n, p = X.shape
    ncat = [int(np.nanmax(X[:, j])) for j in range(p)]
    A0 = _init_rows(X, dim, init)
    par0 = np.concatenate([A0.ravel(), np.full(p * dim, 0.1)] + _ord_start(X, p, ncat))
    res = optimize.minimize(_ord_obj, par0, args=(X, n, p, dim, ncat, penalization), jac=True, method="L-BFGS-B",
                            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxcor": 30})
    A, B, taus = _ord_unpack(res.x, n, p, dim, ncat)
    U, d, Vt = np.linalg.svd(A, full_matrices=False)
    A, B = A @ Vt.T, B @ Vt.T
    # fitted category probabilities and classification
    acc = np.full((n, p), np.nan)
    probs = []
    dev_null, dev_fit = np.zeros(p), np.zeros(p)
    for j in range(p):
        tau = np.concatenate([[-np.inf], taus[j], [np.inf]])
        eta = A @ B[j]
        Fc = expit(tau[None, :] - eta[:, None])
        Pj = np.diff(Fc, axis=1)
        probs.append(Pj)
        m = np.isfinite(X[:, j])
        c = X[m, j].astype(int)
        acc[m, j] = (Pj[m].argmax(1) + 1) == c
        freq = np.bincount(c, minlength=ncat[j] + 1)[1:] / m.sum()
        dev_null[j] = -2 * np.sum(np.log(np.clip(freq[c - 1], 1e-300, None)))
        dev_fit[j] = -2 * np.sum(np.log(np.clip(Pj[m, c - 1], 1e-300, None)))
    dev = dev_null - dev_fit
    col_names = col_names or [f"V{j + 1}" for j in range(p)]
    row_names = row_names or [f"I{i + 1}" for i in range(n)]
    cols = pd.DataFrame(np.column_stack([B, dev, sps.chi2.sf(dev, dim), np.nanmean(acc, 0) * 100,
                                         1 - dev_fit / dev_null]), index=col_names,
                        columns=[f"b{k + 1}" for k in range(dim)] + ["Deviance", "p-value", "% correct", "pseudo R2"])
    return LogisticBiplot(title="Ordinal Logistic Biplot", n=n, p=p, dim=dim, penalization=penalization, A=A, B=B,
                          thresholds=taus, fitted=probs, objective=res.fun, converged=res.success, columns=cols,
                          global_percent=np.nanmean(acc) * 100, row_names=row_names, col_names=col_names)


# --------------------------------------------------------------------------- nominal
def _nom_obj(par, Xi, n, p, r, ncat, lam, fixed_A=None):
    o = 0
    if fixed_A is None:
        A = par[:n * r].reshape(n, r); o = n * r
    else:
        A = fixed_A
    f = lam / 2 * (A ** 2).sum() if fixed_A is None else 0.0
    gA = lam * A if fixed_A is None else None
    gs = []
    for j in range(p):
        C = ncat[j]
        Bj = par[o:o + (C - 1) * (r + 1)].reshape(C - 1, r + 1); o += (C - 1) * (r + 1)
        m = Xi[:, j] >= 0
        eta = np.column_stack([np.zeros(m.sum()), Bj[:, 0] + A[m] @ Bj[:, 1:].T])
        lse = logsumexp(eta, axis=1)
        c = Xi[m, j]
        f -= np.sum(eta[np.arange(len(c)), c] - lse) - lam / 2 * (Bj[:, 1:] ** 2).sum()
        P = np.exp(eta - lse[:, None])
        Y = np.zeros_like(P); Y[np.arange(len(c)), c] = 1
        E = (P - Y)[:, 1:]
        gB = np.column_stack([E.sum(0), E.T @ A[m] + lam * Bj[:, 1:]])
        gs.append(gB.ravel())
        if gA is not None:
            gA[m] += E @ Bj[:, 1:]
    g = np.concatenate(([gA.ravel()] if gA is not None else []) + gs)
    return f, g


def _codes(X):
    cats, Xi = [], np.full(X.shape, -1, dtype=int)
    for j in range(X.shape[1]):
        col = pd.Series(X[:, j])
        levels = sorted(col.dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
        cats.append(levels)
        for k, lev in enumerate(levels):
            Xi[(col == lev).to_numpy(), j] = k
    return Xi, cats


def nominal_logistic_fit(y, A, penalization=0.0):
    """Multinomial (baseline-category) logistic regression of one nominal variable
    on fixed row coordinates; equivalent to ``nnet::multinom`` without decay."""
    Xi, cats = _codes(np.asarray(y, dtype=object)[:, None])
    n, r = A.shape
    C = len(cats[0])
    par0 = np.zeros((C - 1) * (r + 1))
    res = optimize.minimize(_nom_obj, par0, args=(Xi, n, 1, r, [C], penalization, A), jac=True, method="BFGS",
                            options={"gtol": 1e-10, "maxiter": 20000})
    return {"categories": cats[0], "coef": res.x.reshape(C - 1, r + 1), "deviance": 2 * res.fun,
            "converged": res.success}


def nominal_logistic_biplot(X, dim=2, penalization=0.01, init="indicator", maxiter=10000, row_names=None,
                            col_names=None) -> LogisticBiplot:
    """Nominal logistic biplot by penalized joint ML (first category = baseline).
    Each variable is a column of category labels."""
    if isinstance(X, pd.DataFrame):
        row_names = row_names or [str(i) for i in X.index]
        col_names = col_names or [str(c) for c in X.columns]
        X = X.to_numpy(object)
    X = np.asarray(X, dtype=object)
    n, p = X.shape
    Xi, cats = _codes(X)
    ncat = [len(c) for c in cats]
    if isinstance(init, np.ndarray):
        A0 = init
    else:
        Z = np.column_stack([(Xi[:, j] == k).astype(float) for j in range(p) for k in range(ncat[j])])
        A0 = _init_rows(Z, dim, "svd")
    par0 = np.concatenate([A0.ravel()] + [np.zeros((ncat[j] - 1) * (dim + 1)) for j in range(p)])
    res = optimize.minimize(_nom_obj, par0, args=(Xi, n, p, dim, ncat, penalization), jac=True, method="L-BFGS-B",
                            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxcor": 30})
    A = res.x[:n * dim].reshape(n, dim)
    o = n * dim
    Bs = []
    for j in range(p):
        k = (ncat[j] - 1) * (dim + 1)
        Bs.append(res.x[o:o + k].reshape(ncat[j] - 1, dim + 1)); o += k
    U, d, Vt = np.linalg.svd(A, full_matrices=False)
    A = A @ Vt.T
    Bs = [np.column_stack([B[:, 0], B[:, 1:] @ Vt.T]) for B in Bs]
    acc = np.full((n, p), np.nan)
    probs, dev = [], np.zeros(p)
    for j in range(p):
        eta = np.column_stack([np.zeros(n), Bs[j][:, 0] + A @ Bs[j][:, 1:].T])
        P = np.exp(eta - logsumexp(eta, axis=1)[:, None])
        probs.append(P)
        m = Xi[:, j] >= 0
        c = Xi[m, j]
        acc[m, j] = P[m].argmax(1) == c
        freq = np.bincount(c, minlength=ncat[j]) / m.sum()
        dev[j] = -2 * np.sum(np.log(freq[c])) + 2 * np.sum(np.log(np.clip(P[m, c], 1e-300, None)))
    col_names = col_names or [f"V{j + 1}" for j in range(p)]
    row_names = row_names or [f"I{i + 1}" for i in range(n)]
    dfs = [(ncat[j] - 1) * dim for j in range(p)]
    cols = pd.DataFrame({"categories": ncat, "Deviance": dev, "df": dfs, "p-value": sps.chi2.sf(dev, dfs),
                         "% correct": np.nanmean(acc, 0) * 100}, index=col_names)
    return LogisticBiplot(title="Nominal Logistic Biplot", n=n, p=p, dim=dim, penalization=penalization, A=A,
                          B=Bs, categories=cats, fitted=probs, objective=res.fun, converged=res.success,
                          columns=cols, global_percent=np.nanmean(acc) * 100, row_names=row_names,
                          col_names=col_names)
