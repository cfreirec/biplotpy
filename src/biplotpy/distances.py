"""Distance-based biplots: proximities, principal coordinates (PCoA),
metric/non-metric MDS by SMACOF, external logistic biplots and canonical
analysis of distances.

* ``binary_distances`` / ``continuous_distances`` / ``gower_distances`` —
  the coefficients of ``MultBiplotR::BinaryDistances`` and
  ``ContinuousDistances`` (with three defects of the R code corrected, see
  ``FIXES``).
* ``pcoa`` — ``MultBiplotR::PrincipalCoordinates`` (= ``ade4::dudi.pco``).
* ``mds`` — symmetric SMACOF of ``smacof::mds`` (ratio, interval, ordinal
  with primary/secondary/tertiary ties).
* ``ridge_logistic`` / ``external_logistic_biplot`` —
  ``MultBiplotR::RidgeBinaryLogistic`` and ``ExternalBinaryLogisticBiplot``:
  binary variables projected onto a PCoA/MDS configuration by penalized
  logistic regression.
* ``canonical_distance_analysis`` — ``MultBiplotR::CanonicalDistanceAnalysis``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .correspondence import OrdinationBiplot

BINARY_COEFFICIENTS = ["Kulczynski", "Russell_and_Rao", "Jaccard", "Simple_Matching", "Anderberg",
                       "Rogers_and_Tanimoto", "Sorensen_Dice_and_Czekanowski", "Sneath_and_Sokal", "Hamman",
                       "Kulczynski2", "Anderberg2", "Ochiai", "S13", "Pearson_phi", "Yule", "Sorensen", "Dice"]
TRANSFORMATIONS = ["Identity", "1-S", "sqrt(1-S)", "-log(s)", "1/S-1", "sqrt(2(1-S))", "1-(S+1)/2", "1-abs(S)",
                   "1/(S+1)"]


def binary_similarity(X, Y=None, coefficient="Jaccard"):
    X = np.asarray(X, float)
    Y = X if Y is None else np.asarray(Y, float)
    a = Y @ X.T; b = Y @ (1 - X).T; c = (1 - Y) @ X.T; d = (1 - Y) @ (1 - X).T
    with np.errstate(divide="ignore", invalid="ignore"):
        S = {"Kulczynski": a / (b + c), "Russell_and_Rao": a / (a + b + c + d), "Jaccard": a / (a + b + c),
             "Simple_Matching": (a + d) / (a + b + c + d), "Anderberg": a / (a + 2 * (b + c)),
             "Rogers_and_Tanimoto": (a + d) / (a + 2 * (b + c) + d),
             "Sorensen_Dice_and_Czekanowski": a / (a + 0.5 * (b + c)),
             "Sneath_and_Sokal": (a + d) / (a + 0.5 * (b + c) + d), "Hamman": (a - (b + c) + d) / (a + b + c + d),
             "Kulczynski2": 0.5 * (a / (a + b) + a / (a + c)),
             "Anderberg2": 0.25 * (a / (a + b) + a / (a + c) + d / (c + d) + d / (b + d)),
             "Ochiai": a / np.sqrt((a + b) * (a + c)), "S13": a * d / np.sqrt((a + b) * (a + c) * (d + b) * (d + c)),
             "Pearson_phi": (a * d - b * c) / np.sqrt((a + b) * (a + c) * (d + b) * (d + c)),
             "Yule": (a * d - b * c) / (a * d + b * c), "Sorensen": 2 * a / (2 * a + b + c),
             "Dice": 2 * a / (2 * a + b + c)}[coefficient]
    return S


def _transform_similarity(S, transformation):
    f = {"Identity": lambda: S, "1-S": lambda: 1 - S, "sqrt(1-S)": lambda: np.sqrt(np.maximum(1 - S, 0)),
         "-log(s)": lambda: -np.log(S), "1/S-1": lambda: 1 / S - 1,
         "sqrt(2(1-S))": lambda: np.sqrt(np.maximum(2 * (1 - S), 0)), "1-(S+1)/2": lambda: 1 - (S + 1) / 2,
         "1-abs(S)": lambda: 1 - np.abs(S), "1/(S+1)": lambda: 1 / (S + 1)}
    with np.errstate(divide="ignore", invalid="ignore"):
        return f[transformation]()


def binary_distances(X, Y=None, coefficient="Jaccard", transformation=None):
    """Dissimilarities between the rows of a 0/1 matrix (``Y`` rows against
    ``X`` rows when given). Default transformation as MultBiplotR:
    sqrt(1-S) for most coefficients, 1-(S+1)/2 for Hamman, phi and Yule,
    1/(S+1) for Kulczynski."""
    if isinstance(coefficient, int):
        coefficient = BINARY_COEFFICIENTS[coefficient - 1]
    if transformation is None:
        transformation = {"Kulczynski": "1/(S+1)", "Hamman": "1-(S+1)/2", "Pearson_phi": "1-(S+1)/2",
                          "Yule": "1-(S+1)/2"}.get(coefficient, "sqrt(1-S)")
    return _transform_similarity(binary_similarity(X, Y, coefficient), transformation)


def continuous_distances(X, Y=None, coef="Pythagorean", r=1.0):
    """Dissimilarities between rows (``MultBiplotR::ContinuousDistances``)."""
    X = np.asarray(X, float)
    Y = X if Y is None else np.asarray(Y, float)
    Dif = Y[:, None, :] - X[None, :, :]
    Sm = Y[:, None, :] + X[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        if coef == "Pythagorean":
            return np.sqrt((Dif ** 2).sum(2))
        if coef == "Taxonomic":
            return np.sqrt((Dif ** 2 / r ** 2).sum(2))
        if coef == "City":
            return np.abs(Dif).sum(2)
        if coef == "Minkowski":
            return (np.abs(Dif) ** r).sum(2) ** (1 / r)
        if coef == "Divergence":
            return np.sqrt((Dif ** 2 / Sm ** 2).sum(2))
        if coef == "dif_sum":
            return (np.abs(Dif) / np.abs(Sm)).sum(2)
        if coef == "Canberra":
            return (np.abs(Dif) / (np.abs(Y)[:, None, :] + np.abs(X)[None, :, :])).sum(2)
        if coef == "Bray_Curtis":
            return np.abs(Dif).sum(2) / Sm.sum(2)
        if coef == "Soergel":
            return np.abs(Dif).sum(2) / np.maximum(Y[:, None, :], X[None, :, :]).sum(2)
        if coef == "Ware_Hedges":
            return (1 - np.minimum(Y[:, None, :], X[None, :, :]) / np.maximum(Y[:, None, :], X[None, :, :])).sum(2)
        if coef == "Gower":
            rng = np.vstack([X, Y]).max(0) - np.vstack([X, Y]).min(0)
            return (np.abs(Dif) / rng).sum(2)
    raise ValueError(f"unknown coefficient {coef}")


def gower_distances(df: pd.DataFrame):
    """Gower dissimilarity (1 - similarity) for mixed numeric / categorical data."""
    n = len(df)
    S = np.zeros((n, n)); W = np.zeros((n, n))
    for col in df.columns:
        v = df[col]
        ok = v.notna().to_numpy()
        w = np.outer(ok, ok).astype(float)
        if pd.api.types.is_numeric_dtype(v) and v.nunique() > 2:
            x = v.to_numpy(float); rng = np.nanmax(x) - np.nanmin(x)
            s = 1 - np.abs(x[:, None] - x[None, :]) / rng
        else:
            x = v.astype(str).to_numpy()
            s = (x[:, None] == x[None, :]).astype(float)
        S += np.nan_to_num(s) * w; W += w
    return 1 - S / W


# --------------------------------------------------------------------------- PCoA
def pcoa(D, dim=2, weights=None, row_names=None, sup_distances=None) -> OrdinationBiplot:
    """Principal coordinates of a dissimilarity matrix (Gower, 1966).

    Weighted version with the correct weighted centring (I - 1 w') on both
    sides. Fit statistics: stress-1 and -2, s-stress, R2 and Spearman's rho
    between the dissimilarities and the configuration distances."""
    D = np.asarray(D, float)
    n = D.shape[0]
    w = np.full(n, 1 / n) if weights is None else np.asarray(weights, float) / np.sum(weights)
    J = np.eye(n) - np.outer(np.ones(n), w)
    b = -0.5 * J @ D ** 2 @ J.T
    Bw = np.sqrt(w)[:, None] * b * np.sqrt(w)[None, :]
    ev, U = np.linalg.eigh(Bw)
    ev, U = ev[::-1], U[:, ::-1]
    pos = ev > 1e-10 * ev[0]
    g = U[:, pos] * np.sqrt(ev[pos]) / np.sqrt(w)[:, None]
    qual = g ** 2 / (g ** 2).sum(1, keepdims=True) * 100
    G = g[:, :dim]
    iu = np.triu_indices(n, 1)
    Dh = D[iu]
    Dc = np.sqrt(((G[:, None] - G[None]) ** 2).sum(2))[iu]
    res = OrdinationBiplot(title="Principal Coordinates Analysis", type="PCoA", dim=dim, eigenvalues=ev,
                           inertia=ev / ev.sum() * 100, row_coordinates=G, row_qualities=qual[:, :dim],
                           negative_eigenvalues=ev[ev < -1e-10 * ev[0]], centered_products=b,
                           stress1=np.sqrt(np.sum((Dc - Dh) ** 2) / np.sum(Dc ** 2)),
                           stress2=np.sqrt(np.sum((Dc - Dh) ** 2) / np.sum((Dc - Dc.mean()) ** 2)),
                           sstress1=np.sqrt(np.sum((Dc ** 2 - Dh ** 2) ** 2) / np.sum(Dc ** 4)),
                           rsq=np.corrcoef(Dh, Dc)[0, 1] ** 2, rho=sps.spearmanr(Dh, Dc)[0],
                           row_names=row_names or [f"I{i + 1}" for i in range(n)], col_coordinates=None)
    if sup_distances is not None:
        Ds = np.asarray(sup_distances, float)
        sc = 0.5 * (np.diag(b)[None, :] - Ds ** 2) @ g / ev[pos]
        res.sup_row_coordinates = sc[:, :dim]
    return res


# --------------------------------------------------------------------------- SMACOF (symmetric)
def _pava(y, w):
    """Weighted isotonic (non-decreasing) regression by pool-adjacent-violators."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            ws = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / ws if ws > 0 else (vals[-2] + vals[-1]) / 2
            c = cnt[-2] + cnt[-1]
            vals[-2:] = [v]; wts[-2:] = [ws]; cnt[-2:] = [c]
    return np.repeat(vals, cnt)


def _torgerson(D, p):
    n = D.shape[0]
    D2 = D ** 2
    B = -0.5 * (D2 - D2.mean(1)[:, None] - D2.mean(0)[None, :] + D2.mean())
    ev, U = np.linalg.eigh(B)
    ev, U = ev[::-1], U[:, ::-1]
    return U[:, :p] * np.sqrt(np.maximum(ev[:p], 0))


def mds(D, dim=2, type="ratio", ties="primary", weights=None, init=None, itmax=1000, eps=1e-6,
        row_names=None) -> OrdinationBiplot:
    """Symmetric SMACOF (``smacof::mds``): metric (ratio, interval) or
    non-metric (ordinal) multidimensional scaling with Torgerson start.
    Returns the configuration, disparities, stress-1 and stress per point."""
    D = np.asarray(D, float)
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    diss = D.T[np.tril_indices(n, -1)]                       # column-major lower triangle, as R's dist
    li, lj = np.tril_indices(n, -1)
    order_idx = np.lexsort((li, lj))                        # R stores dist by columns
    li, lj, diss = li[order_idx], lj[order_idx], D[li[order_idx], lj[order_idx]]
    m = len(diss)
    W = np.ones((n, n)) if weights is None else np.asarray(weights, float)
    wg = W[li, lj]
    nn = n * (n - 1) / 2

    def dist(x):
        return np.sqrt(((x[li] - x[lj]) ** 2).sum(1))

    def normq(v):
        return v * np.sqrt(nn / np.sum(wg * v ** 2))

    x = _torgerson(D, dim) if init is None else np.asarray(init, float)
    dhat = normq(diss)
    V = -np.zeros((n, n)); V[li, lj] = -wg; V[lj, li] = -wg
    V[np.diag_indices(n)] = -V.sum(1)
    Vinv = np.linalg.inv(V + 1 / n) - 1 / n
    d = dist(x)
    lb = np.sum(wg * d * dhat) / np.sum(wg * d ** 2)
    x, d = lb * x, lb * d
    sold = np.sum(wg * (dhat - d) ** 2) / nn
    # tie structure of the dissimilarities
    iord = np.lexsort((np.arange(m), diss))
    xs = diss[iord]
    starts = np.r_[0, np.where(np.diff(xs) != 0)[0] + 1]
    blocks = np.split(iord, starts[1:])
    xu = xs[starts]
    itel = 1
    while True:
        z = d < 1e-12
        bv = wg * dhat * (~z) / (d + z)
        Bm = np.zeros((n, n)); Bm[li, lj] = -bv; Bm[lj, li] = -bv
        Bm[np.diag_indices(n)] = -Bm.sum(1)
        y = Vinv @ (Bm @ x)
        e = dist(y)
        if type == "ratio":
            new = diss.copy()
        elif type == "interval":
            yb = np.array([np.sum(wg[b] * e[b]) / np.sum(wg[b]) for b in blocks])
            wb = np.array([np.sum(wg[b]) for b in blocks])
            Xb = np.column_stack([np.ones(len(xu)), xu - xu[0]])
            coef = np.linalg.solve(Xb.T @ (wb[:, None] * Xb), Xb.T @ (wb * yb))
            fitb = Xb @ coef
            new = np.empty(m)
            for b, v in zip(blocks, fitb):
                new[b] = v
        elif type == "ordinal" and ties == "primary":
            o = np.lexsort((e, diss))
            new = np.empty(m); new[o] = _pava(e[o], wg[o])
        elif type == "ordinal":
            yb = np.array([np.sum(wg[b] * e[b]) / np.sum(wg[b]) for b in blocks])
            wb = np.array([np.sum(wg[b]) for b in blocks])
            ycon = _pava(yb, wb)
            new = np.empty(m)
            for b, v, ym in zip(blocks, ycon, yb):
                new[b] = v if ties == "secondary" else e[b] + (v - ym)
        else:
            raise ValueError("type must be ratio, interval or ordinal")
        dhat = normq(new)
        snon = np.sum(wg * (dhat - e) ** 2) / nn
        if (sold - snon) < eps or itel == itmax:
            break
        x, d, sold = y, e, snon
        itel += 1
    Dh = np.zeros((n, n)); Dh[li, lj] = dhat; Dh = Dh + Dh.T
    Dc = np.sqrt(((y[:, None] - y[None]) ** 2).sum(2))
    res_mat = W * (Dh - Dc) ** 2
    np.fill_diagonal(res_mat, np.nan)
    spp = np.nanmean(res_mat, 0); spp = spp / spp.sum() * 100
    return OrdinationBiplot(title=f"MDS ({type})", type="MDS", dim=dim, row_coordinates=y, stress=np.sqrt(snon),
                            disparities=Dh, config_distances=Dc, spp=spp, niter=itel,
                            eigenvalues=np.var(y, axis=0) * n, inertia=np.var(y, axis=0) / np.var(y, axis=0).sum() * 100,
                            row_names=row_names or [f"I{i + 1}" for i in range(n)], col_coordinates=None)


# --------------------------------------------------------------------------- logistic regression on a configuration
def ridge_logistic(y, X, penalization=0.2, intercept=True, tol=1e-5, maxiter=100):
    """Ridge-penalized binary logistic regression by Newton-Raphson
    (``MultBiplotR::RidgeBinaryLogistic``): maximizes loglik - lambda * sum(beta_j^2),
    intercept not penalized. Returns coefficients and fit statistics."""
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    Xd = np.column_stack([np.ones(n), X]) if intercept else X

    def fit(Xm):
        m = Xm.shape[1]
        beta = np.zeros(m)
        Im = np.eye(m); Im[0, 0] = 0
        for _ in range(maxiter):
            mu = 1 / (1 + np.exp(-(Xm @ beta)))
            In = Xm.T @ ((mu * (1 - mu))[:, None] * Xm) + 2 * penalization * Im
            U = Xm.T @ (y - mu) - 2 * penalization * Im @ beta
            new = beta + np.linalg.pinv(In) @ U
            err = np.sum(np.abs(new - beta)); beta = new
            if err <= tol:
                break
        return beta
    b0 = fit(np.ones((n, 1)))
    beta = fit(Xd)
    p0 = 1 / (1 + np.exp(-b0[0])); p = 1 / (1 + np.exp(-(Xd @ beta)))
    dev0 = -2 * np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    dev = -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    dif = dev0 - dev; df = Xd.shape[1] - 1
    cs = 1 - np.exp(-dif / n)
    pred = (p >= 0.5).astype(float)
    return {"beta": beta, "fitted": p, "deviance": dev, "null_deviance": dev0, "dif": dif, "df": df,
            "p": sps.chi2.sf(dif, df), "cox_snell": cs, "nagelkerke": cs / (1 - np.exp(-dev0 / 2) ** (2 / n)),
            "mcfadden": 1 - dev / dev0, "R2": 1 - np.sum((y - p) ** 2) / np.sum(y ** 2),
            "percent_correct": np.mean(pred == y)}


def external_logistic_biplot(coords, Xbin, penalization=0.2, col_names=None, row_names=None) -> OrdinationBiplot:
    """External binary logistic biplot: each binary variable is fitted by a
    ridge logistic regression on a given configuration (PCoA, MDS...), giving
    its direction in the plane (``MultBiplotR::ExternalBinaryLogisticBiplot``)."""
    A = np.asarray(coords, float)
    Xb = np.asarray(Xbin, float)
    p = Xb.shape[1]
    fits = [ridge_logistic(Xb[:, j], A, penalization) for j in range(p)]
    B = np.array([f["beta"] for f in fits])
    dd = np.sqrt(1 + (B[:, 1:] ** 2).sum(1))
    info = pd.DataFrame({"Deviance": [f["dif"] for f in fits], "df": [f["df"] for f in fits],
                         "p-value": [f["p"] for f in fits],
                         "Bonferroni": np.minimum([f["p"] * p for f in fits], 1),
                         "Nagelkerke": [f["nagelkerke"] for f in fits], "R2": [f["R2"] for f in fits],
                         "% correct": [f["percent_correct"] * 100 for f in fits]},
                        index=col_names or [f"V{j + 1}" for j in range(p)])
    return OrdinationBiplot(title="External Binary Logistic Biplot", type="ExternalLogistic", dim=A.shape[1],
                            row_coordinates=A, col_parameters=B, col_coordinates=B[:, 1:], loadings=B[:, 1:] / dd[:, None],
                            var_info=info, total_percent=np.mean([f["percent_correct"] for f in fits]) * 100,
                            eigenvalues=np.var(A, axis=0), inertia=np.var(A, axis=0) / np.var(A, axis=0).sum() * 100,
                            row_names=row_names or [f"I{i + 1}" for i in range(A.shape[0])],
                            col_names=list(info.index), data=Xb)


# --------------------------------------------------------------------------- canonical analysis of distances
def canonical_distance_analysis(D, groups, dim=2, nperm=999, weighted=False, X=None, seed=None) -> OrdinationBiplot:
    """Canonical analysis of distances (Gower & Krzanowski, 1999; port of
    ``MultBiplotR::CanonicalDistanceAnalysis``): pseudo-F (distance-based
    ANOVA) with permutation p-value, PCoA of the group means and projection of
    the individuals (Euclidean distances on ``X``, Gower's add-a-point formula)."""
    D = np.asarray(D, float)
    g = np.asarray([str(v) for v in groups])
    levels = list(groups.cat.categories) if hasattr(groups, "cat") else sorted(set(g))
    levels = [str(v) for v in levels]
    G = np.column_stack([(g == l).astype(float) for l in levels])
    ng = G.sum(0); n = len(g); k = len(levels)
    D2 = 0.5 * D ** 2

    def stats(Dm):
        Fm = (G.T @ Dm @ G) / np.outer(ng, ng)
        f = np.diag(Fm)
        DE = 0.5 * (2 * Fm - f[:, None] - f[None, :])
        tss = Dm.sum() / n
        bss = ng @ DE @ ng / n
        return Fm, tss, bss
    Fm, tss, bss = stats(D2)
    wss = tss - bss
    Fexp = (bss / (k - 1)) / (wss / (n - k))
    rng = np.random.default_rng(seed)
    perm = np.empty(nperm)
    for i in range(nperm):
        o = rng.permutation(n)
        _, _, b = stats(D2[np.ix_(o, o)])
        perm[i] = (b / (k - 1)) / ((tss - b) / (n - k))
    H = np.eye(k) - (np.outer(np.ones(k), ng) / n if weighted else np.ones((k, k)) / k)
    B = -H @ Fm @ H.T
    ev, U = np.linalg.eigh((B + B.T) / 2)
    ev, U = ev[::-1], U[:, ::-1]
    Y = U * np.sqrt(np.maximum(ev, 0))
    d0 = (Y ** 2).sum(1)
    qual = Y[:, :dim] ** 2 / d0[:, None] * 100
    res = OrdinationBiplot(title="Canonical Analysis of Distances", type="CDA", dim=dim, F=Fexp, TSS=tss, BSS=bss,
                           WSS=wss, df=(k - 1, n - k), p_value=float(np.mean(perm > Fexp)), nperm=nperm,
                           eigenvalues=ev, inertia=ev / ev.sum() * 100, group_coordinates=Y[:, :dim],
                           group_qualities=qual, group_names=levels, groups=g, row_coordinates=None,
                           col_coordinates=None)
    if X is not None:
        Xm = np.asarray(X, float)
        means = (G.T @ Xm) / ng[:, None]
        Di = continuous_distances(means, Xm) ** 2
        Yd = Y[:, :dim]
        Yi = -0.5 * np.linalg.solve(Yd.T @ Yd, Yd.T @ H @ (Di - d0[None, :]).T)
        res.row_coordinates = Yi.T
    return res
