"""Factor-type biplots and two-table biplots.

* ``rotate`` — gradient-projection rotations (port of ``GPArotation``
  GPForth / GPFoblq): varimax, quartimax, equamax, parsimax, entropy
  (orthogonal); quartimin and oblimin (oblique).
* ``fa_biplot`` — ``MultBiplotR::FA.Biplot`` (principal-component
  extraction of the correlation matrix, rotation, factor scores).
* ``nipals_biplot`` — PCA biplot by NIPALS that accepts missing values
  (``ade4::nipals``), with sparse and truncated variants
  (``MultBiplotR::Sparse.NIPALSPCA`` / ``Truncated.NIPALSPCA``).
* ``pls_biplot`` — PLS regression (NIPALS, orthogonal scores, as
  ``pls::plsr(method = "oscorespls")``) and its biplot
  (``MultBiplotR::Biplot.PLSR``).
* ``coinertia`` — co-inertia analysis of two tables (``ade4::coinertia``).
* ``polychoric`` / ``polychoric_biplot`` — two-step polychoric correlations
  (``psych::polychoric``) and the ordinal biplot built from them
  (``MultBiplotR::PolyBiplot``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps
from scipy.special import expit

from .classical import ContinuousBiplot
from .transforms import initial_transform, transform_name


# --------------------------------------------------------------------------- rotations
def _crit(method, L, kappa=0.0, gam=0.0):
    L2 = L ** 2
    if method == "varimax":
        QL = L2 - L2.mean(0)
        return -L * QL, -np.sum(QL ** 2) / 4
    if method == "quartimax":
        return -L ** 3, -np.sum(L2 ** 2) / 4
    if method == "cf":
        p, k = L.shape
        N = np.ones((k, k)) - np.eye(k); M = np.ones((p, p)) - np.eye(p)
        f = (1 - kappa) * np.sum(L2 * (L2 @ N)) / 4 + kappa * np.sum(L2 * (M @ L2)) / 4
        return (1 - kappa) * L * (L2 @ N) + kappa * L * (M @ L2), f
    if method == "entropy":
        lg = np.log(L2 + (L2 == 0))
        return -(L * lg + L), -np.sum(L2 * lg) / 2
    if method in ("oblimin", "quartimin"):
        X = L2 @ (1 - np.eye(L.shape[1]))
        if gam:
            p = L.shape[0]
            X = (np.eye(p) - gam / p) @ X
        return L * X, np.sum(L2 * X) / 4
    raise ValueError(method)


def rotate(A, method="varimax", normalize=False, eps=1e-5, maxit=1000, gamma=0.0):
    """Rotate a loading matrix. Returns ``(loadings, T, Phi)``; Phi is the
    factor correlation matrix (identity for orthogonal rotations)."""
    A = np.asarray(A, float)
    p, k = A.shape
    if method == "none" or k < 2:
        return A, np.eye(k), np.eye(k)
    W = np.sqrt((A ** 2).sum(1))[:, None] if normalize else np.ones((p, 1))
    A = A / W
    kw = {}
    m = method
    if method == "equamax":
        m, kw = "cf", {"kappa": k / (2 * p)}
    elif method == "parsimax":
        m, kw = "cf", {"kappa": (k - 1) / (p + k - 2)}
    elif method == "oblimin":
        kw = {"gam": gamma}
    oblique = m in ("oblimin", "quartimin")
    T = np.eye(k); al = 1.0
    if not oblique:
        L = A @ T
        Gq, f = _crit(m, L, **kw)
        G = A.T @ Gq
        for _ in range(maxit + 1):
            M = T.T @ G
            Gp = G - T @ ((M + M.T) / 2)
            s = np.sqrt(np.sum(Gp ** 2))
            if s < eps:
                break
            al *= 2
            for _ in range(11):
                U, _, Vt = np.linalg.svd(T - al * Gp)
                Tt = U @ Vt
                L = A @ Tt
                Gqt, ft = _crit(m, L, **kw)
                if ft < f - 0.5 * s ** 2 * al:
                    break
                al /= 2
            T, f, G = Tt, ft, A.T @ Gqt
        L = A @ T
        Phi = np.eye(k)
    else:
        L = A @ np.linalg.inv(T).T
        Gq, f = _crit(m, L, **kw)
        G = -(L.T @ Gq @ np.linalg.inv(T)).T
        for _ in range(maxit + 1):
            Gp = G - T @ np.diag(np.sum(T * G, 0))
            s = np.sqrt(np.sum(Gp ** 2))
            if s < eps:
                break
            al *= 2
            for _ in range(11):
                X = T - al * Gp
                Tt = X / np.sqrt((X ** 2).sum(0))
                L = A @ np.linalg.inv(Tt).T
                Gqt, ft = _crit(m, L, **kw)
                if f - ft > 0.5 * s ** 2 * al:
                    break
                al /= 2
            T, f = Tt, ft
            G = -(L.T @ Gqt @ np.linalg.inv(Tt)).T
        Phi = T.T @ T
    return L * W, T, Phi


# --------------------------------------------------------------------------- FA biplot
def fa_biplot(X, dim=2, rotation="varimax", normalize=False, scores="regression", row_names=None,
              col_names=None) -> ContinuousBiplot:
    """Factor-analysis biplot with principal-component extraction of the
    correlation matrix (``MultBiplotR::FA.Biplot``, Extraction = "PC").

    ``scores``: 'regression' (Thurstone, X R^-1 B) or 'bartlett'
    (X Psi^-1 B (B' Psi^-1 B)^-1, Psi = diag(uniquenesses)). MultBiplotR
    swaps the two names and squares the uniquenesses (see ``FIXES``)."""
    if isinstance(X, pd.DataFrame):
        row_names = row_names or [str(i) for i in X.index]; col_names = col_names or [str(c) for c in X.columns]
    X0 = np.asarray(X, float); n, p = X0.shape
    Z = (X0 - X0.mean(0)) / X0.std(0, ddof=1)
    R = np.corrcoef(Z, rowvar=False)
    ev, U = np.linalg.eigh(R); ev, U = ev[::-1], U[:, ::-1]
    B0 = U[:, :dim] * np.sqrt(ev[:dim])
    B, T, Phi = rotate(B0, rotation, normalize)
    h = (B ** 2).sum(1) if rotation not in ("oblimin", "quartimin") else np.diag(B @ Phi @ B.T)
    u = 1 - h
    if scores == "regression":
        A = Z @ np.linalg.solve(R, B @ Phi) if rotation in ("oblimin", "quartimin") else Z @ np.linalg.solve(R, B)
    elif scores == "bartlett":
        Pi = np.diag(1 / u)
        A = Z @ Pi @ B @ np.linalg.inv(B.T @ Pi @ B)
    else:
        raise ValueError("scores must be 'regression' or 'bartlett'")
    inertia = ev[:dim] / p * 100
    row_names = row_names or [f"I{i + 1}" for i in range(n)]
    col_names = col_names or [f"V{j + 1}" for j in range(p)]
    sf = (Z ** 2).sum(1)
    rc = (A * np.sqrt(ev[:dim])) ** 2 / sf[:, None] * 100
    return ContinuousBiplot(title="Factor Analysis Biplot", type="FA", alpha=0, normalization=f"FA ({rotation})",
                            dim=dim, transformation="Standardize columns", non_scaled_data=X0, scaled_data=Z,
                            means=X0.mean(0), deviations=X0.std(0, ddof=1), minima=X0.min(0), maxima=X0.max(0),
                            medians=np.median(X0, 0), gmean=X0.mean(), eigenvalues=ev, inertia=np.r_[inertia, np.zeros(p - dim)],
                            cum_inertia=np.cumsum(inertia), row_coordinates=A, col_coordinates=B,
                            row_contributions=rc, col_contributions=B ** 2 * 100, communalities=h, uniquenesses=u,
                            rotation=rotation, Phi=Phi, R=R, structure=np.corrcoef(np.column_stack([Z, A]), rowvar=False)[:p, p:],
                            scale_factor=1.0, row_names=row_names, col_names=col_names, n=n, p=p,
                            sup_row_coordinates=None, sup_col_coordinates=None)


# --------------------------------------------------------------------------- NIPALS
def nipals(X, dim=2, tol=1e-9, maxit=500, standardize=True):
    """NIPALS PCA with missing values (``ade4::nipals``): columns centred and
    scaled (population sd) on the available values; returns scores ``li``,
    normed loadings ``c1``, eigenvalues and the reconstructed (imputed)
    standardized table ``rec``."""
    X = np.asarray(X, float)
    n, p = X.shape
    M = np.isfinite(X)
    mu = np.nanmean(X, 0)
    nr = M.sum(0)
    sd = np.nanstd(X, 0, ddof=1) * np.sqrt((nr - 1) / nr) if standardize else np.ones(p)
    E = (X - mu) / sd
    li = np.zeros((n, dim)); c1 = np.zeros((p, dim)); eig = np.zeros(dim); nb = np.zeros(dim, int)
    for h in range(dim):
        th = E[:, 0].copy()
        ph1 = np.full(p, 1 / np.sqrt(p))
        it = 0
        while True:
            Ez, Mz = np.where(M, E, 0.0), M
            thz = np.where(np.isfinite(th), th, 0.0)
            ph2 = (Ez * thz[:, None]).sum(0) / (Mz * thz[:, None] ** 2).sum(0)
            ph2 = ph2 / np.sqrt(np.sum(ph2 ** 2))
            th = (Ez * ph2[None, :]).sum(1) / (Mz * ph2[None, :] ** 2).sum(1)
            diff = ph2 - ph1; ph1 = ph2; it += 1
            if np.sum(diff ** 2) <= tol or it > maxit:
                break
        E = E - np.outer(th, ph1)
        li[:, h], c1[:, h], nb[h] = th, ph1, it
        eig[h] = np.sum(th ** 2) / (n - 1)
    rec = li @ c1.T
    return {"li": li, "c1": c1, "co": c1 * np.sqrt(eig), "eig": eig, "rec": rec, "imputed": rec * sd + mu,
            "means": mu, "sd": sd, "iterations": nb}


def _nipals_regular(X, dim, tol=1e-6, maxit=1000, mode="regular", lam=0.02):
    n, p = X.shape
    U = np.zeros((n, dim)); V = np.zeros((p, dim)); d = np.zeros(dim)
    E = X.copy()
    for i in range(dim):
        u = E[:, 0].copy()
        for _ in range(maxit):
            uold = u
            v = u @ E / np.sum(u ** 2); v /= np.linalg.norm(v)
            if mode == "sparse":
                v = np.sign(v) * np.maximum(np.abs(v) - lam, 0)
                v /= np.linalg.norm(v)
            elif mode == "truncated":
                v = v * (np.abs(v) > lam); v /= np.linalg.norm(v)
            u = E @ v
            if np.sum((u - uold) ** 2) <= tol:
                break
        d[i] = np.linalg.norm(u)
        E = E - np.outer(u, v)
        U[:, i] = u / d[i]; V[:, i] = v
    return U, d, V


def nipals_biplot(X, dim=2, alpha=1.0, scaling=5, mode="regular", lam=0.02, row_names=None,
                  col_names=None) -> ContinuousBiplot:
    """NIPALS biplot (``MultBiplotR::NIPALS.Biplot``). With missing values
    (NaN) the ``ade4::nipals`` algorithm is used on standardized data;
    ``mode`` = 'sparse' (soft thresholding) or 'truncated' (hard
    thresholding) gives sparse loadings for complete data."""
    if isinstance(X, pd.DataFrame):
        row_names = row_names or [str(i) for i in X.index]; col_names = col_names or [str(c) for c in X.columns]
    X0 = np.asarray(X, float); n, p = X0.shape
    row_names = row_names or [f"I{i + 1}" for i in range(n)]
    col_names = col_names or [f"V{j + 1}" for j in range(p)]
    if np.isnan(X0).any():
        r = nipals(X0, dim)
        Xs = r["rec"]
        U = r["li"] / np.sqrt((r["li"] ** 2).sum(0)); d = np.sqrt((r["li"] ** 2).sum(0)); V = r["c1"]
        tname = "Standardize columns (available values)"
    else:
        tname = transform_name(scaling)
        Xs = initial_transform(X0, tname)["X"]
        U, d, V = _nipals_regular(Xs, dim, mode=mode, lam=lam)
        if mode == "truncated":
            U = Xs @ V; d = np.sqrt((U ** 2).sum(0)); U = U / d
    a, b = U * d, V.copy()
    tot = np.nansum(Xs ** 2)
    inertia = d ** 2 / tot * 100
    if alpha <= 1:
        a = U * d ** alpha; b = V * d ** (1 - alpha)
        scf = np.sqrt(np.sqrt(((b ** 2).sum() / p) / ((a ** 2).sum() / n)))
        a, b = a * scf, b / scf
    else:
        b = V * d; scf = 1.0
    return ContinuousBiplot(title={"regular": "NIPALS Biplot", "sparse": "Sparse NIPALS Biplot",
                                   "truncated": "Truncated NIPALS Biplot"}[mode], type="NIPALS", alpha=alpha,
                            normalization="NIPALS", dim=dim, transformation=tname, non_scaled_data=X0, scaled_data=Xs,
                            means=np.nanmean(X0, 0), deviations=np.nanstd(X0, 0, ddof=1), minima=np.nanmin(X0, 0),
                            maxima=np.nanmax(X0, 0), medians=np.nanmedian(X0, 0), gmean=np.nanmean(X0),
                            eigenvalues=d ** 2, inertia=inertia, cum_inertia=np.cumsum(inertia), row_coordinates=a,
                            col_coordinates=b, V=V, U=U, d=d, scale_factor=scf, row_names=row_names, col_names=col_names,
                            n=n, p=p, row_contributions=None, col_contributions=None, sup_row_coordinates=None,
                            sup_col_coordinates=None, missing=np.isnan(X0))


# --------------------------------------------------------------------------- PLS
def pls_regression(X, Y, ncomp=2, scale=True, tol=1e-12, maxit=500):
    """PLS regression by NIPALS with orthogonal scores (``pls::plsr``,
    method 'oscorespls'): X centred (and scaled by its sd when ``scale``),
    Y centred. Returns scores T, loadings P, Y loadings Q, weights W,
    regression coefficients (for the scaled X) and fitted values."""
    X0 = np.asarray(X, float); Y0 = np.asarray(Y, float)
    if Y0.ndim == 1:
        Y0 = Y0[:, None]
    xm, ym = X0.mean(0), Y0.mean(0)
    xs = X0.std(0, ddof=1) if scale else np.ones(X0.shape[1])
    E = (X0 - xm) / xs; F = Y0 - ym
    n, p = E.shape; q = F.shape[1]
    T = np.zeros((n, ncomp)); P = np.zeros((p, ncomp)); Q = np.zeros((q, ncomp)); W = np.zeros((p, ncomp))
    U = np.zeros((n, ncomp))
    for a in range(ncomp):
        if q == 1:
            w = E.T @ F[:, 0]; w /= np.linalg.norm(w)
        else:
            u = F[:, np.argmax((F ** 2).sum(0))].copy()
            w_old = None
            for _ in range(maxit):
                w = E.T @ u; w /= np.linalg.norm(w)
                t = E @ w
                c = F.T @ t / (t @ t)
                u = F @ c / (c @ c)
                if w_old is not None and np.sum((w - w_old) ** 2) < tol:
                    break
                w_old = w
        t = E @ w
        tt = t @ t
        pa = E.T @ t / tt; qa = F.T @ t / tt
        E = E - np.outer(t, pa); F = F - np.outer(t, qa)
        T[:, a], P[:, a], Q[:, a], W[:, a] = t, pa, qa, w
        U[:, a] = (Y0 - ym) @ qa / (qa @ qa)
    R = W @ np.linalg.inv(P.T @ W)
    B = R @ Q.T
    fitted = ym + ((X0 - xm) / xs) @ B
    return {"scores": T, "loadings": P, "y_loadings": Q, "weights": W, "projection": R, "coef": B,
            "fitted": fitted, "y_scores": U, "x_means": xm, "x_sd": xs, "y_means": ym,
            "explained_x": (T ** 2).sum(0) * (P ** 2).sum(0) / np.sum(((X0 - xm) / xs) ** 2) * 100}


def _sup_coords(A, Y):
    """Coordinates of (standardized) supplementary variables regressed on the row coordinates."""
    Y = Y[:, None] if Y.ndim == 1 else Y
    Ys = (Y - Y.mean(0)) / Y.std(0, ddof=1)
    D = np.column_stack([np.ones(len(A)), A])
    return np.linalg.lstsq(D, Ys, rcond=None)[0][1:].T


def pls_biplot(X, Y, ncomp=2, scale=True, row_names=None, col_names=None, y_names=None) -> ContinuousBiplot:
    """PLS-R biplot (``MultBiplotR::Biplot.PLSR``): individuals by their X
    scores, X variables by their loadings (rescaled for display) and the
    responses projected as supplementary variables (regression of the
    standardized responses on the row coordinates, as AddContVars2Biplot)."""
    if isinstance(X, pd.DataFrame):
        row_names = row_names or [str(i) for i in X.index]; col_names = col_names or [str(c) for c in X.columns]
    if isinstance(Y, pd.DataFrame):
        y_names = y_names or [str(c) for c in Y.columns]
    r = pls_regression(X, Y, ncomp, scale)
    X0 = np.asarray(X, float); n, p = X0.shape
    a, b, qy = r["scores"], r["loadings"], r["y_loadings"]
    scf = np.sqrt(np.sqrt(((b ** 2).sum() / p) / ((a ** 2).sum() / n)))
    Z = (X0 - r["x_means"]) / r["x_sd"]
    fit = np.array([np.sum((np.outer(a[:, k], b[:, k])) ** 2) for k in range(ncomp)]) / np.sum(Z ** 2) * 100
    return ContinuousBiplot(title="PLSR Biplot", type="PLSR", alpha=0, normalization="PLS", dim=ncomp,
                            transformation="Standardize columns" if scale else "Column centering", non_scaled_data=X0,
                            scaled_data=Z, means=X0.mean(0), deviations=X0.std(0, ddof=1), minima=X0.min(0),
                            maxima=X0.max(0), medians=np.median(X0, 0), gmean=X0.mean(), eigenvalues=fit / 100,
                            inertia=fit, cum_inertia=np.cumsum(fit), row_coordinates=a * scf, col_coordinates=b / scf,
                            y_coordinates=_sup_coords(a * scf, np.asarray(Y, float)), y_names=y_names or [f"Y{k + 1}" for k in range(qy.shape[0])],
                            pls=r, scale_factor=scf, row_names=row_names or [f"I{i + 1}" for i in range(n)],
                            col_names=col_names or [f"V{j + 1}" for j in range(p)], n=n, p=p,
                            row_contributions=None, col_contributions=None, sup_row_coordinates=None,
                            sup_col_coordinates=None)


# --------------------------------------------------------------------------- co-inertia
def coinertia(X, Y, scale_x=True, scale_y=True, dim=2) -> dict:
    """Co-inertia analysis of two tables with the same rows (``ade4::coinertia``
    on two ``dudi.pca``). Returns eigenvalues, RV, normed axes of X (``c1``)
    and Y (``l1``), variable coordinates (``co``, ``li``) and row scores in
    each table (``lX``, ``lY``)."""
    def prep(M, sc):
        M = np.asarray(M, float); Mc = M - M.mean(0)
        return Mc / M.std(0) if sc else Mc
    Xt, Yt = prep(X, scale_x), prep(Y, scale_y)
    n = Xt.shape[0]
    C = Yt.T @ Xt / n
    U, d, Vt = np.linalg.svd(C, full_matrices=False)
    rank = int(np.sum(d ** 2 / d[0] ** 2 > 1e-7))
    eig = d[:rank] ** 2
    V = Vt.T
    c1, l1 = V[:, :dim], U[:, :dim]
    eX = np.linalg.eigvalsh(Xt.T @ Xt / n); eY = np.linalg.eigvalsh(Yt.T @ Yt / n)
    RV = eig.sum() / np.sqrt(np.sum(eX ** 2)) / np.sqrt(np.sum(eY ** 2))
    return {"eig": eig, "RV": RV, "c1": c1, "l1": l1, "co": c1 * d[:dim], "li": l1 * d[:dim],
            "lX": Xt @ c1, "lY": Yt @ l1, "inertia": eig / eig.sum() * 100}


# --------------------------------------------------------------------------- polychoric
_GL = {6: (np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970]),
           np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904])),
       12: (np.array([0.9815606342467191, 0.9041172563704750, 0.7699026741943050, 0.5873179542866171,
                      0.3678314989981802, 0.1252334085114692]),
            np.array([0.04717533638651177, 0.1069393259953183, 0.1600783285433464, 0.2031674267230659,
                      0.2334925365383547, 0.2491470458134029])),
       20: (np.array([0.9931285991850949, 0.9639719272779138, 0.9122344282513259, 0.8391169718222188,
                      0.7463319064601508, 0.6360536807265150, 0.5108670019508271, 0.3737060887154196,
                      0.2277858511416451, 0.07652652113349733]),
            np.array([0.01761400713915212, 0.04060142980038694, 0.06267204833410906, 0.08327674157670475,
                      0.1019301198172404, 0.1181945319615184, 0.1316886384491766, 0.1420961093183821,
                      0.1491729864726037, 0.1527533871307259]))}


def _bvnu(dh, dk, r):
    """Upper bivariate normal probability P(X > dh, Y > dk) with correlation r
    (Genz's BVNU / Drezner-Wesolowsky algorithm, double precision)."""
    if abs(r) < 0.3:
        x, w = _GL[6]
    elif abs(r) < 0.75:
        x, w = _GL[12]
    else:
        x, w = _GL[20]
    h, k = dh, dk
    hk = h * k
    bvn = 0.0
    if abs(r) < 0.925:
        hs = (h * h + k * k) / 2
        asr = np.arcsin(r)
        for sgn in (-1, 1):
            sn = np.sin(asr * (sgn * x + 1) / 2)
            bvn += np.sum(w * np.exp((sn * hk - hs) / (1 - sn * sn)))
        bvn = bvn * asr / (4 * np.pi) + sps.norm.cdf(-h) * sps.norm.cdf(-k)
    else:
        if r < 0:
            k, hk = -k, -hk
        if abs(r) < 1:
            as_ = (1 - r) * (1 + r)
            a = np.sqrt(as_)
            bs = (h - k) ** 2
            c = (4 - hk) / 8
            d = (12 - hk) / 16
            bvn = a * np.exp(-(bs / as_ + hk) / 2) * (1 - c * (bs - as_) * (1 - d * bs / 5) / 3 + c * d * as_ * as_ / 5)
            if hk > -160:
                b = np.sqrt(bs)
                bvn -= np.exp(-hk / 2) * np.sqrt(2 * np.pi) * sps.norm.cdf(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3)
            a = a / 2
            for sgn in (-1, 1):
                xs = (a * (sgn * x + 1)) ** 2
                rs = np.sqrt(1 - xs)
                asr = -(bs / xs + hk) / 2
                ok = asr > -100
                bvn += a * np.sum(w[ok] * np.exp(asr[ok]) * (np.exp(-hk * (1 - rs[ok]) / (2 * (1 + rs[ok]))) / rs[ok]
                                                           - (1 + c * xs[ok] * (1 + d * xs[ok]))))
            bvn = -bvn / (2 * np.pi)
        if r > 0:
            bvn += sps.norm.cdf(-max(h, k))
        else:
            bvn = -bvn
            if k > h:
                bvn += sps.norm.cdf(k) - sps.norm.cdf(h) if h < 0 else sps.norm.cdf(-h) - sps.norm.cdf(-k)
    return bvn


def _bvn_cdf(h, k, r):
    """P(X < h, Y < k) for a standard bivariate normal with correlation r."""
    if h == -np.inf or k == -np.inf:
        return 0.0
    if h == np.inf and k == np.inf:
        return 1.0
    if h == np.inf:
        return sps.norm.cdf(k)
    if k == np.inf:
        return sps.norm.cdf(h)
    return _bvnu(-h, -k, r)


def _cell_probs(t1, t2, r):
    a = np.r_[-np.inf, t1, np.inf]; b = np.r_[-np.inf, t2, np.inf]
    Fc = np.array([[_bvn_cdf(x, y, r) for y in b] for x in a])
    return Fc[1:, 1:] - Fc[:-1, 1:] - Fc[1:, :-1] + Fc[:-1, :-1]


def polychoric(X, correct=0.0):
    """Two-step polychoric correlation matrix (``psych::polychoric``):
    thresholds from the marginal proportions of each variable, then the
    correlation of each pair by maximum likelihood with the thresholds fixed.
    Zero cells may receive ``correct`` (psych applies it only with global = FALSE;
    the default 0 reproduces psych with global thresholds). Categories must be 1..C."""
    X = np.asarray(X, int)
    n, p = X.shape
    cats = [np.unique(X[:, j]) for j in range(p)]
    taus = []
    for j in range(p):
        cnt = np.array([np.sum(X[:, j] == c) for c in cats[j]], float)
        taus.append(sps.norm.ppf(np.cumsum(cnt)[:-1] / n))
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            tab = np.array([[np.sum((X[:, i] == a) & (X[:, j] == b)) for b in cats[j]] for a in cats[i]], float)
            if correct:
                tab[tab == 0] = correct
            f = lambda r: -np.sum(tab * np.log(np.maximum(_cell_probs(taus[i], taus[j], r), 1e-300)))
            R[i, j] = R[j, i] = optimize.minimize_scalar(f, bounds=(-0.999, 0.999), method="bounded",
                                                         options={"xatol": 1e-8}).x
    maxc = max(len(t) for t in taus)
    tau = np.full((p, maxc), np.nan)
    for j, t in enumerate(taus):
        tau[j, :len(t)] = t
    return {"rho": R, "tau": tau}


def polychoric_biplot(X, dim=2, logistic_scale=1.702, penalization=1.0, row_names=None, col_names=None):
    """Ordinal biplot from the factorization of the polychoric correlation
    matrix (``MultBiplotR::PolyBiplot``, fm = 'pca'): loadings of the
    principal components of the polychoric matrix converted to the
    cumulative-logit parameters b_j = c * lambda_j / sqrt(1 - h_j) and
    thresholds tau_jk / sqrt(1 - h_j) (c = 1.702, normal-ogive to logistic),
    then row scores by penalized maximum likelihood with those parameters fixed
    (``penalization`` = 1 is a standard normal prior on the scores, which keeps
    the scores of all-extreme response patterns finite)."""
    if isinstance(X, pd.DataFrame):
        row_names = row_names or [str(i) for i in X.index]; col_names = col_names or [str(c) for c in X.columns]
    Xm = np.asarray(X, int)
    n, p = Xm.shape
    pc = polychoric(Xm)
    R = pc["rho"]
    ev = np.linalg.eigvalsh(R)
    if ev.min() < 0:
        from scipy.linalg import eigh
        w, Vv = eigh(R); w = np.maximum(w, 1e-6); R = Vv @ np.diag(w) @ Vv.T
        dd = np.sqrt(np.diag(R)); R = R / np.outer(dd, dd)
    ev, U = np.linalg.eigh(R); ev, U = ev[::-1], U[:, ::-1]
    L = U[:, :dim] * np.sqrt(ev[:dim])
    h = (L ** 2).sum(1)
    s = np.sqrt(np.maximum(1 - h, 1e-6))
    B = logistic_scale * L / s[:, None]
    thr = logistic_scale * pc["tau"] / s[:, None]
    ncat = [int(np.sum(np.isfinite(thr[j]))) + 1 for j in range(p)]

    def negll(a, x):
        eta = B @ a
        ll = 0.0
        for j in range(p):
            t = np.r_[-np.inf, thr[j, :ncat[j] - 1], np.inf]
            c = x[j]
            pr = expit(t[c] - eta[j]) - expit(t[c - 1] - eta[j])
            ll += np.log(max(pr, 1e-300))
        return -ll + 0.5 * penalization * a @ a
    A = np.array([optimize.minimize(negll, np.zeros(dim), args=(Xm[i],), method="BFGS").x for i in range(n)])
    from .logistic import LogisticBiplot
    return LogisticBiplot(title="Polychoric (ordinal) Biplot", n=n, p=p, dim=dim, penalization=penalization, A=A, B=B,
                          thresholds=[thr[j, :ncat[j] - 1] for j in range(p)], loadings=L, communalities=h,
                          polychoric=R, objective=np.nan, converged=True, columns=pd.DataFrame(
                              np.column_stack([L, h]), index=col_names or [f"V{j + 1}" for j in range(p)],
                              columns=[f"Dim {k + 1}" for k in range(dim)] + ["Communality"]),
                          global_percent=np.nan, row_names=row_names or [f"I{i + 1}" for i in range(n)],
                          col_names=col_names or [f"V{j + 1}" for j in range(p)], fitted=None)
