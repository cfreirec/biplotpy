"""Constrained (restricted) logistic biplots and the external ordinal logistic
biplot of MultBiplotR.

* ``constrained_logistic_biplot``: binary responses ``Y`` explained by
  continuous variables ``X`` (``ConstrainedLogisticBiplot``). Each response is
  regressed on ``X`` by ridge logistic regression; the standardized linear
  predictors are summarized by an SVD, whose first components are the row
  coordinates, and each response is then refitted on those coordinates.
* ``constrained_ordinal_logistic_biplot``: the same with ordinal responses and
  ridge cumulative-logit regressions (``ConstrainedOrdinalLogisticBiplot``).
* ``external_ordinal_logistic_biplot``: ordinal variables fitted by ridge
  cumulative-logit regressions on a given configuration (PCA, PCoA, MDS...),
  as ``AddOrdVars2Biplot``.
* ``ridge_ordinal_logistic``: the Newton (Fisher scoring) fit of MultBiplotR
  (``OrdinalLogisticFit``/``RidgeOrdinalLogistic``), with the ridge penalty on
  thresholds and slopes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.special import expit

from .distances import ridge_logistic
from .extras import add_continuous_variables
from .logistic import LogisticBiplot
from .transforms import initial_transform, transform_name


# --------------------------------------------------------------------------- helpers
def _ridge_binary_fit(y, Xd, penalization=0.2, tol=1e-5, maxiter=100):
    """``RidgeBinaryLogisticFit``: Newton-Raphson with ridge penalty (first column,
    the intercept, not penalized). ``Xd`` already contains the intercept."""
    n, m = Xd.shape
    beta = np.zeros(m)
    Im = np.eye(m); Im[0, 0] = 0.0
    err, it = 0.1, 0
    while err > tol and it < maxiter:
        it += 1
        old = beta
        mu = expit(Xd @ old)
        In = Xd.T @ ((mu * (1 - mu))[:, None] * Xd) + 2 * penalization * Im
        U = Xd.T @ (y - mu) - 2 * penalization * Im @ old
        beta = old + np.linalg.pinv(In) @ U
        err = np.sum(np.abs(old - beta))
    return beta


def _codes(y):
    """Ordinal variable -> integer codes 1..C (pandas ordered categorical, or numbers)."""
    if hasattr(y, "cat"):
        return y.cat.codes.to_numpy() + 1, list(y.cat.categories)
    y = np.asarray(y)
    lev = sorted(set(y.tolist()))
    return np.array([lev.index(v) + 1 for v in y]), lev


def ridge_ordinal_logistic(y, x=None, penalization=0.1, tol=1e-4, maxiter=200) -> dict:
    """Ridge cumulative-logit regression, ``logit P(y <= c) = tau_c - x beta``,
    fitted by Fisher scoring as ``MultBiplotR::OrdinalLogisticFit`` (the
    penalty ``penalization * |(tau, beta)|^2`` includes the thresholds).

    ``y``: integer codes 1..J. Returns thresholds, coefficients, linear
    predictors, fitted category probabilities, log-likelihood, deviance and,
    against the model without covariates, the likelihood-ratio test and the
    pseudo R2 of Cox-Snell, Nagelkerke and McFadden."""
    def fit(xm):
        y_ = np.asarray(y, int)
        n = len(y_)
        p = 0 if xm is None else xm.shape[1]
        J = int(y_.max())
        if J < 2:
            raise ValueError("There is a variable with the same value for all the items.")
        npar = J + p - 1
        Y = np.zeros((n, J))
        for i in range(n):
            if y_[i] > 0:
                Y[i, y_[i] - 1] = 1
        R = np.cumsum(Y, axis=1)
        A = R[:, :J - 1].sum(0) / n
        Beta = np.zeros(p)
        B = np.concatenate([A, Beta])
        err, it = 0.1, 0
        while err > tol and it < maxiter:
            it += 1
            ETA = np.tile(A, (n, 1)) if xm is None else A[None, :] - (xm @ Beta)[:, None]
            PIA = np.column_stack([np.exp(ETA) / (1 + np.exp(ETA)), np.ones(n)])
            PI = np.column_stack([PIA[:, :1], PIA[:, 1:J] - PIA[:, :J - 1]])
            Rho = np.log(PIA[:, :J - 1] / (PIA[:, 1:J] - PIA[:, :J - 1]))
            gRho = np.log(1 + np.exp(Rho))
            U = PIA[:, 1:J] / (PIA[:, :J - 1] * (PIA[:, 1:J] - PIA[:, :J - 1]))
            D = R[:, :J - 1] - R[:, 1:J] * (PIA[:, :J - 1] / PIA[:, 1:J])
            D2 = PIA[:, :J] * (1 - PIA[:, :J])
            Q = np.zeros((n, J - 1, npar))
            GR = np.zeros(npar)
            for k in range(npar):
                P = np.zeros((n, J))
                if k < J - 1:
                    P[:, k] = 1.0
                else:
                    P[:, :J - 1] = -xm[:, k - (J - 1)][:, None]
                Q[:, :, k] = P[:, :J - 1] * D2[:, :J - 1] - P[:, 1:J] * (PIA[:, :J - 1] / PIA[:, 1:J]) * D2[:, 1:J]
                GR[k] = np.sum(D * U * Q[:, :, k])
            HESS = np.einsum("nj,njs,njk->sk", U, Q, Q) + 2 * np.eye(npar) * penalization
            Bnew = B + np.linalg.solve(HESS, GR - 2 * B * penalization)
            err = np.sum((Bnew - B) ** 2)
            B = Bnew
            A = B[:J - 1]
            Beta = np.zeros(p) if xm is None else B[J - 1:J - 1 + p]
        L = np.sum(R[:, :J - 1] * Rho - R[:, 1:J] * gRho)
        pred = PI.argmax(1) + 1
        return {"thresholds": A, "coefficients": Beta, "eta": ETA, "fitted": PI, "pred": pred,
                "logLik": L, "deviance": -2 * L, "percent_correct": np.mean(y_ == pred), "iter": it,
                "J": J, "nvar": p, "covariances": np.linalg.inv(HESS)}
    x = None if x is None else np.asarray(x, float).reshape(len(y), -1)
    model = fit(x)
    null = fit(None)
    n = len(y)
    dif = null["deviance"] - model["deviance"]
    cs = 1 - np.exp(-dif / n)
    model.update({"null_deviance": null["deviance"], "dif": dif, "df": model["nvar"],
                  "p": sps.chi2.sf(dif, model["nvar"]), "cox_snell": cs,
                  "nagelkerke": cs / (1 - np.exp(null["deviance"] / -2) ** (2 / n)),
                  "mcfadden": 1 - model["deviance"] / null["deviance"]})
    return model


def _ordinal_columns(Ycodes, A, penalization, tol, maxiter, names):
    """Fit each ordinal variable on the configuration ``A`` (``AddOrdVars2Biplot``)."""
    n, p = Ycodes.shape
    dim = A.shape[1]
    ncat = [int(Ycodes[:, j].max()) for j in range(p)]
    coef = np.zeros((p, dim)); thr = np.zeros((p, max(ncat) - 1))
    fitcols = ["logLik", "Deviance", "df", "p-value", "PCC", "CoxSnell", "Macfaden", "Nagelkerke", "NullDeviance"]
    fit = np.zeros((p, 9)); models = []
    for j in range(p):
        m = ridge_ordinal_logistic(Ycodes[:, j], A, penalization, tol, maxiter)
        models.append(m)
        coef[j] = m["coefficients"]; thr[j, :len(m["thresholds"])] = m["thresholds"]
        fit[j] = [m["logLik"], m["deviance"], m["df"], m["p"], m["percent_correct"], m["cox_snell"],
                  m["mcfadden"], m["nagelkerke"], m["null_deviance"]]
    d = np.sqrt((coef ** 2).sum(1) + 1)
    loadings = coef / d[:, None]
    nn = n * p
    dev0, dev, df = fit[:, 8].sum(), fit[:, 1].sum(), fit[:, 2].sum()
    dif = dev0 - dev
    cs = 1 - np.exp(-dif / nn)
    ll = fit[:, 0].sum()
    return {"coefficients": pd.DataFrame(coef, index=names, columns=[f"Dim_{k + 1}" for k in range(dim)]),
            "thresholds": pd.DataFrame(thr, index=names, columns=[f"C_{c + 1}" for c in range(thr.shape[1])]),
            "fit": pd.DataFrame(fit, index=names, columns=fitcols), "loadings": loadings,
            "communalities": (loadings ** 2).sum(1), "ncats": ncat, "logLik": ll, "deviance": dev,
            "null_deviance": dev0, "df": df, "dif": dif, "p": sps.chi2.sf(dif, df), "cox_snell": cs,
            "nagelkerke": cs / (1 - np.exp(dev0 / -2) ** (2 / nn)), "mcfadden": 1 - dev / dev0,
            "AIC": -2 * ll + 2 * df, "BIC": -2 * ll + df * np.log(nn), "models": models}


def _ordinal_frame(Y):
    if isinstance(Y, pd.DataFrame):
        names = [str(c) for c in Y.columns]
        cols = [_codes(Y[c]) for c in Y.columns]
    else:
        Y = np.asarray(Y)
        names = [f"V{j + 1}" for j in range(Y.shape[1])]
        cols = [_codes(Y[:, j]) for j in range(Y.shape[1])]
    return np.column_stack([c[0] for c in cols]), [c[1] for c in cols], names


# --------------------------------------------------------------------------- binary
def constrained_logistic_biplot(Y, X, dim=2, scaling=5, penalization=0.1, tol=1e-5, maxiter=100,
                                row_names=None) -> LogisticBiplot:
    """Constrained binary logistic biplot (``MultBiplotR::ConstrainedLogisticBiplot``).

    The row coordinates are linear combinations of the explanatory variables
    ``X``: each binary response is regressed on ``X`` (ridge logistic
    regression, ``penalization``), the standardized linear predictors are
    decomposed by SVD and the first ``dim`` components (scaled by the singular
    values) are the row coordinates. Each response is then refitted on them
    (ridge penalty 0.2, as in R) and the explanatory variables are projected as
    supplementary continuous variables.

    Two details differ from MultBiplotR on purpose: the p-values of the
    deviance tests use ``dim`` degrees of freedom (R uses 1), and the
    percentages of correct classification threshold the probability at 0.5 (R
    thresholds the linear predictor at 0.5)."""
    if isinstance(Y, pd.DataFrame):
        col_names = [str(c) for c in Y.columns]; row_names = row_names or [str(i) for i in Y.index]
    else:
        col_names = [f"V{j + 1}" for j in range(np.shape(Y)[1])]
    x_names = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else None
    Y = np.asarray(Y, float); Xo = np.asarray(X, float)
    n, p = Y.shape
    q = Xo.shape[1]
    r = min(p, q)
    tname = transform_name(scaling)
    Xs = initial_transform(Xo, tname)["X"]
    Xd = np.column_stack([np.ones(n), Xs])
    Yh = np.zeros((n, p)); Betas = np.zeros((p, q + 1))
    for j in range(p):
        fit = ridge_logistic(Y[:, j], Xd, penalization=penalization, intercept=False, tol=tol, maxiter=maxiter)
        Betas[j] = fit["beta"]; Yh[:, j] = Xd @ fit["beta"]
    Yh = initial_transform(Yh, "Standardize columns")["X"]
    U, d, Vt = np.linalg.svd(Yh, full_matrices=False)
    ev = d[:r] ** 2
    A = U[:, :dim] * d[:dim]
    aa = np.column_stack([np.ones(n), A])
    B = np.vstack([_ridge_binary_fit(Y[:, j], aa, 0.2, tol, maxiter) for j in range(p)])
    esp = aa @ B.T
    P = expit(esp)
    pred = (P > 0.5).astype(float)
    acier = pred == Y
    pres = Y.sum(0); abs_ = n - pres
    sens = ((acier) & (Y == 1)).sum(0) / pres
    spec = ((acier) & (Y == 0)).sum(0) / abs_
    p0 = expit(B[:, 0])
    d1 = -2 * (Y * np.log(p0) + (1 - Y) * np.log(1 - p0)).sum(0)
    d2 = -2 * (Y * np.log(P) + (1 - Y) * np.log(1 - P)).sum(0)
    dev = d1 - d2
    cs = 1 - np.exp(-dev / n)
    dd = np.sqrt(1 + (B[:, 1:] ** 2).sum(1))
    loadings = B[:, 1:] / dd[:, None]
    row_names = row_names or [f"I{i + 1}" for i in range(n)]
    cols = pd.DataFrame(np.column_stack([B, dev, sps.chi2.sf(dev, dim), acier.mean(0) * 100, 1 - d2 / d1]),
                        index=col_names, columns=["b0"] + [f"b{k + 1}" for k in range(dim)] +
                        ["Deviance", "p-value", "% correct", "McFadden R2"])
    out = LogisticBiplot(
        title="Constrained Binary Logistic Biplot", n=n, p=p, dim=dim, penalization=penalization, A=A, B=B,
        fitted=P, objective=float(d2.sum() / 2), converged=True, columns=cols,
        betas=pd.DataFrame(Betas, index=col_names, columns=["Intercept"] + (x_names or [f"X{k + 1}" for k in range(q)])),
        eigenvalues=ev, inertia=ev / ev.sum() * 100, null_deviances=d1, model_deviances=d2, deviances=dev,
        cox_snell=cs, nagelkerke=cs / (1 - np.exp(d1 / -2) ** (2 / n)), mcfadden=1 - d2 / d1,
        loadings=loadings, communalities=(loadings ** 2).sum(1), sensitivity=sens, specificity=spec,
        global_percent=acier.mean() * 100, row_percent=acier.mean(1) * 100, row_names=row_names, col_names=col_names)
    out.row_coordinates = A
    out.sup_variables = add_continuous_variables(out, Xo, dims=dim, scaling=scaling, names=x_names)
    return out


# --------------------------------------------------------------------------- ordinal
def constrained_ordinal_logistic_biplot(Y, X, dim=2, scaling=5, penalization=0.1, tol=1e-5, maxiter=100,
                                        row_names=None) -> LogisticBiplot:
    """Constrained ordinal logistic biplot (``MultBiplotR::ConstrainedOrdinalLogisticBiplot``).

    ``Y``: ordinal variables (DataFrame of ordered categoricals, or integer
    codes 1..C). Each is regressed on the transformed ``X`` by ridge
    cumulative-logit regression; the standardized linear predictors (first
    threshold) are decomposed by SVD; the first ``dim`` components are the row
    coordinates and each ordinal variable is refitted on them."""
    Yc, cats, names = _ordinal_frame(Y)
    if isinstance(Y, pd.DataFrame):
        row_names = row_names or [str(i) for i in Y.index]
    x_names = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else None
    Xs = initial_transform(np.asarray(X, float), transform_name(scaling))["X"]
    n, p = Yc.shape
    r = min(p, Xs.shape[1])
    Yh = np.zeros((n, p)); fits = []
    for j in range(p):
        m = ridge_ordinal_logistic(Yc[:, j], Xs, penalization, tol, maxiter)
        fits.append(m)
        e = m["eta"][:, 0]
        Yh[:, j] = (e - e.mean()) / e.std(ddof=1)
    U, d, Vt = np.linalg.svd(Yh, full_matrices=False)
    ev = d[:r] ** 2
    A = U[:, :dim] * d[:dim]
    cp = _ordinal_columns(Yc, A, penalization, tol, maxiter, names)
    row_names = row_names or [f"I{i + 1}" for i in range(n)]
    cols = pd.concat([cp["coefficients"], cp["fit"][["Deviance", "p-value", "PCC"]]], axis=1)
    out = LogisticBiplot(
        title="Constrained Ordinal Logistic Biplot", n=n, p=p, dim=dim, penalization=penalization, A=A,
        B=cp["coefficients"].to_numpy(), thresholds=[cp["thresholds"].to_numpy()[j, :cp["ncats"][j] - 1] for j in range(p)],
        objective=-cp["logLik"], converged=True, columns=cols, column_parameters=cp, eigenvalues=ev,
        inertia=ev / ev.sum() * 100, categories=cats, regressions=fits,
        global_percent=float(np.mean(cp["fit"]["PCC"]) * 100), row_names=row_names, col_names=names,
        loadings=cp["loadings"], communalities=cp["communalities"], deviance=cp["deviance"],
        null_deviance=cp["null_deviance"], df=cp["df"], pvalue=cp["p"], AIC=cp["AIC"], BIC=cp["BIC"])
    out.row_coordinates = A
    out.sup_variables = add_continuous_variables(out, Xs, dims=dim, scaling=scaling, names=x_names)
    return out


def external_ordinal_logistic_biplot(coords, Y, penalization=0.2, tol=1e-6, maxiter=100,
                                     row_names=None) -> LogisticBiplot:
    """External ordinal logistic biplot: each ordinal variable is fitted by a
    ridge cumulative-logit regression on a given configuration ``coords``
    (principal components, coordinates of a PCoA or an MDS...), as
    ``MultBiplotR::AddOrdVars2Biplot``. The lines of the variables and the
    category boundaries are drawn with ``plot()``."""
    A = np.asarray(getattr(coords, "row_coordinates", coords), float)
    Yc, cats, names = _ordinal_frame(Y)
    n, p = Yc.shape
    cp = _ordinal_columns(Yc, A, penalization, tol, maxiter, names)
    cols = pd.concat([cp["coefficients"], cp["fit"][["Deviance", "p-value", "PCC"]]], axis=1)
    return LogisticBiplot(
        title="External Ordinal Logistic Biplot", n=n, p=p, dim=A.shape[1], penalization=penalization, A=A,
        B=cp["coefficients"].to_numpy(), thresholds=[cp["thresholds"].to_numpy()[j, :cp["ncats"][j] - 1] for j in range(p)],
        objective=-cp["logLik"], converged=True, columns=cols, column_parameters=cp, categories=cats,
        global_percent=float(np.mean(cp["fit"]["PCC"]) * 100), row_names=row_names or [f"I{i + 1}" for i in range(n)],
        col_names=names, loadings=cp["loadings"], communalities=cp["communalities"], deviance=cp["deviance"],
        null_deviance=cp["null_deviance"], df=cp["df"], pvalue=cp["p"], AIC=cp["AIC"], BIC=cp["BIC"])
