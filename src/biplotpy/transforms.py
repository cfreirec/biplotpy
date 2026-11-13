"""Initial transformations of a data matrix (port of MultBiplotR::InitialTransform).

The transformation can be given by name or by its number in MultBiplotR (1-12).
Two documented defects of the R code are corrected here (see ``FIXES``).
"""
from __future__ import annotations

import numpy as np

TRANSFORMS = ["Raw Data", "Substract the global mean", "Double centering", "Column centering",
              "Standardize columns", "Row centering", "Standardize rows",
              "Divide by the column means and center", "Normalized residuals from independence",
              "Divide by the range", "Within groups standardization", "Ranks"]

FIXES = {
    "Within groups standardization": "MultBiplotR divides by the within-group variance; biplotpy divides by its square root (standard deviation).",
    "Logit": "MultBiplotR computes the logit into an unused variable, so the data are not transformed; biplotpy applies it.",
    "Supplementary rows": "PCA.Biplot in MultBiplotR projects the untransformed supplementary rows; biplotpy projects the transformed ones.",
    "StatisBiplot transform": "StatisBiplot and CanonicalStatisBiplot pass InitTransform positionally to the wrong argument of TransformIni, so the data are always standardized whatever is requested; biplotpy honours `transform` (the default, 5, gives identical results).",
    "StatisBiplot column names": "StatisBiplot fails when the tables have no column names; biplotpy generates them.",
    "DualStatisBiplot": "DualStatisBiplot is unfinished in MultBiplotR (stops with a dimnames error); biplotpy implements the full method, and its core (pooled SD, RV, weights, compromise) is validated against MultBiplotR's own lines.",
    "CanonicalStatisBiplot within": "CanonicalStatisBiplot stores the group means in $Within; biplotpy stores the within-group SSCP matrices.",
    "BinaryLogBiplotGD": "MultBiplotR's gradient-descent binary logistic biplot divides the log-likelihood by 2 in the objective but not in the gradient; biplotpy minimizes a single consistent penalized objective, validated against R optim on that objective.",
    "NIPALS.Biplot": "MultBiplotR's NIPALS.Biplot multiplies the row scores X V by the singular values again (rows = U D^2), so the biplot does not reproduce the data; biplotpy uses U D (and ade4's algorithm when there are missing values, which MultBiplotR does not handle).",
    "Coinertia": "MultBiplotR's Coinertia replaces Y by X (Y = InitialTransform(X, ...)) and returns no coordinates; biplotpy follows ade4::coinertia.",
    "BinaryDistances": "In MultBiplotR's BinaryDistances, Kulczynski2 is unreachable (the switch has Kulezynski twice), the sqrt(2(1-S)) transformation uses == instead of = (error), and 1/(S+1) computes 1/S + 1; biplotpy implements the stated formulas.",
    "PrincipalCoordinates weights": "MultBiplotR's PrincipalCoordinates centres with (I - 1 w) on both sides, which is not the weighted double centring unless the weights are equal; biplotpy uses (I - 1 w') b (I - w 1'). Equal weights give identical results.",
    "FA.Biplot scores": "MultBiplotR's FA.Biplot calls 'Regression' the Bartlett formula and vice versa, and squares the uniquenesses in Psi^-1; biplotpy uses the standard formulas (validated against psych::factor.scores).",
    "PolyBiplot": "MultBiplotR's PolyBiplot multiplies the loadings and thresholds by sqrt(1 - h^2) instead of dividing (normal-ogive to item parameters) and uses unpenalized scores that diverge for all-extreme response patterns; biplotpy divides, rescales by 1.702 to the logistic metric and uses a standard normal prior on the scores.",
    "PLSR": "MultBiplotR's PLSR normalizes the scores by their squared norm and reports W C' as regression coefficients without (P'W)^-1, which gives wrong coefficients; biplotpy follows pls::plsr (oscorespls) and reproduces its scores, loadings, coefficients and fitted values.",
    "Unfolding stress at perfect fit": "smacof can return NaN stress-1 when the fit is exact (1 - x slightly negative by rounding); biplotpy truncates at 0.",
}


def transform_name(t):
    return TRANSFORMS[t - 1] if isinstance(t, (int, np.integer)) else t


def _sd(a, axis):
    return np.std(a, axis=axis, ddof=1)


def _rank(v):
    from scipy.stats import rankdata
    return rankdata(v, method="average")


def initial_transform(X, transform="Standardize columns", sup_rows=None, sup_cols=None, grouping=None,
                      init_transform="None"):
    """Transform ``X`` (and optional supplementary rows/columns).

    Returns a dict with ``X``, ``sup_rows``, ``sup_cols`` and the statistics used
    (``col_means``, ``col_sds``, ``gmean``, ``col_ranges``)."""
    X = np.array(X, float)
    n, p = X.shape
    transform = transform_name(transform)
    sr = None if sup_rows is None else np.array(sup_rows, float)
    sc = None if sup_cols is None else np.array(sup_cols, float)
    if init_transform == "Log":
        if np.any(X <= 0):
            raise ValueError("Initial log transformation needs positive values")
        X = np.log(X)
        sr = None if sr is None else np.log(sr)
        sc = None if sc is None else np.log(sc)
    elif init_transform == "Logit":
        X = X + 0.01 * (X == 0) - 0.01 * (X == 1)
        X = np.log(X / (1 - X))
    out = {"transform": transform, "gmean": X.mean(), "col_means": X.mean(0), "col_sds": _sd(X, 0),
           "row_means": X.mean(1), "row_sds": _sd(X, 1) if p > 1 else np.full(n, np.nan), "col_ranges": X.max(0) - X.min(0)}
    nsup = (0 if sr is None else len(sr)) + (0 if sc is None else sc.shape[1])
    if transform in ("Double centering", "Divide by the column means and center",
                     "Normalized residuals from independence") and nsup > 0:
        raise ValueError(f"'{transform}' is not compatible with supplementary rows or columns")
    cm, cs = out["col_means"], out["col_sds"]
    if transform == "Raw Data":
        pass
    elif transform == "Substract the global mean":
        X = X - out["gmean"]
        sr = None if sr is None else sr - out["gmean"]
        sc = None if sc is None else sc - out["gmean"]
    elif transform == "Double centering":
        X = X - X.mean(0) - X.mean(1)[:, None] + X.mean()
    elif transform == "Column centering":
        X = X - cm
        sr = None if sr is None else sr - cm
        sc = None if sc is None else sc - sc.mean(0)
    elif transform == "Standardize columns":
        X = (X - cm) / cs
        sr = None if sr is None else (sr - cm) / cs
        sc = None if sc is None else (sc - sc.mean(0)) / _sd(sc, 0)
    elif transform == "Row centering":
        X = X - X.mean(1)[:, None]
        sr = None if sr is None else sr - sr.mean(1)[:, None]
        sc = None if sc is None else sc - sc.mean(1)[:, None]
    elif transform == "Standardize rows":
        rs = out["row_sds"]
        X = (X - X.mean(1)[:, None]) / rs[:, None]
        sr = None if sr is None else (sr - sr.mean(1)[:, None]) / _sd(sr, 1)[:, None]
        sc = None if sc is None else (sc - out["row_means"][:, None]) / rs[:, None]
    elif transform == "Divide by the column means and center":
        X = X / cm
        X = X - X.mean(0)
    elif transform == "Normalized residuals from independence":
        nt = X.sum()
        dr, dc = X.sum(1), X.sum(0)
        esp = np.outer(dr, dc) / nt
        var = np.outer(1 - dr / nt, 1 - dc / nt)
        X = ((X - esp) / np.sqrt(esp)) / np.sqrt(var)
    elif transform == "Divide by the range":
        X = X / out["col_ranges"]
    elif transform == "Within groups standardization":
        if grouping is None:
            raise ValueError("A grouping factor is needed for the within groups standardization")
        g = np.asarray(grouping)
        levs = list(dict.fromkeys(g))
        Xc = X - cm
        wss = np.zeros(p)
        for lev in levs:
            m = g == lev
            wss += ((X[m] - X[m].mean(0)) ** 2).sum(0)
        sdw = np.sqrt(wss / (n - len(levs)))
        out["col_sds"] = sdw
        X = Xc / sdw
    elif transform == "Ranks":
        X = np.column_stack([_rank(X[:, j]) for j in range(p)])
    else:
        raise ValueError(f"Unknown transformation: {transform}")
    out.update(X=X, sup_rows=sr, sup_cols=sc)
    return out
