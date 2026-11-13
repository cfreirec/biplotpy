"""Bootstrap of the PCA biplot, supplementary variables and calibrated axes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transforms import initial_transform, transform_name
from .utils import r_pretty


# --------------------------------------------------------------------------- bootstrap
def pca_bootstrap(X, dim=2, scaling="Standardize columns", B=1000, seed=None, level=0.95) -> dict:
    """Non-parametric bootstrap of the PCA biplot (port of ``PCA.Bootstrap``).

    Individuals are resampled, the data re-transformed and the SVD recomputed;
    each axis is reflected to agree in sign with the original solution. Returns
    the replicated eigenvalues, explained inertia, loadings, structure
    correlations and qualities of the variables, with percentile and normal
    ("basic", as in MultBiplotR) intervals for the eigenvalues and percentile
    intervals for the rest.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    n, p = X.shape
    XT = initial_transform(X, transform_name(scaling))["X"]
    U0, d0, V0t = np.linalg.svd(XT, full_matrices=False)
    V0 = V0t.T[:, :dim]
    k = min(n, p)
    eig = np.zeros((B, k)); iner = np.zeros((B, k))
    Vs = np.zeros((B, p, dim)); struct = np.zeros((B, p, dim)); qual = np.zeros((B, p, dim))
    samples = np.zeros((B, n), dtype=int)
    for i in range(B):
        s = rng.integers(n, size=n)
        samples[i] = s
        XB = initial_transform(X[s], transform_name(scaling))["X"]
        U, d, Vt = np.linalg.svd(XB, full_matrices=False)
        V = Vt.T[:, :dim]
        signs = np.sign(np.diag(V0.T @ V)); signs[signs == 0] = 1
        V = V * signs
        U = U[:, :dim] * signs
        eig[i, :len(d)] = d[:k]
        iner[i, :len(d)] = 100 * d[:k] ** 2 / np.sum(d ** 2)
        Vs[i] = V
        A = XB @ V
        struct[i] = np.corrcoef(np.column_stack([XB, A]), rowvar=False)[:p, p:]
        b = V * d[:dim]
        qual[i] = np.cumsum(b ** 2 / (XB ** 2).sum(0)[:, None] * 100, axis=1)
    a = (1 - level) / 2
    ev2 = eig ** 2
    med = ev2.mean(0)
    sdev = ev2.std(0, ddof=1)
    return {"eigenvalues": eig, "inertia": iner, "loadings": Vs, "structure": struct, "qualities": qual,
            "samples": samples, "initial_svd": (U0[:, :dim], d0, V0),
            "mean_eigenvalue": med,
            "ci_percentile_eigenvalue": np.quantile(ev2, [a, 1 - a], axis=0),
            "ci_basic_eigenvalue": np.vstack([med - 1.96 * sdev, med + 1.96 * sdev]),
            "ci_structure": np.quantile(struct, [a, 1 - a], axis=0),
            "ci_qualities": np.quantile(qual, [a, 1 - a], axis=0),
            "ci_inertia": np.quantile(iner, [a, 1 - a], axis=0)}


# --------------------------------------------------------------------------- supplementary
def add_continuous_variables(bip, X, dims=None, scaling=5, names=None) -> dict:
    """Project continuous supplementary variables on a biplot by regressing each
    (transformed) variable on the row coordinates (port of ``AddContVars2Biplot``).

    Returns intercepts ``b0``, coordinates, R2 (%) and structure correlations."""
    if isinstance(X, pd.DataFrame):
        names = names or [str(c) for c in X.columns]
        X = X.to_numpy(float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    dims = dims or bip.row_coordinates.shape[1]
    Xs = initial_transform(X, transform_name(scaling))["X"]
    A = bip.row_coordinates[:, :dims]
    DE = np.column_stack([np.ones(n), A])
    b = np.linalg.solve(DE.T @ DE, DE.T @ Xs).T
    res = Xs - DE @ b.T
    R2 = (1 - (res ** 2).sum(0) / (Xs ** 2).sum(0)) * 100
    structure = np.corrcoef(np.column_stack([Xs, A]), rowvar=False)[:p, p:]
    names = names or [f"SV{j + 1}" for j in range(p)]
    return {"b0": b[:, 0], "coordinates": b[:, 1:], "R2": R2, "structure": structure, "names": names,
            "scaled_data": Xs}


def project_rows(bip, X_new):
    """Coordinates of supplementary individuals (transformed with the statistics
    of the active data) on a classical biplot."""
    X_new = np.asarray(X_new, float)
    t = bip.transformation
    if t == "Standardize columns":
        Z = (X_new - bip.means) / bip.deviations
    elif t == "Column centering":
        Z = X_new - bip.means
    elif t == "Substract the global mean":
        Z = X_new - bip.gmean
    elif t == "Raw Data":
        Z = X_new
    else:
        raise ValueError(f"Supplementary rows are not defined for '{t}'")
    a = Z @ bip.V
    d = bip.d[:bip.dim]
    if bip.alpha <= 1:
        a = a * (1 / d) ** (1 - bip.alpha) * bip.scale_factor
    return a  # HJ-Biplot: principal coordinates X V = U D


# --------------------------------------------------------------------------- calibrated axes
def biplot_scales(bip, nticks=3, type_scale="Complete", values_scale="Original") -> dict:
    """Ticks (on the transformed scale) and labels (on the original or transformed
    scale) of the calibrated axis of every variable (port of ``GetBiplotScales``
    for raw data, global-mean, column-centred and standardized data)."""
    ticks, labels = [], []
    t = bip.transformation
    for j in range(bip.p):
        mn, mx, mean, sd = bip.minima[j], bip.maxima[j], bip.means[j], bip.deviations[j]
        if t in ("Raw Data", "Substract the global mean"):
            lab = r_pretty([mn, mx], nticks)
            tk = lab - (bip.gmean if t == "Substract the global mean" else 0)
        elif t == "Column centering":
            if type_scale == "Complete":
                if values_scale == "Original":
                    lab = r_pretty([mn, mx], nticks); tk = lab - mean
                else:
                    tk = r_pretty([mn - mean, mx - mean], nticks); lab = np.round(tk, 2)
            elif type_scale == "StdDev":
                k = np.array([-3, -2, -1, 1, 2, 3])
                tk = np.round(k * sd, 2); lab = np.round(mean + k * sd, 2) if values_scale == "Original" else tk
            else:
                q = np.array([mn, bip.p25[j], bip.medians[j], bip.p75[j], mx])
                tk = q - mean; lab = q if values_scale == "Original" else np.round(tk, 2)
        elif t == "Standardize columns":
            if type_scale == "Complete":
                if values_scale == "Original":
                    lab = r_pretty([mn, mx], nticks); tk = (lab - mean) / sd
                else:
                    tk = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]); lab = np.round(tk, 2)
            elif type_scale == "StdDev":
                tk = np.array([-3., -2, -1, 1, 2, 3])
                lab = np.round(tk * sd + mean, 1) if values_scale == "Original" else np.round(tk, 2)
            else:
                q = np.array([mn, bip.p25[j], bip.medians[j], bip.p75[j], mx])
                tk = (q - mean) / sd; lab = q if values_scale == "Original" else np.round(tk, 2)
        else:
            col = bip.scaled_data[:, j]
            lab = r_pretty([col.min(), col.max()], nticks); tk = lab
        ticks.append(np.asarray(tk, float)); labels.append(np.asarray(lab, float))
    return {"ticks": ticks, "labels": labels}


def calibrated_axis(b, ticks):
    """Positions of the ticks of a calibrated (predictive) axis with direction
    ``b`` (column marker): the point of the axis whose projection predicts the
    value ``t`` is ``t * b / |b|^2``."""
    b = np.asarray(b, float)
    return np.outer(np.asarray(ticks, float), b / (b @ b))
