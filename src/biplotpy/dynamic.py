"""Dynamic biplot (Egido & Galindo, 2015).

A reference biplot (HJ by default) is computed on a reference matrix (the mean
of the occasions or one chosen occasion). The data of every occasion are
transformed with the statistics of the reference and projected as
supplementary rows (and, optionally, columns), giving the trajectory of each
individual and variable in the reference plane.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classical import pca_biplot


def dynamic_biplot(X, reference="mean", alpha=2, dim=2, scaling=5, names=None, row_names=None,
                   col_names=None) -> dict:
    """``X``: list (or dict) of n x p matrices with the same rows and columns.

    Returns the reference biplot, the coordinates of the rows and columns at
    every occasion, displacement (first to last occasion) and path length of
    each row, and the quality (%) of the projection of each row at each
    occasion (squared cosine on the reference plane)."""
    if isinstance(X, dict):
        names = names or list(X)
        X = list(X.values())
    if isinstance(X[0], pd.DataFrame):
        row_names = row_names or [str(i) for i in X[0].index]
        col_names = col_names or [str(c) for c in X[0].columns]
    X = [np.asarray(x, float) for x in X]
    T = len(X)
    names = names or [f"t{i + 1}" for i in range(T)]
    ref = np.mean(X, axis=0) if reference == "mean" else X[int(reference)]
    bip = pca_biplot(ref, alpha=alpha, dim=dim, scaling=scaling, row_names=row_names, col_names=col_names)
    mu, sd = ref.mean(0), ref.std(0, ddof=1)
    t = bip.transformation
    if t == "Standardize columns":
        Z = [(x - mu) / sd for x in X]
    elif t == "Column centering":
        Z = [x - mu for x in X]
    elif t == "Raw Data":
        Z = X
    else:
        raise ValueError("dynamic_biplot supports raw data, column centering and standardized columns")
    d = bip.d[:dim]
    U, V = bip.U, bip.V
    if alpha > 1:
        rf, cf = np.ones(dim), np.ones(dim)
    else:
        rf = (1 / d) ** (1 - alpha) * bip.scale_factor
        cf = (1 / d) ** alpha / bip.scale_factor
    rows = [z @ V * rf for z in Z]                   # supplementary rows
    cols = [z.T @ U * cf for z in Z]                 # supplementary columns
    qual = np.array([((z @ V) ** 2).sum(1) / np.maximum((z ** 2).sum(1), 1e-300) * 100 for z in Z]).T
    traj = np.stack(rows, axis=1)                    # n x T x dim
    disp = np.sqrt(((traj[:, -1] - traj[:, 0]) ** 2).sum(1))
    path = np.sqrt((np.diff(traj, axis=1) ** 2).sum(2)).sum(1)
    rn = bip.row_names
    return {"reference": bip, "row_coordinates": dict(zip(names, rows)), "col_coordinates": dict(zip(names, cols)),
            "trajectories": traj, "displacement": pd.Series(disp, index=rn), "path_length": pd.Series(path, index=rn),
            "quality": pd.DataFrame(qual, index=rn, columns=names), "occasions": names}
