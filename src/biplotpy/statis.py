"""STATIS-ACT family (port of MultBiplotR::StatisBiplot, DualStatisBiplot and
CanonicalStatisBiplot; Lavit et al., 1994; Vallejo-Arboleda,
Vicente-Villardón & Galindo-Villardón, 2007).

* ``statis``: several tables with the same individuals (rows). Objects
  W_t = X_t X_t'; interstructure from the RV matrix; compromise
  W = sum alpha_t W_t with alpha from the first eigenvector; biplot of the
  compromise with the variables of every table.
* ``dual_statis``: several tables with the same variables (columns). Objects
  W_t = X_t' X_t (cross-product matrices); compromise of the variables.
* ``canonical_statis``: several tables with the same individuals and a group
  structure; objects W_t = Xb_t S_t^{-1} Xb_t' (Mahalanobis scalar products
  between group means), giving a compromise canonical biplot of the groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transforms import initial_transform, transform_name


class StatisResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def plot(self, *args, **kwargs):
        from .plot import plot_statis
        return plot_statis(self, *args, **kwargs)

    def __repr__(self):
        return (f"{self.title}: {self.n_tables} tables\nRV matrix:\n{self.RV.round(3).to_string()}\n"
                f"Weights: {np.round(self.weights, 4)}\nEigenvalues of the compromise: {np.round(self.eigenvalues[:5], 4)}")


def _interstructure(Wt, names):
    ng = len(Wt)
    P = np.array([[np.sum(Wt[i] * Wt[j]) for j in range(ng)] for i in range(ng)])
    s = 1 / np.sqrt(np.diag(P))
    RV = P * s[:, None] * s[None, :]
    u, d, _ = np.linalg.svd(RV)
    weights = np.abs(u[:, 0]) / np.abs(u[:, 0]).sum()
    inter = -1 * u * np.sqrt(d)
    return (pd.DataFrame(P, index=names, columns=names), pd.DataFrame(RV, index=names, columns=names),
            d, d / d.sum() * 100, inter, weights)


def _compromise(Wt, weights):
    W = sum(w * M for w, M in zip(weights, Wt))
    U, d, _ = np.linalg.svd(W)
    r = int(np.sum(d > 1e-5))
    return W, U, d, r


def statis(X, transform=5, dim=2, names=None) -> StatisResult:
    """STATIS-ACT for tables with the same rows (list or dict of n x p_t matrices)."""
    if isinstance(X, dict):
        names = names or list(X)
        X = list(X.values())
    names = names or [f"Occasion_{i + 1}" for i in range(len(X))]
    col_names = [list(x.columns) if isinstance(x, pd.DataFrame) else [f"V{j + 1}" for j in range(np.shape(x)[1])] for x in X]
    X = [initial_transform(np.asarray(x, float), transform_name(transform))["X"] for x in X]
    n = X[0].shape[0]
    if any(x.shape[0] != n for x in X):
        raise ValueError("The number of rows must be the same in all occasions")
    Wt = [x @ x.T for x in X]
    P, RV, eig, iner, inter, w = _interstructure(Wt, names)
    W, U, d, r = _compromise(Wt, w)
    rows = U[:, :dim] * np.sqrt(d[:dim])
    sf = ((U[:, :r] * np.sqrt(d[:r])) ** 2).sum(1)
    row_contr = rows ** 2 / sf[:, None] * 100
    trajin = [Wt[j] @ U[:, :r] * np.sqrt(1 / d[:r]) for j in range(len(X))]
    contvar, cols, col_labels = [], [], []
    for j, x in enumerate(X):
        tv = x.T @ U[:, :r]
        contvar.append(tv ** 2 / (tv ** 2).sum(1, keepdims=True) * 100)
        cols.append(x.T @ U[:, :r] / d[:r])
        col_labels += [f"{c}-{names[j]}" for c in col_names[j]]
    col_coords = np.vstack([c[:, :dim] for c in cols])
    col_contr = np.vstack([c[:, :dim] for c in contvar])
    p = col_coords.shape[0]
    sca = (rows ** 2).sum() / n
    scb = (col_coords ** 2).sum() / p
    scf = np.sqrt(np.sqrt(scb / sca))
    return StatisResult(title="STATIS-ACT", n_tables=len(X), names=names, data=X, objects=Wt, scalar_products=P,
                        RV=RV, eig_inter=eig, inertia_inter=iner, interstructure=inter, weights=w, compromise=W,
                        eigenvalues=d[:r], inertia=d[:r] / d[:r].sum() * 100,
                        row_coordinates=rows * scf, row_contributions=row_contr,
                        col_coordinates=col_coords / scf, col_contributions=col_contr, col_labels=col_labels,
                        scale_factor=scf, trajectories=[t[:, :dim] * scf for t in trajin])


def _common_sd_transform(X, transform):
    if transform_name(transform) == "Standardize columns":
        nr = np.array([x.shape[0] for x in X])
        sds = np.array([np.std(x, axis=0, ddof=1) for x in X])
        common = np.sqrt(((nr - 1)[:, None] * sds ** 2).sum(0) / (nr - 1).sum())
        return [(x - x.mean(0)) / common for x in X], common
    return [initial_transform(x, transform_name(transform))["X"] for x in X], None


def dual_statis(X, transform=5, dim=2, names=None) -> StatisResult:
    """Dual STATIS for tables with the same variables (columns). Standardization
    uses the pooled within-table standard deviation (as MultBiplotR)."""
    if isinstance(X, dict):
        names = names or list(X)
        X = list(X.values())
    names = names or [f"Occasion_{i + 1}" for i in range(len(X))]
    var_names = list(X[0].columns) if isinstance(X[0], pd.DataFrame) else [f"V{j + 1}" for j in range(np.shape(X[0])[1])]
    X, common = _common_sd_transform([np.asarray(x, float) for x in X], transform)
    Wt = [x.T @ x for x in X]
    P, RV, eig, iner, inter, w = _interstructure(Wt, names)
    W, U, d, r = _compromise(Wt, w)
    cols = U[:, :dim] * np.sqrt(d[:dim])
    sf = ((U[:, :r] * np.sqrt(d[:r])) ** 2).sum(1)
    contr = cols ** 2 / sf[:, None] * 100
    rows = [x @ U[:, :dim] for x in X]     # individuals of each table projected on the compromise axes
    return StatisResult(title="Dual STATIS-ACT", n_tables=len(X), names=names, data=X, objects=Wt,
                        scalar_products=P, RV=RV, eig_inter=eig, inertia_inter=iner, interstructure=inter, weights=w,
                        compromise=W, eigenvalues=d[:r], inertia=d[:r] / d[:r].sum() * 100, col_coordinates=cols,
                        col_contributions=contr, var_names=var_names, row_coordinates=rows, common_sd=common)


def canonical_statis(X, groups, transform=5, dim=2, names=None) -> StatisResult:
    """Canonical STATIS-ACT (Vallejo-Arboleda et al., 2007)."""
    if isinstance(X, dict):
        names = names or list(X)
        X = list(X.values())
    names = names or [f"Occasion_{i + 1}" for i in range(len(X))]
    g = np.asarray([str(v) for v in groups])
    levels = list(groups.cat.categories) if hasattr(groups, "cat") else sorted(set(g))
    levels = [str(v) for v in levels]
    X = [initial_transform(np.asarray(x, float), transform_name(transform))["X"] for x in X]
    Z = np.column_stack([(g == lev).astype(float) for lev in levels])
    ng = Z.sum(0)
    Xbt, Bt, St, Wt = [], [], [], []
    for x in X:
        Xb = (Z.T @ x) / ng[:, None]
        B = Xb.T @ np.diag(ng) @ Xb
        S = x.T @ x - B
        Xbt.append(Xb); Bt.append(B); St.append(S)
        Wt.append(Xb @ np.linalg.solve(S, Xb.T))
    P, RV, eig, iner, inter, w = _interstructure(Wt, names)
    W, U, d, r = _compromise(Wt, w)
    rows = U[:, :dim] * np.sqrt(d[:dim])
    sf = ((U[:, :r] * np.sqrt(d[:r])) ** 2).sum(1)
    contr = rows ** 2 / sf[:, None] * 100
    trajin = [Wt[j] @ U[:, :r] * np.sqrt(1 / d[:r]) for j in range(len(X))]
    cols, ccontr = [], []
    for xb in Xbt:
        tv = xb.T @ U[:, :r]
        ccontr.append((tv ** 2 / (tv ** 2).sum(1, keepdims=True) * 100)[:, :dim])
        cols.append((tv / d[:r])[:, :dim])
    col_coords = np.vstack(cols)
    p = col_coords.shape[0]
    scf = np.sqrt(np.sqrt(((col_coords ** 2).sum() / p) / ((rows ** 2).sum() / len(levels))))
    return StatisResult(title="Canonical STATIS-ACT", n_tables=len(X), names=names, levels=levels, data=X,
                        means=Xbt, between=Bt, within=St, objects=Wt, scalar_products=P, RV=RV, eig_inter=eig,
                        inertia_inter=iner, interstructure=inter, weights=w, compromise=W, eigenvalues=d[:r],
                        inertia=d[:r] / d[:r].sum() * 100, row_coordinates=rows * scf, row_contributions=contr,
                        col_coordinates=col_coords / scf, col_contributions=np.vstack(ccontr), scale_factor=scf,
                        trajectories=[t[:, :dim] * scf for t in trajin])
