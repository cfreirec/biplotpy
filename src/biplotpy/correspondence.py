"""Correspondence-type biplots: simple and multiple correspondence analysis,
canonical correspondence analysis (CCA) and redundancy analysis (RDA).

* ``ca``  — port of ``MultBiplotR::CA`` (alpha = 1 rows principal / columns
  standard, 0 the reverse, 0.5 symmetric biplot, 2 = both principal, the
  usual "symmetric map" of ``ca``/``FactoMineR``); also returns the absolute
  contributions (to the axis) of ``ca::ca``/``FactoMineR::CA``.
* ``mca`` — multiple correspondence analysis as the CA of the indicator matrix
  (``FactoMineR::MCA`` conventions), with the Benzécri and Greenacre adjusted
  inertias.
* ``cca`` — port of ``MultBiplotR::CCA`` (ter Braak, 1986), eigenvalues equal
  to ``vegan::cca``.
* ``rda`` — redundancy analysis with the scores and scalings of ``vegan::rda``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transforms import initial_transform, transform_name


class OrdinationBiplot:
    """Result of a CA/MCA/CCA/RDA/PCoA/MDS-type biplot."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        ev = np.asarray(self.eigenvalues)[:5]
        return f"{self.title}: eigenvalues {np.round(ev, 4)}; inertia (%) {np.round(np.asarray(self.inertia)[:5], 2)}"

    def plot(self, *args, **kwargs):
        from .plot import plot_ordination
        return plot_ordination(self, *args, **kwargs)


def _names(x, n, p, rp="I", cp="V"):
    if isinstance(x, pd.DataFrame):
        return [str(i) for i in x.index], [str(c) for c in x.columns]
    return [f"{rp}{i + 1}" for i in range(n)], [f"{cp}{j + 1}" for j in range(p)]


# --------------------------------------------------------------------------- CA
def ca(N, dim=2, alpha=2.0, sup_rows=None, sup_cols=None) -> OrdinationBiplot:
    """Simple correspondence analysis of a contingency table ``N``.

    ``alpha``: 2 = symmetric map (rows and columns in principal coordinates);
    1 = row principal (RMP); 0 = column principal (CMP); 0.5 = symmetric
    biplot (MultBiplotR's convention for alpha <= 1)."""
    rn, cn = _names(N, *np.shape(N))
    X = np.asarray(N, float)
    n, p = X.shape
    tot = X.sum()
    P = X / tot
    r, c = P.sum(1), P.sum(0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, d, Vt = np.linalg.svd(S, full_matrices=False)
    k = min(n, p) - 1
    U, d, V = U[:, :k], d[:k], Vt.T[:, :k]
    Fs, Gs = U / np.sqrt(r)[:, None], V / np.sqrt(c)[:, None]        # standard coordinates
    F, G = Fs * d, Gs * d                                              # principal coordinates
    ev = d ** 2
    inertia = ev / ev.sum() * 100
    qual_r = F ** 2 / (F ** 2).sum(1, keepdims=True) * 100
    qual_c = G ** 2 / (G ** 2).sum(1, keepdims=True) * 100
    ctr_r = r[:, None] * F ** 2 / ev * 100
    ctr_c = c[:, None] * G ** 2 / ev * 100
    if alpha > 1:
        A, B = F, G
    else:
        A, B = Fs * d ** alpha, Gs * d ** (1 - alpha)
    res = OrdinationBiplot(title="Correspondence Analysis", type="CA", alpha=alpha, dim=dim, eigenvalues=ev,
                           inertia=inertia, cum_inertia=np.cumsum(inertia), total_inertia=ev.sum(),
                           chi2=ev.sum() * tot, row_coordinates=A[:, :dim], col_coordinates=B[:, :dim],
                           row_principal=F, col_principal=G, row_standard=Fs, col_standard=Gs,
                           row_masses=r, col_masses=c, row_qualities=qual_r[:, :dim], col_qualities=qual_c[:, :dim],
                           row_contributions=ctr_r[:, :dim], col_contributions=ctr_c[:, :dim],
                           row_names=rn, col_names=cn, data=X)
    if sup_rows is not None:
        R = np.asarray(sup_rows, float); R = R / R.sum(1, keepdims=True)
        res.sup_row_coordinates = (R @ Gs)[:, :dim]
    if sup_cols is not None:
        C = np.asarray(sup_cols, float); C = C / C.sum(0, keepdims=True)
        res.sup_col_coordinates = (C.T @ Fs)[:, :dim]
    return res


# --------------------------------------------------------------------------- MCA
def indicator_matrix(X):
    """Complete disjunctive table of a data frame of categorical variables."""
    df = pd.DataFrame(X).astype(str)
    cols, levs, var_of = [], [], []
    for v in df.columns:
        for lev in sorted(df[v].unique()):
            cols.append((df[v] == lev).astype(float).to_numpy())
            levs.append(lev); var_of.append(v)
    # category labels as in FactoMineR: the level itself when unique, else variable_level
    names = levs if len(set(levs)) == len(levs) else [f"{v}_{l}" for v, l in zip(var_of, levs)]
    return np.column_stack(cols), names, var_of


def mca(X, dim=2, sup_rows=None) -> OrdinationBiplot:
    """Multiple correspondence analysis (CA of the indicator matrix).

    Coordinates follow ``FactoMineR::MCA``: individuals and categories in
    principal coordinates; eigenvalues of the indicator matrix; also the
    Benzécri-corrected and Greenacre-adjusted percentages of inertia."""
    df = pd.DataFrame(X)
    rn = [str(i) for i in df.index]
    Z, names, var_of = indicator_matrix(df)
    Q = df.shape[1]
    res = ca(pd.DataFrame(Z, index=rn, columns=names), dim=max(dim, 2), alpha=2)
    ev = res.eigenvalues
    keep = ev > 1 / Q
    benz = ((Q / (Q - 1)) * (ev[keep] - 1 / Q)) ** 2
    J = Z.shape[1]
    green_total = Q / (Q - 1) * (np.sum(ev ** 2) - (J - Q) / Q ** 2)
    res.title, res.type = "Multiple Correspondence Analysis", "MCA"
    res.row_coordinates, res.col_coordinates = res.row_principal[:, :dim], res.col_principal[:, :dim]
    res.benzecri = benz / benz.sum() * 100
    res.greenacre = benz / green_total * 100
    res.variables = var_of
    res.eta2 = np.array([[np.sum(res.col_masses[np.array(var_of) == v] * res.col_principal[np.array(var_of) == v, k] ** 2)
                          * Q for k in range(dim)] for v in df.columns])
    res.var_names = list(df.columns)
    return res


# --------------------------------------------------------------------------- CCA
def _msqrt(S, inv=False):
    w, U = np.linalg.eigh(S)
    w = np.maximum(w, 0)
    return U @ np.diag(1 / np.sqrt(w) if inv else np.sqrt(w)) @ U.T


def cca(Y, Z, alpha=1.0, dim=2) -> OrdinationBiplot:
    """Canonical correspondence analysis of an abundance table ``Y`` (sites x
    species) constrained by environmental variables ``Z`` (sites x variables).

    Port of ``MultBiplotR::CCA``: species scores, site scores (weighted
    averages, WA) and linear-combination site scores (LC), environmental
    variable scores (biplot arrows), qualities and inertia decompositions.
    ``alpha`` = 1 gives species in principal coordinates."""
    sn, spn = _names(Y, *np.shape(Y), "Si", "Sp")
    en = list(Z.columns) if isinstance(Z, pd.DataFrame) else [f"Ev{k + 1}" for k in range(np.shape(Z)[1])]
    P = np.asarray(Y, float); Zm = np.asarray(Z, float)
    P = P / P.sum()
    r, c = P.sum(1), P.sum(0)
    PC = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    valsin = np.linalg.svd(PC, compute_uv=False)
    in_total = np.sum(PC ** 2)
    iner_rows, iner_cols = (PC ** 2).sum(1), (PC ** 2).sum(0)
    W1 = (P.T @ Zm) / c[:, None]
    mean_z = r @ Zm
    Zc = Zm - mean_z
    s11 = r @ Zc ** 2
    Zs = Zc / np.sqrt(s11)
    W = (P.T @ Zs) / c[:, None]
    S22 = Zs.T @ (r[:, None] * Zs)
    sq, isq = _msqrt(S22), _msqrt(S22, inv=True)
    WW = np.sqrt(c)[:, None] * W @ isq
    U, d, Vt = np.linalg.svd(WW, full_matrices=False)
    V = Vt.T
    x = (P / r[:, None]) @ (U / np.sqrt(c)[:, None])
    x2 = Zs @ isq @ V
    A = (U / np.sqrt(c)[:, None]) * d ** alpha
    B = sq @ V * d ** (1 - alpha)
    Afull = (U / np.sqrt(c)[:, None]) * d
    Bfull = sq @ V * d
    sp_q = Afull ** 2 / (Afull ** 2).sum(1, keepdims=True) * 100
    ev_q = Bfull ** 2 / (Bfull ** 2).sum(1, keepdims=True) * 100
    site_q = (np.sqrt(r)[:, None] * x) ** 2 / iner_rows[:, None] * 100
    sp_q_ca = (U * d) ** 2 / iner_cols[:, None] * 100
    interset = np.corrcoef(np.column_stack([x, Zm]), rowvar=False)[:x.shape[1], x.shape[1]:]
    ev = d ** 2
    return OrdinationBiplot(title="Canonical Correspondence Analysis", type="CCA", alpha=alpha, dim=dim,
                            eigenvalues=ev, inertia=ev / ev.sum() * 100, exp_species_constrained=ev / in_total * 100,
                            exp_species_ca=valsin ** 2 / in_total * 100, total_inertia=in_total,
                            row_coordinates=(x * d ** (-alpha))[:, :dim], lc_site_scores=(x2 * d ** (-alpha))[:, :dim],
                            col_coordinates=A[:, :dim], env_coordinates=B[:, :dim], weighted_averages=W1,
                            species_quality=sp_q[:, :dim], species_quality_ca=sp_q_ca[:, :dim],
                            env_quality=ev_q[:, :dim], site_quality=site_q[:, :dim], interset=interset[:, :dim],
                            row_names=sn, col_names=spn, env_names=en, means_z=mean_z, sd_z=np.sqrt(s11))


# --------------------------------------------------------------------------- RDA
def rda(Y, X, scale=False, scaling=2, dim=2) -> OrdinationBiplot:
    """Redundancy analysis (constrained PCA) of ``Y`` on ``X``.

    Scores and scalings follow ``vegan::rda``/``scores.rda`` (scaling 1 =
    sites, 2 = species, 3 = symmetric; constant sqrt(sqrt((n-1) * total
    variance))): site scores (WA), constrained site scores (LC), species scores
    and biplot scores of the explanatory variables; unconstrained (residual)
    axes are returned as ``ca_eigenvalues``."""
    sn, spn = _names(Y, *np.shape(Y), "S", "Sp")
    xn = list(X.columns) if isinstance(X, pd.DataFrame) else [f"X{k + 1}" for k in range(np.shape(X)[1])]
    Ym = np.asarray(Y, float); Xm = np.asarray(X, float)
    n = Ym.shape[0]
    Yc = Ym - Ym.mean(0)
    if scale:
        Yc = Yc / Ym.std(0, ddof=1)
    Ys = Yc / np.sqrt(n - 1)
    tot = np.sum(Ys ** 2)
    Xc = Xm - Xm.mean(0)
    Qx, _ = np.linalg.qr(Xc)
    rank = np.linalg.matrix_rank(Xc)
    Qx = Qx[:, :rank]
    Fit = Qx @ (Qx.T @ Ys)
    U, d, Vt = np.linalg.svd(Fit, full_matrices=False)
    k = int(np.sum(d > 1e-10 * d[0]))
    U, d, V = U[:, :k], d[:k], Vt.T[:, :k]
    ev = d ** 2
    wa = Ys @ V / d
    biplot = np.corrcoef(np.column_stack([Xm, U]), rowvar=False)[:Xm.shape[1], Xm.shape[1]:]
    Res = Ys - Fit
    res_ev = np.linalg.svd(Res, compute_uv=False) ** 2
    res_ev = res_ev[res_ev > 1e-10 * max(res_ev.max(), 1e-300)]
    const = np.sqrt(np.sqrt((n - 1) * tot))
    eig_all = np.concatenate([ev, res_ev])
    slam = np.sqrt(ev / tot)
    sp_s, site_s = {1: (1, slam), 2: (slam, 1), 3: (np.sqrt(slam), np.sqrt(slam))}[scaling]
    species = const * V * sp_s
    sites = const * wa * site_s
    lc = const * U * site_s
    bp = biplot * site_s
    return OrdinationBiplot(title="Redundancy Analysis", type="RDA", dim=dim, scaling=scaling, eigenvalues=ev,
                            inertia=ev / tot * 100, ca_eigenvalues=res_ev, total_inertia=tot,
                            constrained_proportion=ev.sum() / tot, row_coordinates=sites[:, :dim],
                            lc_site_scores=lc[:, :dim], col_coordinates=species[:, :dim], env_coordinates=bp[:, :dim],
                            row_names=sn, col_names=spn, env_names=xn, const=const, all_eigenvalues=eig_all)
