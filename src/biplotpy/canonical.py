"""Canonical / MANOVA biplot (port of MultBiplotR::CanonicalBiplot and
Canonical.Variate.Analysis; Gabriel, 1972; Amaro, Vicente-Villardón and
Galindo-Villardón, 2004).

The group means are represented with the within-groups metric. With Z the
indicator matrix, Xb the group means, S11 = Z'Z, B = Xb' S11 Xb and
S = X'X - B (within sums of squares and products):
Y = S11^{1/2} Xb S^{-1/2} = U D V'; variable markers H = S^{1/2} V, canonical
weights B = S^{-1/2} V, group markers J = Xb B and individuals X B.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .transforms import initial_transform, transform_name
from .utils import matrix_sqrt


class CanonicalBiplot:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def _df(self, M, idx):
        return pd.DataFrame(M, index=idx, columns=[f"Dim {d + 1}" for d in range(M.shape[1])])

    @property
    def groups(self):
        return self._df(self.group_coordinates, self.group_names)

    @property
    def variables(self):
        return self._df(self.col_coordinates, self.col_names)

    def radii(self, kind="bonferroni"):
        """Radii of the confidence circles for the group means:
        ``'univariate'`` (t), ``'bonferroni'``, ``'multivariate'`` (Hotelling) or ``'chisq'``."""
        return pd.Series({"univariate": self.univ_rad, "bonferroni": self.bonf_rad,
                          "multivariate": self.mult_rad, "chisq": self.chis_rad}[kind], index=self.group_names)

    def predict(self, Xnew):
        """Classify new (untransformed) observations by linear discriminant analysis
        with priors equal to the group proportions."""
        Z = (np.asarray(Xnew, float) - self._t_center) / self._t_scale
        return self._classify(Z @ self.canonical_weights_full)

    def summary(self) -> dict:
        r = len(self.eigenvalues)
        return {"eigen": pd.DataFrame({"Eigenvalue": self.eigenvalues, "Explained": self.inertia,
                                       "Cumulative": self.cum_inertia}, index=[f"Dim {i + 1}" for i in range(r)]),
                "manova": self.manova, "anovas": self.anovas,
                "group_contributions": self._df(self.group_contributions, self.group_names),
                "variable_quality": self._df(self.qlr_vars, self.col_names),
                "structure": self._df(self.structure, self.col_names)}

    def plot(self, *args, **kwargs):
        from .plot import plot_canonical
        return plot_canonical(self, *args, **kwargs)

    def __repr__(self):
        s = self.summary()
        return (f"Canonical/MANOVA Biplot; transformation: {self.transformation}\n{s['eigen'].round(4).to_string()}\n\n"
                f"MANOVA:\n{self.manova.round(4).to_string()}\n\nGroup markers:\n{self.groups.round(4).to_string()}")


def _manova(lam, n, g, m):
    """Wilks, Pillai, Hotelling-Lawley and Roy from the eigenvalues of S^-1 B."""
    s = min(m, g - 1)
    dfh, dfe = g - 1, n - g
    wilks = np.prod(1 / (1 + lam))
    t = np.sqrt((dfh ** 2 * m ** 2 - 4) / (m ** 2 + dfh ** 2 - 5)) if (m ** 2 + dfh ** 2 - 5) > 0 else 1.0
    w = dfe + dfh - 0.5 * (m + dfh + 1)
    df1 = m * dfh
    df2 = w * t - 0.5 * (m * dfh - 2)
    Fw = (1 - wilks ** (1 / t)) / wilks ** (1 / t) * df2 / df1
    pillai = np.sum(lam / (1 + lam))
    mm, nn = 0.5 * (abs(m - dfh) - 1), 0.5 * (dfe - m - 1)
    Fp = (2 * nn + s + 1) / (2 * mm + s + 1) * pillai / (s - pillai)
    dfp = (s * (2 * mm + s + 1), s * (2 * nn + s + 1))
    hl = np.sum(lam)
    b = 2 * (s * nn + 1)
    Fh = 2 * (s * nn + 1) * hl / (s ** 2 * (2 * mm + s + 1))
    dfh_ = (s * (2 * mm + s + 1), b)
    roy = lam.max()
    q = max(m, dfh)
    Fr = roy * (dfe - q + dfh) / q
    dfr = (q, dfe - q + dfh)
    tab = pd.DataFrame({
        "Statistic": [wilks, pillai, hl, roy],
        "F": [Fw, Fp, Fh, Fr],
        "df1": [df1, dfp[0], dfh_[0], dfr[0]],
        "df2": [df2, dfp[1], dfh_[1], dfr[1]]}, index=["Wilks", "Pillai", "Hotelling-Lawley", "Roy"])
    tab["p"] = sps.f.sf(tab["F"], tab["df1"], tab["df2"])
    return tab


def canonical_biplot(X, group, sup=None, transform=5, row_names=None, col_names=None) -> CanonicalBiplot:
    """Canonical (MANOVA) biplot of the groups defined by ``group``.

    ``transform``: initial transformation (5 = standardize columns, as in
    MultBiplotR). ``sup``: supplementary individuals (raw scale).
    """
    if isinstance(X, pd.DataFrame):
        row_names = row_names or [str(i) for i in X.index]
        col_names = col_names or [str(c) for c in X.columns]
        X = X.to_numpy(float)
    X0 = np.array(X, float)
    n, m = X0.shape
    col_names = list(col_names) if col_names is not None else [f"V{j + 1}" for j in range(m)]
    row_names = list(row_names) if row_names is not None else [f"I{i + 1}" for i in range(n)]
    grp = np.asarray(group)
    if hasattr(group, "cat"):
        levels = [str(c) for c in group.cat.categories]
    elif hasattr(group, "categories"):
        levels = [str(c) for c in group.categories]
    else:
        levels = sorted({str(v) for v in grp})
    grp = np.array([str(v) for v in grp])
    tr = initial_transform(X0, transform_name(transform))
    Xs = tr["X"]
    name = transform_name(transform)
    t_center = tr["col_means"] if name in ("Standardize columns", "Column centering") else np.zeros(m)
    t_scale = tr["col_sds"] if name == "Standardize columns" else np.ones(m)
    g = len(levels)
    r = min(g - 1, m)
    Z = np.column_stack([(grp == lev).astype(float) for lev in levels])
    ng = Z.sum(0)
    S11 = np.diag(ng)
    Xb = (Z.T @ Xs) / ng[:, None]
    Bm = Xb.T @ S11 @ Xb
    S = Xs.T @ Xs - Bm
    Y = np.diag(np.sqrt(ng)) @ Xb @ matrix_sqrt(S, inverse=True)
    U, d, Vt = np.linalg.svd(Y, full_matrices=False)
    Vv = Vt.T
    H = matrix_sqrt(S) @ Vv[:, :r]
    Bw = matrix_sqrt(S, inverse=True) @ Vv[:, :r]
    J = Xb @ Bw
    Vi = Xs @ Bw
    VS = None if sup is None else ((np.asarray(sup, float) - t_center) / t_scale) @ Bw
    vprop = d[:r]
    iner = vprop ** 2 / np.sum(vprop ** 2) * 100
    lam = vprop ** 2
    manova = _manova(lam, n, g, m)
    # contributions and qualities
    gcontr = J ** 2 / (J ** 2).sum(1, keepdims=True)
    ccontr = H ** 2 / (H ** 2).sum(1, keepdims=True)
    SCT = (Xs ** 2).sum()
    SCRows, SCCols = (Xs ** 2).sum(1), (Xs ** 2).sum(0)
    expl = np.zeros(r); rowc = np.zeros((n, r)); qlr = np.zeros((m, r))
    for j in range(r):
        res = Xs - Vi[:, :j + 1] @ H[:, :j + 1].T
        expl[j] = 1 - (res ** 2).sum() / SCT
        rowc[:, j] = 1 - (res ** 2).sum(1) / SCRows
        qlr[:, j] = 1 - (res ** 2).sum(0) / SCCols
    sctotal = np.diag(Xs.T @ Xs)
    scdentro = np.diag(S)
    scentre = sctotal - scdentro
    glh, gle = g - 1, n - g
    fs = (scentre / glh) / (scdentro / gle)
    anovas = pd.DataFrame({"Total": sctotal, "Groups": scentre, "Error": scdentro, "F": fs,
                           "p-val": sps.f.sf(fs, glh, gle)}, index=col_names)
    base = 1 / np.sqrt(ng) / np.sqrt(n - g)
    univ = sps.t.ppf(1 - 0.025, n - g) * base
    bonf = sps.t.ppf(1 - 0.025 / (g * m), n - g) * base
    mult = np.sqrt(sps.f.ppf(0.95, m, n - g - m + 1) * ((n - g) * m) / (n - g - m + 1)) * base
    chis = np.sqrt(sps.chi2.ppf(0.95, 2)) * base
    structure = np.corrcoef(np.column_stack([Xs, Vi]), rowvar=False)[:m, m:]
    # LDA classification in the full canonical space (within covariance = S/(n-g))
    Bfull = matrix_sqrt(S, inverse=True) @ Vv[:, :r] * np.sqrt(n - g)
    Jf = Xb @ Bfull
    logp = np.log(ng / n)

    def classify(Zc):
        d2 = ((Zc[:, None, :] - Jf[None, :, :]) ** 2).sum(2)
        return np.array(levels)[np.argmax(-0.5 * d2 + logp, axis=1)]

    pred = classify(Xs @ Bfull)
    ct = pd.crosstab(pd.Series(grp, name="group"), pd.Series(pred, name="predicted"))
    return CanonicalBiplot(
        title="Canonical/MANOVA Biplot", transformation=name, n=n, p=m, g=g, dim=r, group=grp,
        group_names=levels, col_names=col_names, row_names=row_names, scaled_data=Xs,
        eigenvalues=vprop, inertia=iner, cum_inertia=np.cumsum(iner), wilks_f=manova.loc["Wilks", "F"],
        wilks_p=manova.loc["Wilks", "p"], manova=manova, anovas=anovas,
        group_coordinates=J, col_coordinates=H, canonical_weights=Bw, row_coordinates=Vi,
        sup_coordinates=VS, structure=structure, group_contributions=gcontr, col_contributions=ccontr,
        qlr_vars=qlr, row_contributions=rowc, expl_total=expl,
        univ_rad=univ, bonf_rad=bonf, mult_rad=mult, chis_rad=chis,
        predicted=pred, classification_table=ct,
        percent_correct=float(np.mean(pred == grp)),
        canonical_weights_full=Bfull, _classify=classify, _t_center=t_center, _t_scale=t_scale)


def canonical_variate_analysis(X, group, transform=5, **kw) -> CanonicalBiplot:
    """Canonical variate analysis (same solution, MANOVA and classification)."""
    res = canonical_biplot(X, group, transform=transform, **kw)
    res.title = "Canonical Analysis"
    return res
