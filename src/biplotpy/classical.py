"""Classical biplots: GH (CMP), JK (RMP), SQRT and HJ-Biplot.

Port of ``PCA.Biplot`` and ``HJ.Biplot`` (MultBiplotR, Vicente-Villardón).
With the singular value decomposition X = U D V' of the transformed data:

* alpha = 1  : JK / RMP-Biplot   rows A = U D,          columns B = V
* alpha = 0  : GH / CMP-Biplot   rows A = U,            columns B = V D
* alpha = 0.5: SQRT biplot       rows A = U D^{1/2},    columns B = V D^{1/2}
* HJ-Biplot (Galindo, 1986): A = U D and B = V D, both with optimal quality.

For alpha <= 1 the markers are rescaled by a common factor so that rows and
columns have comparable spread (``scale_factor``), as in MultBiplotR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transforms import initial_transform, transform_name


class ContinuousBiplot:
    """Result of a classical biplot (fields mirror MultBiplotR's ``ContinuousBiplot``)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    # labelled views ---------------------------------------------------------
    def _df(self, M, idx):
        return pd.DataFrame(M, index=idx, columns=[f"Dim {d + 1}" for d in range(M.shape[1])])

    @property
    def rows(self):
        return self._df(self.row_coordinates, self.row_names)

    @property
    def columns(self):
        return self._df(self.col_coordinates, self.col_names)

    def qualities(self, which="columns"):
        """Quality of representation: cumulative relative contributions (%)."""
        if which == "columns":
            return self._df(np.cumsum(self.col_contributions, 1), self.col_names)
        return self._df(np.cumsum(self.row_contributions, 1), self.row_names)

    def summary(self) -> dict:
        d = self.dim
        eig = pd.DataFrame({"Eigenvalue": self.eigenvalues[:d], "Exp. Var": self.inertia[:d],
                            "Cumulative": self.cum_inertia[:d]}, index=[f"Dim {i + 1}" for i in range(d)])
        return {"eigen": eig,
                "col_contributions": self._df(self.col_contributions, self.col_names),
                "col_qualities": self.qualities("columns"),
                "row_contributions": self._df(self.row_contributions, self.row_names),
                "row_qualities": self.qualities("rows"),
                "structure": self._df(self.structure, self.col_names),
                "row_contrib_to_factor": self._df(self.row_contrib_to_factor, self.row_names),
                "col_contrib_to_factor": self._df(self.col_contrib_to_factor, self.col_names)}

    def plot(self, *args, **kwargs):
        from .plot import plot_biplot
        return plot_biplot(self, *args, **kwargs)

    def __repr__(self):
        s = self.summary()
        return (f"{self.title} ({self.normalization}); transformation: {self.transformation}\n"
                f"{s['eigen'].round(3).to_string()}\n\nQuality of representation of the columns (%):\n"
                f"{s['col_qualities'].round(2).to_string()}")


_NORMALIZATION = {2: "Principal normalization (HJ-Biplot)", 1: "Row principal normalization (JK / RMP-Biplot)",
                  0: "Column principal normalization (GH / CMP-Biplot)", 0.5: "Symmetrical normalization (SQRT-Biplot)"}


def pca_biplot(X, alpha=1.0, dim=2, scaling=5, sup_rows=None, sup_cols=None, grouping=None,
               row_names=None, col_names=None) -> ContinuousBiplot:
    """Classical PCA biplot with normalization ``alpha`` (``alpha > 1`` gives the HJ-Biplot).

    ``scaling`` is the initial transformation (name or MultBiplotR number; 5 =
    standardize columns, 4 = column centering).
    """
    if isinstance(X, pd.DataFrame):
        row_names = row_names or [str(i) for i in X.index]
        col_names = col_names or [str(c) for c in X.columns]
        X = X.to_numpy(float)
    X0 = np.array(X, float)
    n, p = X0.shape
    row_names = list(row_names) if row_names is not None else [f"I{i + 1}" for i in range(n)]
    col_names = list(col_names) if col_names is not None else [f"V{j + 1}" for j in range(p)]
    tr = initial_transform(X0, transform_name(scaling), sup_rows, sup_cols, grouping)
    Xs = tr["X"]
    U, d, Vt = np.linalg.svd(Xs, full_matrices=False)
    V = Vt.T
    ev = d ** 2
    inertia = ev / ev.sum() * 100
    a = U[:, :dim] * d[:dim]
    b = V[:, :dim] * d[:dim]
    structure = np.corrcoef(np.column_stack([Xs, a]), rowvar=False)[:p, p:]
    sf = (Xs ** 2).sum(1)
    rc = a ** 2 / sf[:, None] * 100
    scc = (Xs ** 2).sum(0)
    cc = b ** 2 / scc[:, None] * 100
    # contributions of each element to the factor (absolute contributions, %)
    rcf = a ** 2 / (a ** 2).sum(0) * 100
    ccf = b ** 2 / (b ** 2).sum(0) * 100
    sr, sc = tr["sup_rows"], tr["sup_cols"]
    a_s = b_s = None
    if sr is not None:
        a_s = sr @ V[:, :dim]
        rcs = a_s ** 2 / (sr ** 2).sum(1)[:, None] * 100
    if sc is not None:
        b_s = sc.T @ U[:, :dim]
        ccs = b_s ** 2 / (sc ** 2).sum(0)[:, None] * 100
    if alpha <= 1:
        a = a * (1 / d[:dim]) ** (1 - alpha)
        b = b * (1 / d[:dim]) ** alpha
        if a_s is not None:
            a_s = a_s * (1 / d[:dim]) ** (1 - alpha)
        if b_s is not None:
            b_s = b_s * (1 / d[:dim]) ** alpha
        sca = (a ** 2).sum() / n
        scb = (b ** 2).sum() / p
        scf = np.sqrt(np.sqrt(scb / sca))
        a, b = a * scf, b / scf
        a_s = None if a_s is None else a_s * scf
        b_s = None if b_s is None else b_s / scf
    else:
        scf = 1.0
    norm = _NORMALIZATION.get(alpha if alpha <= 1 else 2, f"Custom normalization (alpha = {alpha})")
    return ContinuousBiplot(
        title="HJ Biplot" if alpha > 1 else "PCA Biplot", type="HJ" if alpha > 1 else "PCA", alpha=alpha,
        normalization=norm, dim=dim, transformation=tr["transform"], non_scaled_data=X0, scaled_data=Xs,
        means=X0.mean(0), medians=np.median(X0, 0), deviations=tr["col_sds"], minima=X0.min(0), maxima=X0.max(0),
        p25=np.quantile(X0, 0.25, axis=0), p75=np.quantile(X0, 0.75, axis=0), gmean=X0.mean(),
        eigenvalues=ev, inertia=inertia, cum_inertia=np.cumsum(inertia), V=V[:, :dim], U=U[:, :dim], d=d,
        structure=structure, row_coordinates=a, col_coordinates=b, row_contributions=rc, col_contributions=cc,
        row_contrib_to_factor=rcf, col_contrib_to_factor=ccf, scale_factor=scf, row_names=row_names,
        col_names=col_names, n=n, p=p,
        sup_row_coordinates=a_s, sup_col_coordinates=b_s,
        sup_row_contributions=None if sr is None else rcs, sup_col_contributions=None if sc is None else ccs)


def hj_biplot(X, dim=3, scaling=5, **kw) -> ContinuousBiplot:
    """HJ-Biplot (Galindo, 1986): rows and columns with optimal quality of representation."""
    return pca_biplot(X, alpha=2, dim=dim, scaling=scaling, **kw)


def gh_biplot(X, dim=2, scaling=5, **kw) -> ContinuousBiplot:
    """GH-Biplot (column metric preserving, alpha = 0)."""
    return pca_biplot(X, alpha=0, dim=dim, scaling=scaling, **kw)


def jk_biplot(X, dim=2, scaling=5, **kw) -> ContinuousBiplot:
    """JK-Biplot (row metric preserving, alpha = 1)."""
    return pca_biplot(X, alpha=1, dim=dim, scaling=scaling, **kw)


def sqrt_biplot(X, dim=2, scaling=5, **kw) -> ContinuousBiplot:
    """Symmetrical (SQRT) biplot, alpha = 0.5."""
    return pca_biplot(X, alpha=0.5, dim=dim, scaling=scaling, **kw)
