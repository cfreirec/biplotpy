"""Multivariate tests that accompany canonical biplots and are not available
in Python: Mardia's multivariate normality test (``psych::mardia``),
heteroscedastic MANOVA with the Wald-type and modified ANOVA-type statistics
and parametric / wild bootstrap (``MANOVA.RM``; Konietschke et al., 2015;
Friedrich & Pauly, 2018), and the hypothesis and error matrices of HE plots
(``heplots``; Friendly, 2007)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


# --------------------------------------------------------------------------- Mardia
def mardia(X, unbiased: bool = True) -> dict:
    """Mardia's multivariate skewness and kurtosis.

    ``unbiased=True`` uses the covariance with n - 1 (as ``psych::mardia``);
    ``False`` uses the ML covariance of Mardia (1970). Returns b1p, b2p, the
    skewness statistic (chi-square with p(p+1)(p+2)/6 df), its small-sample
    version, the standardized kurtosis (normal) and their p-values."""
    X = np.asarray(X, float)
    X = X[np.isfinite(X).all(1)]
    n, p = X.shape
    Xc = X - X.mean(0)
    S = np.cov(Xc, rowvar=False, ddof=1 if unbiased else 0)
    D = Xc @ np.linalg.solve(S, Xc.T)
    b1p = np.sum(D ** 3) / n ** 2
    b2p = np.trace(D ** 2) / n
    df = p * (p + 1) * (p + 2) / 6
    k = (p + 1) * (n + 1) * (n + 3) / (n * ((n + 1) * (p + 1) - 6))
    skew, small = n * b1p / 6, n * k * b1p / 6
    kurt = (b2p - p * (p + 2)) * np.sqrt(n / (8 * p * (p + 2)))
    return {"n": n, "p": p, "b1p": b1p, "b2p": b2p, "skew": skew, "small_skew": small, "df": df,
            "p_skew": sps.chi2.sf(skew, df), "p_small": sps.chi2.sf(small, df), "kurtosis": kurt,
            "p_kurt": 2 * sps.norm.sf(abs(kurt)), "d": np.sqrt(np.diag(D))}


# --------------------------------------------------------------------------- MANOVA.RM
def _wts_mats(means, V, n, H):
    N = n.sum()
    Sn = N * _bdiag(V)
    T = H.T @ np.linalg.pinv(H @ Sn @ H.T) @ H
    wts = N * means @ T @ means
    D = np.diag(np.diag(Sn))
    mats = N * means @ H.T @ np.linalg.pinv(H @ D @ H.T) @ H @ means
    return wts, mats


def _bdiag(blocks):
    p = sum(b.shape[0] for b in blocks)
    M = np.zeros((p, p)); o = 0
    for b in blocks:
        k = b.shape[0]; M[o:o + k, o:o + k] = b; o += k
    return M


def manova_wts(Y, groups, H=None, iter: int = 10000, resampling: str = "paramBS", alpha: float = 0.05,
               seed=None) -> dict:
    """One-way MANOVA without normality or equal covariances (``MANOVA.RM::MANOVA``).

    ``H`` is the hypothesis matrix acting on the stacked vector of group means
    (default: equality of all group mean vectors, (I_a - J_a/a) kron I_p).
    Returns the Wald-type statistic (WTS) with its asymptotic chi-square
    p-value, the modified ANOVA-type statistic (MATS) and the bootstrap p-values
    of both (``resampling`` = 'paramBS' or 'WildBS')."""
    Y = np.asarray(Y, float)
    g = np.asarray([str(v) for v in groups])
    levels = list(groups.cat.categories) if hasattr(groups, "cat") else sorted(set(g))
    levels = [str(v) for v in levels]
    p = Y.shape[1]; a = len(levels)
    Ys = [Y[g == lev] for lev in levels]
    n = np.array([len(y) for y in Ys])
    if H is None:
        H = np.kron(np.eye(a) - np.ones((a, a)) / a, np.eye(p))
    H = np.asarray(H, float)
    means = np.concatenate([y.mean(0) for y in Ys])
    V = [np.cov(y, rowvar=False) / len(y) for y in Ys]
    wts, mats = _wts_mats(means, V, n, H)
    df = np.linalg.matrix_rank(H)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    bw, bm = np.empty(iter), np.empty(iter)
    for b in range(iter):
        if resampling == "paramBS":
            Xb = [rng.multivariate_normal(np.zeros(p), len(y) * Vi, size=len(y), method="cholesky")
                  for y, Vi in zip(Ys, V)]
        elif resampling == "WildBS":
            Xb = [(2 * rng.integers(0, 2, len(y)) - 1)[:, None] * (y - y.mean(0)) for y in Ys]
        else:
            raise ValueError("resampling must be 'paramBS' or 'WildBS'")
        mb = np.concatenate([x.mean(0) for x in Xb])
        bw[b], bm[b] = _wts_mats(mb, [np.cov(x, rowvar=False) / len(x) for x in Xb], n, H)
    return {"levels": levels, "n": n, "means": means.reshape(a, p), "WTS": wts, "df": df,
            "p_WTS_chisq": sps.chi2.sf(wts, df), "MATS": mats,
            "p_WTS_boot": float(np.mean(bw > wts)), "p_MATS_boot": float(np.mean(bm > mats)),
            "quantiles": {"WTS": float(np.quantile(bw, 1 - alpha)), "MATS": float(np.quantile(bm, 1 - alpha))},
            "resampling": resampling, "iter": iter}


# --------------------------------------------------------------------------- HE plots
def roy_crit(alpha, p, dfh, dfe):
    """Critical value of Roy's largest root (F upper bound), as ``heplots``."""
    df1 = max(p, dfh); df2 = dfe - df1 + dfh
    return df1 / df2 * sps.f.isf(alpha, df1, df2)


def hlt_crit(alpha, p, dfh, dfe):
    s = min(p, dfh); m = (abs(p - dfh) - 1) / 2; nn = (dfe - p - 1) / 2
    df1 = 2 * m + s + 1; df2 = 2 * (s * nn + 1)
    return s * df1 / df2 * sps.f.isf(alpha, df1, df2)


def he_matrices(Y, groups) -> dict:
    """Hypothesis (H) and error (E) SSP matrices of a one-way MANOVA, their
    degrees of freedom and the four multivariate statistics."""
    Y = np.asarray(Y, float)
    g = np.asarray([str(v) for v in groups])
    levels = list(groups.cat.categories) if hasattr(groups, "cat") else sorted(set(g))
    levels = [str(v) for v in levels]
    gm = Y.mean(0)
    H = np.zeros((Y.shape[1],) * 2); E = np.zeros_like(H)
    means = []
    for lev in levels:
        y = Y[g == lev]; m = y.mean(0); means.append(m)
        H += len(y) * np.outer(m - gm, m - gm)
        E += (y - m).T @ (y - m)
    dfh, dfe = len(levels) - 1, len(Y) - len(levels)
    ev = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(E, H))))[::-1]
    stats = {"Pillai": np.sum(ev / (1 + ev)), "Wilks": np.prod(1 / (1 + ev)),
             "Hotelling-Lawley": np.sum(ev), "Roy": ev[0]}
    return {"H": H, "E": E, "dfh": dfh, "dfe": dfe, "eigenvalues": ev, "stats": stats,
            "means": pd.DataFrame(means, index=levels), "grand_mean": gm}


def he_ellipses(Y, groups, variables=(0, 1), size="evidence", alpha=0.05, level=0.68, segments=60) -> dict:
    """Coordinates of the H and E ellipses of an HE plot (``heplots::heplot``).

    E is scaled by 1/dfe; H by 1/dfe and, with ``size='evidence'``, also by the
    critical value of Roy's test, so that H projects outside E in some
    direction if and only if Roy's test is significant at ``alpha``."""
    he = he_matrices(Y, groups)
    v = list(variables)
    dfe = he["dfe"]; p = np.asarray(Y).shape[1]
    radius = np.sqrt(2 * sps.f.ppf(level, 2, dfe))
    factor = roy_crit(alpha, p, he["dfh"], dfe) if size in ("evidence", "significance") else 1.0
    th = np.linspace(0, 2 * np.pi, segments + 1)
    circle = radius * np.column_stack([np.cos(th), np.sin(th)])
    c = he["grand_mean"][v]

    def ell(S):
        ev, U = np.linalg.eigh(S)
        return c + circle @ (U @ np.diag(np.sqrt(np.maximum(ev, 0)))).T
    H = he["H"][np.ix_(v, v)] / dfe / factor
    E = he["E"][np.ix_(v, v)] / dfe
    return {"H": ell(H), "E": ell(E), "H_shape": H, "E_shape": E, "means": he["means"].iloc[:, v],
            "radius": radius, "factor": factor, "roy_significant": he["eigenvalues"][0] > roy_crit(alpha, p, he["dfh"], dfe)}


def he_plot(Y, groups, variables=(0, 1), size="evidence", alpha=0.05, ax=None, labels=None):
    """Draw an HE plot with matplotlib."""
    import matplotlib.pyplot as plt
    r = he_ellipses(Y, groups, variables, size, alpha)
    ax = ax or plt.gca()
    ax.plot(*r["E"].T, color="tab:red", label="Error")
    ax.plot(*r["H"].T, color="tab:blue", label="Groups")
    ax.scatter(*r["means"].to_numpy().T, color="tab:blue", s=12)
    for name, (x, y) in r["means"].iterrows():
        ax.annotate(name, (x, y), fontsize=8)
    if labels:
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    ax.legend(frameon=False)
    return ax


# --------------------------------------------------------------------------- multi-factor MANOVA and HE plots
def _sum_contrast_cols(f, levels):
    """Columns of ``contr.sum`` coding of a factor (k - 1 columns)."""
    k = len(levels)
    C = np.zeros((len(f), k - 1))
    for a, lev in enumerate(levels[:-1]):
        C[:, a] = (f == lev).astype(float)
    C[f == levels[-1]] = -1.0
    return C


def _design(factors: pd.DataFrame, terms):
    """Model matrix (intercept + sum-coded terms) and the columns of each term."""
    n = len(factors)
    cols = {}
    for name in factors.columns:
        f = factors[name].astype(str).to_numpy()
        levels = list(factors[name].cat.categories) if hasattr(factors[name], "cat") else sorted(set(f))
        cols[name] = _sum_contrast_cols(f, [str(v) for v in levels])
    blocks, Xs = {}, [np.ones((n, 1))]
    start = 1
    for t in terms:
        parts = t.split(":")
        M = np.ones((n, 1))
        for p_ in parts:
            M = (M[:, :, None] * cols[p_][:, None, :]).reshape(n, -1)
        Xs.append(M); blocks[t] = list(range(start, start + M.shape[1])); start += M.shape[1]
    return np.column_stack(Xs), blocks


def _sse(Y, X):
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    R = Y - X @ b
    return R.T @ R


def manova_terms(Y, factors, terms=None, type="II") -> dict:
    """Multivariate linear model with several factors (and their interactions):
    hypothesis SSP matrix of each term, error SSP matrix and the four
    multivariate statistics, as ``car::Anova(lm(cbind(...) ~ ...), type =)``.

    ``factors``: DataFrame with one column per factor. ``terms``: list such as
    ``["A", "B", "A:B"]`` (default: all main effects and interactions).
    ``type='II'``: each term adjusted for the terms that do not contain it;
    ``type='III'``: each term adjusted for all the others (sum contrasts)."""
    Y = np.asarray(Y, float)
    factors = pd.DataFrame(factors).copy()
    names = list(factors.columns)
    if terms is None:
        from itertools import combinations
        terms = [":".join(c) for k in range(1, len(names) + 1) for c in combinations(names, k)]
    X, blocks = _design(factors, terms)
    n = len(Y)
    E = _sse(Y, X)
    dfe = n - np.linalg.matrix_rank(X)
    out = {}
    for t in terms:
        tset = set(t.split(":"))
        if type == "III":
            keep = [c for c in range(X.shape[1]) if c not in blocks[t]]
            H = _sse(Y, X[:, keep]) - E
        else:
            contain = [u for u in terms if u != t and tset <= set(u.split(":"))]
            base = [0] + [c for u in terms if u != t and u not in contain for c in blocks[u]]
            H = _sse(Y, X[:, base]) - _sse(Y, X[:, sorted(base + blocks[t])])
        H = (H + H.T) / 2
        ev = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(E, H))))[::-1]
        ev = np.where(np.abs(ev) < 1e-12, 0.0, ev)
        out[t] = {"H": H, "df": len(blocks[t]), "eigenvalues": ev,
                  "stats": {"Pillai": np.sum(ev / (1 + ev)), "Wilks": np.prod(1 / (1 + ev)),
                            "Hotelling-Lawley": np.sum(ev), "Roy": ev[0]}}
    return {"terms": out, "E": E, "dfe": dfe, "type": type}


def he_plot_terms(Y, factors, terms=None, type="II", variables=(0, 1), alpha=0.05, ax=None, labels=None,
                  level=0.68, segments=60):
    """HE plot of a multi-factor MANOVA: the error ellipse and one hypothesis
    ellipse per term, each scaled by the critical value of Roy's test for its
    degrees of freedom (``size = 'evidence'`` of ``heplots::heplot``): a term
    is significant by Roy's test at ``alpha`` exactly when its ellipse goes
    outside the error ellipse in some direction."""
    import matplotlib.pyplot as plt
    m = manova_terms(Y, factors, terms, type)
    Y = np.asarray(Y, float)
    v = list(variables)
    dfe = m["dfe"]; p = Y.shape[1]
    radius = np.sqrt(2 * sps.f.ppf(level, 2, dfe))
    th = np.linspace(0, 2 * np.pi, segments + 1)
    circle = radius * np.column_stack([np.cos(th), np.sin(th)])
    c = Y.mean(0)[v]

    def ell(S):
        ev, U = np.linalg.eigh(S)
        return c + circle @ (U @ np.diag(np.sqrt(np.maximum(ev, 0)))).T
    ax = ax or plt.gca()
    ax.plot(*ell(m["E"][np.ix_(v, v)] / dfe).T, color="tab:red", label="Error")
    cols = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:brown", "tab:olive", "tab:cyan"]
    for k, (t, r) in enumerate(m["terms"].items()):
        crit = roy_crit(alpha, p, r["df"], dfe)
        sig = r["eigenvalues"][0] > crit
        ax.plot(*ell(r["H"][np.ix_(v, v)] / dfe / crit).T, color=cols[k % len(cols)],
                lw=2 if sig else 1, ls="-" if sig else "--", label=t + (" *" if sig else ""))
    if labels:
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"HE plot (type {type})")
    return m
