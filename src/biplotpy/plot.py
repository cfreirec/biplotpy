"""Graphics for every biplot in biplotpy (matplotlib).

Classical and canonical biplots (arrows or calibrated axes, group ellipses,
convex hulls, alpha-bags, densities, CVA classification regions,
correlation circle, contributions, scree plot, 3-D biplot), bootstrap
regions, logistic biplots (binary, ordinal, nominal prediction regions),
STATIS family, dynamic biplot, unfolding, DISTATIS, CDPCA, GGE biplot (the
ten views of ``GGEBiplots::GGEPlot``), AMMI1/AMMI2, correspondence-type
biplots (CA, MCA, CCA and RDA triplots), PCoA/MDS, external logistic
biplot, canonical analysis of distances, co-inertia and PLS biplots.
"""
from __future__ import annotations

import numpy as np

from .extras import biplot_scales, calibrated_axis
from .utils import r_pretty

GREY = "0.25"


def _ax(ax, size=(7, 6.5)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    return ax


def _colors(k):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(k)]


def _finish(ax, lim, xlabel="", ylabel="", title="", axes0=True):
    if axes0:
        ax.axhline(0, ls=":", color="0.65", lw=0.8); ax.axvline(0, ls=":", color="0.65", lw=0.8)
    if np.isscalar(lim):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    else:
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def _dimlab(obj, i):
    iner = getattr(obj, "inertia", None)
    if iner is not None and len(np.atleast_1d(iner)) > i and np.isfinite(np.atleast_1d(iner)[i]):
        return f"Dim {i + 1} ({np.atleast_1d(iner)[i]:.1f}%)"
    return f"Dim {i + 1}"


def _arrows(ax, B, names, color=GREY, scale=1.0, fontsize=8, lw=1.0):
    for k, name in enumerate(names):
        ax.annotate("", xy=scale * B[k], xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=9))
        ax.text(*(scale * B[k] * 1.07), str(name), fontsize=fontsize, color=color,
                ha="left" if B[k, 0] >= 0 else "right", va="bottom" if B[k, 1] >= 0 else "top")


# --------------------------------------------------------------------------- group overlays
def concentration_ellipse(P, level=0.95, n=100):
    """Normal-theory concentration ellipse of a 2-D point cloud."""
    P = np.asarray(P, float)
    m = P.mean(0)
    S = np.cov(P, rowvar=False)
    from scipy import stats as sps
    r = np.sqrt(sps.chi2.ppf(level, 2))
    w, U = np.linalg.eigh(S)
    th = np.linspace(0, 2 * np.pi, n)
    return m + (np.column_stack([np.cos(th), np.sin(th)]) * r * np.sqrt(np.maximum(w, 0))) @ U.T


def convex_hull(P):
    from scipy.spatial import ConvexHull
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    h = ConvexHull(P)
    return P[np.r_[h.vertices, h.vertices[0]]]


def alpha_bag(P, alpha=0.95):
    """Approximate alpha-bag (biplotEZ): convex hull of the ``alpha`` share of
    the deepest points, depth measured by convex-hull peeling."""
    from scipy.spatial import ConvexHull
    P = np.asarray(P, float)
    idx = np.arange(len(P))
    target = int(np.ceil(alpha * len(P)))
    while len(idx) > max(target, 3):
        try:
            h = ConvexHull(P[idx])
        except Exception:
            break
        if len(idx) - len(h.vertices) < target:
            # remove only the most outlying hull vertices (farthest from the median)
            med = np.median(P[idx], 0)
            far = h.vertices[np.argsort(-((P[idx][h.vertices] - med) ** 2).sum(1))]
            idx = np.delete(idx, far[:len(idx) - target])
            break
        idx = np.delete(idx, h.vertices)
    return convex_hull(P[idx])


def _group_overlay(ax, A, groups, kind, level, colors, levels):
    from scipy.stats import gaussian_kde
    for k, g in enumerate(levels):
        P = A[groups == g]
        if len(P) < 3:
            continue
        if kind == "ellipse":
            E = concentration_ellipse(P, level); ax.plot(E[:, 0], E[:, 1], color=colors[k], lw=1.2)
        elif kind == "hull":
            H = convex_hull(P); ax.fill(H[:, 0], H[:, 1], color=colors[k], alpha=0.12, lw=1, ec=colors[k])
        elif kind == "bag":
            H = alpha_bag(P, level); ax.fill(H[:, 0], H[:, 1], color=colors[k], alpha=0.15, lw=1, ec=colors[k])
        elif kind == "density":
            kde = gaussian_kde(P.T)
            xs = np.linspace(P[:, 0].min() - P[:, 0].std(), P[:, 0].max() + P[:, 0].std(), 80)
            ys = np.linspace(P[:, 1].min() - P[:, 1].std(), P[:, 1].max() + P[:, 1].std(), 80)
            XX, YY = np.meshgrid(xs, ys)
            Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax.contour(XX, YY, Z, levels=4, colors=[colors[k]], linewidths=0.8, alpha=0.8)


# --------------------------------------------------------------------------- classical biplots
def plot_biplot(bip, dims=(0, 1), ax=None, groups=None, show_rows=True, row_labels=False, arrows=True,
                calibrated=False, nticks=4, type_scale="Complete", values_scale="Original", col_color=GREY,
                group_shape=None, level=0.95, color_by=None, cmap="viridis", min_quality=0.0, sup=True,
                show_cols=True, density=False):
    """Classical (GH, JK, SQRT, HJ, FA, NIPALS, PLS) biplot.

    ``groups``: colour rows by group; ``group_shape``: 'ellipse'
    (concentration), 'hull', 'bag' (alpha-bag) or 'density' per group;
    ``calibrated=True`` draws predictive calibrated axes (biplotEZ /
    MultBiplotR); ``color_by='quality'`` colours the rows by their quality of
    representation on the plane (factoextra cos2); ``min_quality`` hides
    variables below that quality (%); ``density`` draws a 2-D density of all
    rows (MultBiplotR DensityBiplot)."""
    import matplotlib.pyplot as plt
    ax = _ax(ax)
    i, j = dims
    A, B = bip.row_coordinates[:, [i, j]], bip.col_coordinates[:, [i, j]]
    lim = max(np.abs(A).max() if show_rows else 0, np.abs(B).max() if show_cols else 0) * 1.12
    if show_rows:
        if density:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(A.T)
            xs = np.linspace(-lim, lim, 100); XX, YY = np.meshgrid(xs, xs)
            Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax.contourf(XX, YY, Z, levels=8, cmap="Greys", alpha=0.5)
        if groups is None:
            if color_by == "quality" and getattr(bip, "row_contributions", None) is not None:
                q = bip.row_contributions[:, [i, j]].sum(1)
                sc = ax.scatter(A[:, 0], A[:, 1], s=16, c=q, cmap=cmap, vmin=0, vmax=100)
                plt.colorbar(sc, ax=ax, shrink=0.7, label="Quality (%)")
            else:
                ax.scatter(A[:, 0], A[:, 1], s=14, color="tab:blue", alpha=0.7)
        else:
            groups = np.asarray(groups)
            levels = list(dict.fromkeys(groups))
            cols = _colors(len(levels))
            for k, g in enumerate(levels):
                m = groups == g
                ax.scatter(A[m, 0], A[m, 1], s=14, color=cols[k], alpha=0.7, label=str(g))
            if group_shape:
                _group_overlay(ax, A, groups, group_shape, level, cols, levels)
            ax.legend(fontsize=8, frameon=False)
        if row_labels:
            for r, name in enumerate(bip.row_names):
                ax.annotate(name, A[r], fontsize=6, alpha=0.7, xytext=(2, 2), textcoords="offset points")
        if sup and getattr(bip, "sup_row_coordinates", None) is not None:
            S = bip.sup_row_coordinates[:, [i, j]]
            ax.scatter(S[:, 0], S[:, 1], marker="^", color="tab:red", s=22, label="supplementary")
    if show_cols:
        keep = np.ones(len(B), bool)
        if min_quality > 0 and getattr(bip, "col_contributions", None) is not None:
            keep = bip.col_contributions[:, [i, j]].sum(1) >= min_quality
        names = np.asarray(bip.col_names)
        if calibrated:
            sc = biplot_scales(bip, nticks, type_scale, values_scale)
            for k in np.where(keep)[0]:
                b = B[k]
                u = b / np.linalg.norm(b)
                ax.plot([-lim * u[0], lim * u[0]], [-lim * u[1], lim * u[1]], color=col_color, lw=0.6)
                pos = calibrated_axis(b, sc["ticks"][k])
                perp = np.array([-u[1], u[0]]) * lim * 0.012
                for (x, y), lab in zip(pos, sc["labels"][k]):
                    if abs(x) <= lim and abs(y) <= lim:
                        ax.plot([x - perp[0], x + perp[0]], [y - perp[1], y + perp[1]], color=col_color, lw=0.6)
                        ax.text(x + 2.5 * perp[0], y + 2.5 * perp[1], f"{lab:g}", fontsize=6, color=col_color)
                ax.text(*(u * lim * 1.02), names[k], fontsize=8, color=col_color)
        elif arrows:
            _arrows(ax, B[keep], names[keep], col_color)
        if getattr(bip, "y_coordinates", None) is not None:
            _arrows(ax, bip.y_coordinates[:, [i, j]], bip.y_names, "tab:red")
        if sup and getattr(bip, "sup_col_coordinates", None) is not None:
            _arrows(ax, bip.sup_col_coordinates[:, [i, j]], [f"S{k + 1}" for k in range(len(bip.sup_col_coordinates))],
                    "tab:red")
    return _finish(ax, lim, _dimlab(bip, i), _dimlab(bip, j), bip.title)


def plot_correlation_circle(bip, dims=(0, 1), ax=None, color_by_quality=True):
    """Correlation circle (factoextra ``fviz_pca_var``): correlations of the
    variables with the row coordinates (structure)."""
    import matplotlib.pyplot as plt
    ax = _ax(ax, (6, 6))
    i, j = dims
    S = np.asarray(bip.structure)[:, [i, j]]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="0.5", lw=0.8)
    q = (S ** 2).sum(1)
    cm = plt.get_cmap("Blues")
    for k, name in enumerate(bip.col_names):
        c = cm(0.35 + 0.65 * min(q[k], 1)) if color_by_quality else GREY
        ax.annotate("", xy=S[k], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, mutation_scale=9))
        ax.text(*(S[k] * 1.08), name, fontsize=8, color=c, ha="left" if S[k, 0] >= 0 else "right")
    return _finish(ax, 1.15, _dimlab(bip, i), _dimlab(bip, j), "Correlation circle")


def plot_scree(obj, ax=None, n=None, cumulative=True):
    ax = _ax(ax, (6, 4))
    iner = np.asarray(obj.inertia)
    n = n or min(len(iner), 10)
    x = np.arange(1, n + 1)
    ax.bar(x, iner[:n], color="0.6")
    for xi, v in zip(x, iner[:n]):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    if cumulative:
        ax.plot(x, np.cumsum(iner)[:n], "o-", color="tab:blue", ms=3, lw=1)
    ax.set_xticks(x); ax.set_xlabel("Dimension"); ax.set_ylabel("Explained inertia (%)")
    ax.set_title("Scree plot", fontsize=10)
    return ax


def plot_contributions(obj, which="cols", dim=0, ax=None, top=None):
    """Contribution bars (factoextra ``fviz_contrib``) with the expected
    average contribution as a dashed line."""
    ax = _ax(ax, (7, 4))
    if which == "cols":
        b = np.asarray(obj.col_coordinates)[:, dim]; names = list(obj.col_names)
    else:
        b = np.asarray(obj.row_coordinates)[:, dim]; names = list(obj.row_names)
    c = b ** 2 / np.sum(b ** 2) * 100
    o = np.argsort(-c)[:top or len(c)]
    ax.bar(range(len(o)), c[o], color="tab:blue")
    ax.axhline(100 / len(c), ls="--", color="tab:red", lw=1)
    ax.set_xticks(range(len(o))); ax.set_xticklabels([names[k] for k in o], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Contribution (%)"); ax.set_title(f"Contributions to Dim {dim + 1}", fontsize=10)
    return ax


def plot_biplot_3d(bip, dims=(0, 1, 2), groups=None, ax=None, arrows=True):
    """3-D biplot (bpca)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    A, B = bip.row_coordinates[:, list(dims)], bip.col_coordinates[:, list(dims)]
    if groups is None:
        ax.scatter(*A.T, s=10, color="tab:blue", alpha=0.7)
    else:
        groups = np.asarray(groups); levels = list(dict.fromkeys(groups)); cols = _colors(len(levels))
        for k, g in enumerate(levels):
            ax.scatter(*A[groups == g].T, s=10, color=cols[k], label=str(g))
        ax.legend(fontsize=8)
    if arrows:
        for k, name in enumerate(bip.col_names):
            ax.plot([0, B[k, 0]], [0, B[k, 1]], [0, B[k, 2]], color=GREY)
            ax.text(*(B[k] * 1.08), name, fontsize=8, color=GREY)
    ax.set_xlabel(_dimlab(bip, dims[0])); ax.set_ylabel(_dimlab(bip, dims[1])); ax.set_zlabel(_dimlab(bip, dims[2]))
    ax.set_title(bip.title, fontsize=10)
    return ax


def plot_bootstrap(bip, boot, dims=(0, 1), ax=None, kind="ellipse", level=0.95):
    """Bootstrap confidence regions for the variable markers of a PCA biplot
    (``pca_bootstrap``): ellipse or convex hull of the replicates."""
    ax = plot_biplot(bip, dims, ax=ax, show_rows=True, arrows=True)
    L = np.asarray(boot["loadings"])                  # B x p x dim, unit eigenvectors (sign-aligned)
    ev = np.asarray(boot["eigenvalues"])              # B x k singular values
    i, j = dims
    alpha = bip.alpha if bip.alpha <= 1 else 1.0
    fac = ev[:, [i, j]] ** (1 - alpha) / (bip.scale_factor if bip.alpha <= 1 else 1.0)
    for k in range(L.shape[1]):
        P = L[:, k, :][:, [i, j]] * fac
        if kind == "ellipse":
            E = concentration_ellipse(P, level); ax.plot(E[:, 0], E[:, 1], color="tab:orange", lw=0.9)
        else:
            H = convex_hull(P); ax.plot(H[:, 0], H[:, 1], color="tab:orange", lw=0.9)
    ax.set_title("Bootstrap biplot", fontsize=10)
    return ax


# --------------------------------------------------------------------------- canonical biplot
def plot_canonical(cb, dims=(0, 1), ax=None, radius="bonferroni", show_individuals=True, arrows=True,
                   regions=False, calibrated=False, group_shape=None, nticks=4, level=0.95):
    """Canonical/MANOVA (CVA) biplot: group markers with confidence circles,
    individuals and variables. ``regions=True`` shades the classification
    regions (nearest group mean in the canonical space, biplotEZ);
    ``calibrated=True`` draws calibrated axes in the original units;
    ``group_shape`` adds ellipses / hulls / alpha-bags of the individuals."""
    ax = _ax(ax)
    i, j = dims
    J, H, V = cb.group_coordinates[:, [i, j]], cb.col_coordinates[:, [i, j]], cb.row_coordinates[:, [i, j]]
    cols = _colors(len(cb.group_names))
    rad = cb.radii(radius).values
    lim = max(np.abs(V).max() if show_individuals else 0, (np.abs(J) + rad[:, None]).max()) * 1.1
    if regions:
        xs = np.linspace(-lim, lim, 300); XX, YY = np.meshgrid(xs, xs)
        G = np.column_stack([XX.ravel(), YY.ravel()])
        k = ((G[:, None, :] - J[None]) ** 2).sum(2).argmin(1).reshape(XX.shape)
        from matplotlib.colors import ListedColormap
        ax.contourf(XX, YY, k, levels=np.arange(-0.5, len(J)), cmap=ListedColormap(cols[:len(J)]), alpha=0.12)
    th = np.linspace(0, 2 * np.pi, 100)
    for k, g in enumerate(cb.group_names):
        c = cols[k]
        m = np.asarray(cb.group) == g
        if show_individuals:
            ax.scatter(V[m, 0], V[m, 1], s=10, color=c, alpha=0.4)
        if group_shape:
            _group_overlay(ax, V, np.asarray(cb.group), group_shape, level, cols, list(cb.group_names))
        ax.plot(J[k, 0] + rad[k] * np.cos(th), J[k, 1] + rad[k] * np.sin(th), color=c, lw=1.4)
        ax.scatter(*J[k], color=c, s=50, zorder=3)
        ax.annotate(g, J[k], textcoords="offset points", xytext=(0, 7), ha="center", color=c, fontweight="bold")
    if calibrated:
        center, scale = np.broadcast_to(cb._t_center, (cb.p,)), np.broadcast_to(cb._t_scale, (cb.p,))
        X0 = cb.scaled_data * scale + center
        for k, name in enumerate(cb.col_names):
            b = H[k]; nb = np.linalg.norm(b)
            if nb == 0:
                continue
            u = b / nb
            ax.plot([-lim * u[0], lim * u[0]], [-lim * u[1], lim * u[1]], color=GREY, lw=0.6)
            vals = r_pretty([X0[:, k].min(), X0[:, k].max()], nticks)
            pos = calibrated_axis(b, (vals - center[k]) / scale[k])
            perp = np.array([-u[1], u[0]]) * lim * 0.012
            for (x, y), lab in zip(pos, vals):
                if abs(x) <= lim and abs(y) <= lim:
                    ax.plot([x - perp[0], x + perp[0]], [y - perp[1], y + perp[1]], color=GREY, lw=0.6)
                    ax.text(x + 2.5 * perp[0], y + 2.5 * perp[1], f"{lab:g}", fontsize=6, color=GREY)
            ax.text(*(u * lim * 1.02), name, fontsize=8, color=GREY)
    elif arrows:
        esc = 0.8 * lim / np.sqrt((H ** 2).sum(1)).max()
        _arrows(ax, esc * H, cb.col_names)
    return _finish(ax, lim, _dimlab(cb, i), _dimlab(cb, j), cb.title)


# --------------------------------------------------------------------------- logistic biplots
def plot_logistic(lb, dims=(0, 1), ax=None, groups=None, probs=(0.5, 0.75), row_labels=False, variables=None):
    """Binary logistic biplot: each variable is a line in the direction of its
    slopes, marked at the points predicting the probabilities in ``probs``
    (0.5 and 0.75 by default, as MultBiplotR); ordinal logistic biplot: the
    line is cut at the category boundaries (thresholds)."""
    ax = _ax(ax)
    i, j = dims
    A = lb.A[:, [i, j]]
    lim = np.abs(A).max() * 1.25
    if groups is None:
        ax.scatter(A[:, 0], A[:, 1], s=12, color="tab:blue", alpha=0.6)
    else:
        groups = np.asarray(groups); levels = list(dict.fromkeys(groups)); cols = _colors(len(levels))
        for k, g in enumerate(levels):
            ax.scatter(*A[groups == g].T, s=12, color=cols[k], alpha=0.7, label=str(g))
        ax.legend(fontsize=8, frameon=False)
    if row_labels:
        for r, nm in enumerate(lb.row_names):
            ax.annotate(nm, A[r], fontsize=6, alpha=0.7)
    names = list(lb.col_names)
    sel = range(len(names)) if variables is None else [names.index(v) for v in variables]
    ordinal = ("Ordinal" in lb.title) or ("Polychoric" in lb.title)
    for k in sel:
        if ordinal:
            b = lb.B[k][[i, j]]; nb2 = b @ b
            if nb2 == 0:
                continue
            u = b / np.sqrt(nb2)
            ax.plot([-lim * u[0], lim * u[0]], [-lim * u[1], lim * u[1]], color="0.6", lw=0.6)
            for c, t in enumerate(np.atleast_1d(lb.thresholds[k])):
                x = t * b / nb2
                if np.all(np.abs(x) < lim):
                    perp = np.array([-u[1], u[0]]) * lim * 0.015
                    ax.plot([x[0] - perp[0], x[0] + perp[0]], [x[1] - perp[1], x[1] + perp[1]], color="tab:red", lw=1.2)
                    ax.text(x[0] + 2 * perp[0], x[1] + 2 * perp[1], f"{c + 1}|{c + 2}", fontsize=6, color="tab:red")
            ax.text(*(u * lim * 0.98), names[k], fontsize=8, color=GREY)
        else:
            b0, b = lb.B[k, 0], lb.B[k, 1:][[i, j]]; nb2 = b @ b
            if nb2 == 0:
                continue
            pts = [(np.log(p / (1 - p)) - b0) * b / nb2 for p in probs]
            ax.annotate("", xy=pts[-1], xytext=pts[0],
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1, mutation_scale=9))
            ax.scatter(*pts[0], s=12, color=GREY)
            ax.text(*(pts[-1] + 0.05 * b / np.sqrt(nb2)), names[k], fontsize=8, color=GREY)
    return _finish(ax, lim, f"Dim {i + 1}", f"Dim {j + 1}", lb.title)


def plot_nominal_regions(lb, variable, dims=(0, 1), ax=None, grid=250):
    """Prediction regions of the categories of one variable in a nominal
    logistic biplot (the plane is divided by the most probable category)."""
    from matplotlib.colors import ListedColormap
    ax = _ax(ax, (6, 5.5))
    k = list(lb.col_names).index(variable) if isinstance(variable, str) else int(variable)
    i, j = dims
    A = lb.A[:, [i, j]]
    lim = np.abs(A).max() * 1.15
    xs = np.linspace(-lim, lim, grid); XX, YY = np.meshgrid(xs, xs)
    G = np.zeros((grid * grid, lb.A.shape[1])); G[:, i] = XX.ravel(); G[:, j] = YY.ravel()
    Bk = lb.B[k]
    eta = np.column_stack([np.zeros(len(G)), Bk[:, 0] + G @ Bk[:, 1:].T])
    reg = eta.argmax(1).reshape(XX.shape)
    cats = lb.categories[k]
    cols = _colors(len(cats))
    ax.contourf(XX, YY, reg, levels=np.arange(-0.5, len(cats)), cmap=ListedColormap(cols), alpha=0.18)
    x = np.asarray([str(v) for v in lb.data[:, k]]) if hasattr(lb, "data") else None
    for c, cat in enumerate(cats):
        m = reg == c
        if m.any():
            ax.text(XX[m].mean(), YY[m].mean(), str(cat), color=cols[c], fontsize=11, fontweight="bold",
                    ha="center")
    ax.scatter(A[:, 0], A[:, 1], s=8, color="0.3", alpha=0.6)
    return _finish(ax, lim, f"Dim {i + 1}", f"Dim {j + 1}", f"{lb.col_names[k]}: prediction regions")


# --------------------------------------------------------------------------- STATIS family
def plot_statis(res, kind="compromise", dims=(0, 1), ax=None, trajectories=False, cols=True):
    """STATIS-ACT graphics: 'interstructure' (tables in the RV space),
    'compromise' (biplot of the compromise; ``trajectories`` joins each row
    across tables) or 'weights'."""
    ax = _ax(ax)
    i, j = dims
    names = list(res.names)
    if kind == "interstructure":
        I = res.interstructure[:, [i, j]]
        I = I * np.sign(I.sum(0) + 1e-300)
        for k, nm in enumerate(names):
            ax.annotate("", xy=I[k], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:blue"))
            ax.text(*(I[k] * 1.05), nm, fontsize=8)
        lim = np.abs(I).max() * 1.25
        lim = (min(-0.05, I[:, 0].min() * 1.2), lim, min(-lim * 0.6, I[:, 1].min() * 1.2), max(lim * 0.6, I[:, 1].max() * 1.2))
        return _finish(ax, lim, f"Dim {i + 1} ({res.inertia_inter[i]:.1f}%)", f"Dim {j + 1} ({res.inertia_inter[j]:.1f}%)",
                       "Interstructure (RV)")
    if kind == "weights":
        ax.bar(names, res.weights, color="tab:blue"); ax.set_ylabel("Weight in the compromise")
        ax.set_title("STATIS weights", fontsize=10); return ax
    R = res.row_coordinates[:, [i, j]]
    rn = getattr(res, "levels", None) or [f"I{k + 1}" for k in range(len(R))]
    ax.scatter(R[:, 0], R[:, 1], s=26, color="tab:blue", zorder=3)
    for k, nm in enumerate(rn):
        ax.annotate(str(nm), R[k], fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = np.abs(R).max()
    if trajectories and getattr(res, "trajectories", None) is not None:
        T = np.stack([t[:, [i, j]] for t in res.trajectories], 1)
        tc = _colors(T.shape[1])
        for r in range(T.shape[0]):
            ax.plot(T[r, :, 0], T[r, :, 1], color="0.6", lw=0.7)
            for t in range(T.shape[1]):
                ax.scatter(*T[r, t], s=10, color=tc[t], label=names[t] if r == 0 else None)
        ax.legend(fontsize=7, frameon=False)
        lim = max(lim, np.abs(T).max())
    if cols and getattr(res, "col_coordinates", None) is not None and np.ndim(res.col_coordinates) == 2:
        C = res.col_coordinates[:, [i, j]]
        labels = getattr(res, "col_labels", None) or getattr(res, "var_names", None) or [f"V{k + 1}" for k in range(len(C))]
        sc = 0.9 * lim / np.abs(C).max() if np.abs(C).max() > lim else 1.0
        _arrows(ax, C * sc, labels, fontsize=6)
    return _finish(ax, lim * 1.2, f"Dim {i + 1} ({res.inertia[i]:.1f}%)", f"Dim {j + 1} ({res.inertia[j]:.1f}%)",
                   res.title)


def plot_dynamic(res, dims=(0, 1), ax=None, rows=None, show_columns=True):
    """Dynamic biplot (Egido & Galindo): trajectory of each row across the
    occasions on the reference biplot, with arrows between consecutive
    occasions."""
    ax = _ax(ax)
    i, j = dims
    T = res["trajectories"][:, :, [i, j]]
    rn = list(res["reference"].row_names)
    sel = range(T.shape[0]) if rows is None else [rn.index(r) for r in rows]
    tc = _colors(T.shape[1])
    for r in sel:
        for t in range(T.shape[1] - 1):
            ax.annotate("", xy=T[r, t + 1], xytext=T[r, t],
                        arrowprops=dict(arrowstyle="->", color="0.55", lw=0.8, mutation_scale=7))
        for t in range(T.shape[1]):
            ax.scatter(*T[r, t], s=10, color=tc[t], label=res["occasions"][t] if r == list(sel)[0] else None, zorder=3)
        ax.annotate(rn[r], T[r, -1], fontsize=6, xytext=(2, 2), textcoords="offset points")
    lim = np.abs(T[list(sel)]).max() * 1.15
    if show_columns:
        B = res["reference"].col_coordinates[:, [i, j]]
        _arrows(ax, B * (0.9 * lim / np.abs(B).max()), res["reference"].col_names)
    ax.legend(fontsize=7, frameon=False)
    return _finish(ax, lim, _dimlab(res["reference"], i), _dimlab(res["reference"], j), "Dynamic biplot")


def plot_unfolding(res, ax=None, row_names=None, col_names=None):
    """Joint configuration of an unfolding (individuals as points, objects as
    labelled triangles)."""
    ax = _ax(ax)
    X, Y = res["conf_row"], res["conf_col"]
    ax.scatter(X[:, 0], X[:, 1], s=12, color="tab:blue", alpha=0.7)
    ax.scatter(Y[:, 0], Y[:, 1], s=40, marker="^", color="tab:red")
    for k, nm in enumerate(col_names or [f"O{k + 1}" for k in range(len(Y))]):
        ax.annotate(nm, Y[k], fontsize=8, color="tab:red", xytext=(3, 3), textcoords="offset points")
    if row_names:
        for k, nm in enumerate(row_names):
            ax.annotate(nm, X[k], fontsize=6, alpha=0.7)
    lim = max(np.abs(X).max(), np.abs(Y).max()) * 1.15
    return _finish(ax, lim, "Dim 1", "Dim 2", f"Unfolding (stress-1 = {res['stress']:.3f})")


def plot_distatis(res, kind="compromise", dims=(0, 1), ax=None, partial=True, row_names=None, table_names=None):
    """DISTATIS: 'compromise' (factor scores with the partial scores of each
    table) or 'rv' (tables on the first eigenvectors of the RV matrix)."""
    ax = _ax(ax)
    i, j = dims
    if kind == "rv":
        G = res["G"][:, [i, j]]
        ax.scatter(G[:, 0], G[:, 1], color="tab:blue")
        for k in range(len(G)):
            ax.annotate((table_names or [f"T{t + 1}" for t in range(len(G))])[k], G[k], fontsize=8)
        lim = (G[:, 0].min() - 0.2, G[:, 0].max() + 0.2, G[:, 1].min() - 0.3, G[:, 1].max() + 0.3)
        return _finish(ax, lim, "Dim 1", "Dim 2", "RV map of the tables")
    F = res["F"][:, [i, j]]
    ax.scatter(F[:, 0], F[:, 1], s=40, color="black", zorder=3)
    rn = row_names or [f"O{k + 1}" for k in range(len(F))]
    for k, nm in enumerate(rn):
        ax.annotate(nm, F[k], fontsize=8, xytext=(3, 3), textcoords="offset points")
    lim = np.abs(F).max()
    if partial:
        P = res["partial_F"][:, [i, j], :]
        cols = _colors(P.shape[2])
        for t in range(P.shape[2]):
            for k in range(len(F)):
                ax.plot([F[k, 0], P[k, 0, t]], [F[k, 1], P[k, 1, t]], color=cols[t], lw=0.7)
            ax.scatter(P[:, 0, t], P[:, 1, t], s=10, color=cols[t],
                       label=(table_names or [f"T{q + 1}" for q in range(P.shape[2])])[t])
        ax.legend(fontsize=7, frameon=False)
        lim = max(lim, np.abs(P).max())
    ev = res["eig_Splus"]
    return _finish(ax, lim * 1.15, f"Factor {i + 1} ({ev[i] / ev[ev > 0].sum() * 100:.1f}%)",
                   f"Factor {j + 1} ({ev[j] / ev[ev > 0].sum() * 100:.1f}%)", "DISTATIS compromise")


def plot_cdpca(res, ax=None, truth=None):
    """CDPCA: objects on the disjoint components, coloured by cluster, with
    the cluster centroids (``biplotbootGUI::CDpca`` graphic)."""
    ax = _ax(ax)
    Y, Yb = res["Y"][:, :2], res["Ybar"][:, :2]
    cl = res["cluster"]
    cols = _colors(cl.max())
    for c in range(1, cl.max() + 1):
        ax.scatter(*Y[cl == c].T, s=14, color=cols[c - 1], label=f"Cluster {c}", alpha=0.75)
    ax.scatter(Yb[:, 0], Yb[:, 1], s=90, marker="s", color="black")
    lim = np.abs(Y).max() * 1.15
    ax.legend(fontsize=8, frameon=False)
    return _finish(ax, lim, f"Component 1 ({res['var_percent'][0]:.1f}%)", f"Component 2 ({res['var_percent'][1]:.1f}%)",
                   f"CDPCA (between-cluster deviance {res['bcdev']:.1f}%)")


# --------------------------------------------------------------------------- GGE and AMMI
_GGE_TYPES = {"biplot": 1, "examine_env": 2, "examine_gen": 3, "env_relationship": 4, "compare_gens": 5,
              "which_won_where": 6, "discrimination": 7, "rank_env": 8, "mean_stability": 9, "rank_gen": 10}


def _perp_foot(direction, P):
    """Feet of the perpendiculars from the points P to the line through the
    origin with the given direction."""
    u = np.asarray(direction, float); u = u / np.linalg.norm(u)
    return np.outer(P @ u, u)


def plot_gge(model, type="biplot", dims=(0, 1), ax=None, env=None, gen=None, gen2=None):
    """GGE biplot views of ``GGEBiplots::GGEPlot``: 'biplot', 'examine_env',
    'examine_gen', 'env_relationship', 'compare_gens', 'which_won_where',
    'discrimination' (discriminativeness vs representativeness),
    'rank_env', 'mean_stability' and 'rank_gen'."""
    t = _GGE_TYPES.get(type, type)
    ax = _ax(ax)
    i, j = dims
    G = model["coordgenotype"][:, [i, j]]; E = model["coordenviroment"][:, [i, j]]
    lg, le = model["labelgen"], model["labelenv"]
    cg, ce = "forestgreen", "blue"
    allp = np.vstack([G, E])
    lim = np.abs(allp).max() * 1.2
    ve = model["varexpl"][[i, j]]

    def txt(P, labs, col, size=8):
        for k, nm in enumerate(labs):
            ax.text(P[k, 0], P[k, 1], nm, color=col, fontsize=size, ha="center", va="center")
    if t in (1, 6, 7, 8, 10, 4):
        if t != 4:
            txt(G, lg, cg)
    if t == 1:
        for k in range(len(E)):
            ax.plot([0, E[k, 0]], [0, E[k, 1]], color=ce, alpha=0.5, lw=0.8)
        txt(E, le, ce)
        title = "GGE biplot"
    elif t == 2:
        k = le.index(env); e = E[k]
        ax.plot([-lim * e[0] / np.linalg.norm(e), lim * e[0] / np.linalg.norm(e)],
                [-lim * e[1] / np.linalg.norm(e), lim * e[1] / np.linalg.norm(e)], color=ce, lw=0.8)
        ax.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ce))
        F = _perp_foot(e, G)
        for r in range(len(G)):
            ax.plot([G[r, 0], F[r, 0]], [G[r, 1], F[r, 1]], ls="--", color=cg, lw=0.6)
        txt(G, lg, cg); txt(E[[k]], [env], ce, 10)
        title = f"Examine environment {env}"
    elif t == 3:
        k = lg.index(gen); g = G[k]
        ax.plot([-lim * g[0] / np.linalg.norm(g), lim * g[0] / np.linalg.norm(g)],
                [-lim * g[1] / np.linalg.norm(g), lim * g[1] / np.linalg.norm(g)], color=cg, lw=0.8)
        F = _perp_foot(g, E)
        for r in range(len(E)):
            ax.plot([E[r, 0], F[r, 0]], [E[r, 1], F[r, 1]], ls="--", color=ce, lw=0.6)
        txt(E, le, ce); txt(G[[k]], [gen], cg, 10)
        title = f"Examine genotype {gen}"
    elif t == 4:
        for k in range(len(E)):
            ax.annotate("", xy=E[k], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ce, lw=0.9))
        txt(E * 1.08, le, ce)
        title = "Relationship among environments"
    elif t == 5:
        k1, k2 = lg.index(gen), lg.index(gen2)
        a, b = G[k1], G[k2]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:red")
        d = b - a
        perp = np.array([-d[1], d[0]])
        ax.plot([-lim * perp[0] / np.linalg.norm(perp), lim * perp[0] / np.linalg.norm(perp)],
                [-lim * perp[1] / np.linalg.norm(perp), lim * perp[1] / np.linalg.norm(perp)], color="tab:red", lw=0.8)
        txt(E, le, ce); txt(G, lg, cg); txt(G[[k1, k2]], [gen, gen2], "black", 10)
        title = f"Compare {gen} with {gen2}"
    elif t == 6:
        from scipy.spatial import ConvexHull
        h = ConvexHull(G).vertices
        poly = G[np.r_[h, h[0]]]
        ax.plot(poly[:, 0], poly[:, 1], color="black", lw=1)
        for a, b in zip(poly[:-1], poly[1:]):
            d = b - a
            n_ = np.array([-d[1], d[0]])
            n_ = n_ / np.linalg.norm(n_)
            foot = _perp_foot(n_, a[None])[0]
            if foot @ n_ < 0:
                n_ = -n_
            ax.plot([0, n_[0] * lim * 1.5], [0, n_[1] * lim * 1.5], color="tab:red", lw=0.9)
        txt(E, le, ce)
        for k in h:
            ax.text(G[k, 0], G[k, 1], lg[k], color="black", fontsize=9, fontweight="bold", ha="center", va="center")
        title = "Which won where"
    elif t in (7, 8, 9, 10):
        m = E.mean(0)
        u = m / np.linalg.norm(m)
        ax.plot([-lim * u[0], lim * u[0]], [-lim * u[1], lim * u[1]], color=ce, lw=0.9)
        ax.annotate("", xy=u * np.abs(E).max(), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ce, lw=1.2))
        th = np.linspace(0, 2 * np.pi, 200)
        if t == 7:
            for k in range(len(E)):
                ax.plot([0, E[k, 0]], [0, E[k, 1]], color=ce, alpha=0.5, lw=0.8)
            step = max(np.ptp(E[:, 0]), np.ptp(E[:, 1])) / 10
            for r in range(1, 6):
                ax.plot(r * step * np.cos(th), r * step * np.sin(th), color="tab:red", lw=0.6)
            txt(E, le, ce)
            title = "Discriminativeness vs representativeness"
        elif t == 8:
            mod = np.sqrt((E ** 2).sum(1)).max()
            ideal = np.sign(m[0]) * np.sqrt(mod ** 2 / (1 + m[1] ** 2 / m[0] ** 2)) * np.array([1, m[1] / m[0]])
            step = np.linalg.norm(ideal - m) / 3
            for r in range(1, 9):
                ax.plot(ideal[0] + r * step * np.cos(th), ideal[1] + r * step * np.sin(th), color="tab:red", lw=0.6)
            ax.scatter(*ideal, marker="*", s=80, color="tab:red")
            for k in range(len(E)):
                ax.plot([0, E[k, 0]], [0, E[k, 1]], color=ce, alpha=0.5, lw=0.8)
            txt(E, le, ce)
            title = "Ranking environments"
        elif t == 9:
            F = _perp_foot(m, G)
            perp_line = np.array([-u[1], u[0]])
            ax.plot([-lim * perp_line[0], lim * perp_line[0]], [-lim * perp_line[1], lim * perp_line[1]], color=ce,
                    lw=0.6, ls=":")
            for r in range(len(G)):
                ax.plot([G[r, 0], F[r, 0]], [G[r, 1], F[r, 1]], ls="--", color=cg, lw=0.6)
            txt(G, lg, cg); txt(E, le, ce)
            title = "Mean vs stability"
        else:
            F = _perp_foot(m, G)
            proj = F @ u
            ideal = F[np.argmax(proj)] if proj.max() > 0 else u * 0
            step = np.linalg.norm(ideal - m) / 3
            for r in range(1, 11):
                ax.plot(ideal[0] + r * step * np.cos(th), ideal[1] + r * step * np.sin(th), color="0.6", lw=0.6)
            ax.scatter(*ideal, marker="*", s=80, color="tab:red")
            txt(E, le, ce)
            title = "Ranking genotypes"
    return _finish(ax, lim, f"Component {i + 1} {ve[0]:.2f}%", f"Component {j + 1} {ve[1]:.2f}%", title)


def plot_ammi(res, kind="AMMI2", ax=None):
    """AMMI1 (mean vs IPCA1) or AMMI2 (IPCA1 vs IPCA2) biplot."""
    ax = _ax(ax)
    G, E = res["genotype_scores"].to_numpy(), res["environment_scores"].to_numpy()
    pc = res["analysis"]["percent"].to_numpy()
    if kind == "AMMI1":
        xg, xe = res["genotype_means"].to_numpy(), res["environment_means"].to_numpy()
        ax.scatter(xg, G[:, 0], color="forestgreen", s=14)
        ax.scatter(xe, E[:, 0], color="blue", marker="^", s=24)
        for k, nm in enumerate(res["genotype_scores"].index):
            ax.annotate(nm, (xg[k], G[k, 0]), fontsize=7, color="forestgreen")
        for k, nm in enumerate(res["environment_scores"].index):
            ax.annotate(nm, (xe[k], E[k, 0]), fontsize=8, color="blue")
        ax.axhline(0, ls=":", color="0.6"); ax.axvline(np.mean(xg), ls=":", color="0.6")
        ax.set_xlabel("Mean"); ax.set_ylabel(f"IPCA1 ({pc[0]:.1f}%)"); ax.set_title("AMMI1 biplot", fontsize=10)
        return ax
    ax.scatter(G[:, 0], G[:, 1], color="forestgreen", s=14)
    for k, nm in enumerate(res["genotype_scores"].index):
        ax.annotate(nm, G[k, :2], fontsize=7, color="forestgreen")
    for k, nm in enumerate(res["environment_scores"].index):
        ax.annotate("", xy=E[k, :2], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="blue"))
        ax.text(*(E[k, :2] * 1.06), nm, fontsize=8, color="blue")
    lim = max(np.abs(G[:, :2]).max(), np.abs(E[:, :2]).max()) * 1.15
    return _finish(ax, lim, f"IPCA1 ({pc[0]:.1f}%)", f"IPCA2 ({pc[1]:.1f}%)", "AMMI2 biplot")


# --------------------------------------------------------------------------- ordination-type biplots
def plot_ordination(res, dims=(0, 1), ax=None, groups=None, row_labels=False, col_labels=True, show_rows=True,
                    show_cols=True, env_scale=None, sup_vars=None, lc=False):
    """Biplot of CA, MCA, CCA, RDA (triplot), PCoA, MDS, external logistic
    biplot and canonical analysis of distances."""
    ax = _ax(ax)
    i, j = dims
    t = res.type
    R = getattr(res, "lc_site_scores" if lc else "row_coordinates", None)
    C = getattr(res, "col_coordinates", None)
    pts = []
    if t == "CDA":
        cols = _colors(len(res.group_names))
        Gm = res.group_coordinates[:, [i, j]]
        if res.row_coordinates is not None:
            Rr = res.row_coordinates[:, [i, j]]
            for k, g in enumerate(res.group_names):
                ax.scatter(*Rr[res.groups == g].T, s=10, color=cols[k], alpha=0.4)
            pts.append(Rr)
        for k, g in enumerate(res.group_names):
            ax.scatter(*Gm[k], s=60, color=cols[k], zorder=3)
            ax.annotate(g, Gm[k], fontsize=9, color=cols[k], fontweight="bold", xytext=(0, 7), textcoords="offset points",
                        ha="center")
        pts.append(Gm)
        lim = np.abs(np.vstack(pts)).max() * 1.15
        return _finish(ax, lim, _dimlab(res, i), _dimlab(res, j),
                       f"Canonical analysis of distances (F = {float(np.squeeze(res.F)):.2f}, p = {res.p_value:.3f})")
    if show_rows and R is not None:
        R = R[:, [i, j]]
        if groups is not None:
            groups = np.asarray(groups); levels = list(dict.fromkeys(groups)); gc = _colors(len(levels))
            for k, g in enumerate(levels):
                ax.scatter(*R[groups == g].T, s=12, color=gc[k], alpha=0.7, label=str(g))
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.scatter(R[:, 0], R[:, 1], s=12 if t not in ("CA",) else 26, color="tab:blue", alpha=0.75)
        if row_labels or t == "CA":
            for k, nm in enumerate(res.row_names):
                ax.annotate(nm, R[k], fontsize=7, color="tab:blue", xytext=(3, 3), textcoords="offset points")
        pts.append(R)
    if show_cols and C is not None:
        Cc = C[:, [i, j]]
        if t == "ExternalLogistic":
            for k, nm in enumerate(res.col_names):
                b0, b = res.col_parameters[k, 0], res.col_parameters[k, 1:][[i, j]]
                nb2 = b @ b
                p5, p75 = (0 - b0) * b / nb2, (np.log(3) - b0) * b / nb2
                ax.annotate("", xy=p75, xytext=p5, arrowprops=dict(arrowstyle="-|>", color=GREY, mutation_scale=9))
                ax.scatter(*p5, s=10, color=GREY)
                ax.text(*p75, nm, fontsize=8, color=GREY)
                pts.append(np.vstack([p5, p75]))
        elif t in ("RDA",):
            _arrows(ax, Cc, res.col_names, "tab:red", fontsize=7, lw=0.8)
            pts.append(Cc)
        else:
            ax.scatter(Cc[:, 0], Cc[:, 1], s=28, marker="^", color="tab:red")
            if col_labels:
                for k, nm in enumerate(res.col_names):
                    ax.annotate(nm, Cc[k], fontsize=7, color="tab:red", xytext=(3, -8), textcoords="offset points")
            pts.append(Cc)
    lim = np.abs(np.vstack(pts)).max() * 1.15 if pts else 1
    if getattr(res, "env_coordinates", None) is not None:
        Ev = res.env_coordinates[:, [i, j]]
        sc = env_scale or 0.85 * lim / np.abs(Ev).max()
        _arrows(ax, Ev * sc, res.env_names, "black", lw=1.2)
    if sup_vars is not None:
        Sv = sup_vars["coordinates"][:, [i, j]]
        _arrows(ax, Sv * (0.85 * lim / np.abs(Sv).max()), sup_vars["names"], "black")
    if t in ("PCoA", "MDS"):
        xl, yl = _dimlab(res, i), _dimlab(res, j)
    else:
        xl, yl = _dimlab(res, i), _dimlab(res, j)
    return _finish(ax, lim, xl, yl, res.title)


def plot_coinertia(res, ax=None, row_names=None):
    """Co-inertia plot: row scores in X (start) and Y (arrow head) on the
    co-inertia axes (normalized), as ``ade4::plot.coinertia``."""
    ax = _ax(ax)
    X = res["lX"] / np.sqrt((res["lX"] ** 2).mean(0))
    Y = res["lY"] / np.sqrt((res["lY"] ** 2).mean(0))
    for k in range(len(X)):
        ax.annotate("", xy=Y[k, :2], xytext=X[k, :2], arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.7,
                                                                       mutation_scale=7))
    ax.scatter(X[:, 0], X[:, 1], s=10, color="tab:blue", label="row in X (arrow points to Y)")
    if row_names:
        for k, nm in enumerate(row_names):
            ax.annotate(nm, X[k, :2], fontsize=6)
    lim = max(np.abs(X[:, :2]).max(), np.abs(Y[:, :2]).max()) * 1.15
    ax.legend(fontsize=8, frameon=False)
    return _finish(ax, lim, f"Axis 1 ({res['inertia'][0]:.1f}%)", f"Axis 2 ({res['inertia'][1]:.1f}%)",
                   f"Co-inertia (RV = {res['RV']:.3f})")


# --------------------------------------------------------------------------- phase 5
def plot_bootstrap_ordination(bt, dims=(0, 1), ax=None, kind="ellipse", level=0.95, show_replicates=True,
                              labels=True, title=None):
    """Bootstrap of a PCoA or an MDS: the replicates of each individual (small
    points), the original configuration (large points) and, per individual, a
    concentration ellipse (``kind='ellipse'``) or the convex hull of its
    replicates (``kind='hull'``)."""
    ax = _ax(ax)
    i, j = dims
    cols = _colors(len(bt.coordinates))
    allp = []
    for k, C in enumerate(bt.coordinates):
        P = C[:, [i, j]]; allp.append(P)
        if show_replicates:
            ax.scatter(*P.T, s=4, color=cols[k], alpha=0.35)
        if kind == "hull":
            H = convex_hull(P)
            if H is not None:
                ax.plot(*np.vstack([H, H[:1]]).T, color=cols[k], lw=0.8)
        else:
            ax.plot(*concentration_ellipse(P, level).T, color=cols[k], lw=0.8)
    ref = bt.reference[:, [i, j]]
    ax.scatter(*ref.T, s=22, color="black", zorder=3)
    if labels:
        for k, nm in enumerate(bt.row_names):
            ax.annotate(nm, ref[k], fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = np.abs(np.vstack(allp + [ref])).max() * 1.1
    return _finish(ax, lim, f"Dim {i + 1}", f"Dim {j + 1}", title or bt.title)


def plot_plsbin(res, dims=(0, 1), ax=None, row_labels=False, title=None):
    """Biplot of ``plsr1_bin`` (scores and weights of the predictors, rows
    coloured by the binary response) or of ``binary_plsr`` (X scores, X and Y
    loadings)."""
    ax = _ax(ax)
    i, j = dims
    if "scores" in res:
        T = res["scores"].to_numpy()[:, [i, j]]
        y = np.asarray(res["fit"]["fitted"] > 0.5)
        ax.scatter(*T.T, s=14, c=np.where(y, "tab:red", "tab:blue"), alpha=0.7)
        W = res["weights"].to_numpy()[:, [i, j]]
        sc = np.abs(T).max() / np.abs(W).max() * 0.8
        _arrows(ax, W, res["weights"].index, scale=sc)
        lim = max(np.abs(T).max(), np.abs(W * sc).max()) * 1.2
        return _finish(ax, lim, f"Comp. {i + 1}", f"Comp. {j + 1}", title or "PLSR1 Binary (red: predicted 1)")
    T = res["x_scores"].to_numpy()[:, [i, j]]
    ax.scatter(*T.T, s=14, color="tab:blue", alpha=0.6)
    if row_labels:
        for k, nm in enumerate(res["x_scores"].index):
            ax.annotate(nm, T[k], fontsize=6)
    P = res["x_loadings"].to_numpy()[:, [i, j]]; Q = res["y_loadings"].to_numpy()[:, [i, j]]
    sc = np.abs(T).max() / max(np.abs(P).max(), np.abs(Q).max()) * 0.8
    _arrows(ax, P, res["x_loadings"].index, color=GREY, scale=sc)
    _arrows(ax, Q, res["y_loadings"].index, color="tab:red", scale=sc)
    lim = max(np.abs(T).max(), np.abs(np.vstack([P, Q]) * sc).max()) * 1.2
    return _finish(ax, lim, f"Comp. {i + 1}", f"Comp. {j + 1}", title or "Binary PLS regression (red: responses)")
