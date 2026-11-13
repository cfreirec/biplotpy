"""Bootstrap of distance-based ordinations: principal coordinates
(``MultBiplotR::BootstrapDistance``) and SMACOF multidimensional scaling.

Both resample the residuals between the observed dissimilarities and the
distances fitted by the ordination (with replacement, ``method='sampling'``, or
by permutation), add them to the fitted distances, refit the ordination and
align each replicate to the original configuration by Procrustes. The cloud of
replicates of each individual gives its stability (confidence ellipses or
convex hulls in ``plot_bootstrap_ordination``)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .distances import mds
from .rrng import RRNG


def _dist_vec(M):
    """Lower triangle of a square matrix in R's ``dist`` order (by columns)."""
    n = M.shape[0]
    return np.concatenate([M[j + 1:, j] for j in range(n - 1)])


def _vec_to_mat(v, n):
    M = np.zeros((n, n))
    k = 0
    for j in range(n - 1):
        M[j + 1:, j] = v[k:k + n - 1 - j]
        k += n - 1 - j
    return M + M.T


def _euclid(X):
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))


def simple_procrustes(X, Y, mode="orthogonal"):
    """Align ``Y`` to ``X`` (rotation/reflection, scale and translation).

    ``mode='orthogonal'``: least-squares orthogonal Procrustes, rotation
    ``V U'`` from the SVD ``X'JY = U D V'``. ``mode='MultBiplotR'`` uses the
    rotation ``U V'`` of ``MultBiplotR::SimpleProcrustes``, which is the
    transpose of the optimal one (it fixes reflections but turns small
    rotations the wrong way); it is kept only to reproduce R's results."""
    n = X.shape[0]
    J = np.eye(n) - 1.0 / n
    C = X.T @ J @ Y
    U, d, Vt = np.linalg.svd(C)
    T = U @ Vt if mode == "MultBiplotR" else Vt.T @ U.T
    s = np.trace(C @ T) / np.trace(Y.T @ J @ Y)
    t = (X - s * Y @ T).sum(0) / n
    return s * Y @ T + t


def _resample(Er, rng, method):
    m = len(Er)
    if method == "sampling":
        return Er[rng.sample_replace(m, m)]
    return Er[rng.sample_noreplace(m, m)]


class OrdinationBootstrap:
    """Replicates of a bootstrapped ordination."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def ellipses(self, level=0.95):
        """Centre and covariance of the replicates of each individual (first two dims)."""
        return [(c[:, :2].mean(0), np.cov(c[:, :2], rowvar=False)) for c in self.coordinates]

    def summary(self) -> pd.DataFrame:
        sd = np.array([c.std(0, ddof=1) for c in self.coordinates])
        return pd.DataFrame(sd, index=self.row_names, columns=[f"sd Dim {k + 1}" for k in range(sd.shape[1])])

    def plot(self, *args, **kwargs):
        from .plot import plot_bootstrap_ordination
        return plot_bootstrap_ordination(self, *args, **kwargs)


def bootstrap_pcoa(D, nB=200, dim=2, weights=None, method="sampling", procrustes="orthogonal", seed=None,
                   row_names=None) -> OrdinationBootstrap:
    """Bootstrap of principal coordinates analysis (``MultBiplotR::BootstrapDistance``).

    Residuals ``D - D_hat`` (``D_hat``: distances between the principal
    coordinates in ``dim`` dimensions) are resampled, added to ``D_hat`` and the
    PCoA is recomputed. ``seed`` uses R's generator (``set.seed(seed)``), so with
    ``procrustes='MultBiplotR'`` the replicates are those of R. Returns, for each
    replicate, the eigenvalues, the inertia (%), the aligned coordinates and the
    qualities of representation (%) of each individual."""
    if isinstance(D, pd.DataFrame):
        row_names = row_names or [str(i) for i in D.index]
    D = np.asarray(D, float)
    n = D.shape[0]
    if D.shape[1] != n:
        raise ValueError("The distance matrix must be square")
    if np.trace(D) > 0:
        raise ValueError("That is not a distance matrix (no zeros in the diagonal)")
    method = method.lower()
    Wd = np.eye(n) if weights is None else np.diag(np.asarray(weights, float))
    sW = np.sqrt(Wd); sWi = np.linalg.inv(sW)
    J = np.eye(n) - 1.0 / n

    def pco(Dm):
        B = -0.5 * J @ (Dm ** 2) @ J
        U, d, Vt = np.linalg.svd(sW @ B @ sW)
        return sWi @ U @ np.diag(np.sqrt(d)), d
    PO, d0 = pco(D)
    PO = PO[:, :dim]
    Dhat = _euclid(PO)
    Er = _dist_vec(D - Dhat)
    rng = RRNG(0 if seed is None else seed) if seed is not None else None
    g = np.random.default_rng()
    lam = np.zeros((n, nB)); inert = np.zeros((n, nB))
    coords = np.zeros((n, nB, dim)); qual = np.zeros((n, nB, dim))
    for b in range(nB):
        if rng is not None:
            ErE = _resample(Er, rng, method)
        else:
            ErE = Er[g.integers(0, len(Er), len(Er))] if method == "sampling" else g.permutation(Er)
        DE = _vec_to_mat(ErE, n) + Dhat
        POE, d = pco(DE)
        lam[:, b] = d; inert[:, b] = d / d.sum() * 100
        tot = (POE ** 2).sum(1)
        POE = POE[:, :dim]
        if procrustes:
            POE = simple_procrustes(PO, POE, procrustes)
        coords[:, b] = POE
        qual[:, b] = POE ** 2 / tot[:, None] * 100
    return OrdinationBootstrap(title=f"Bootstrap of PCoA ({method} of residuals)", method=method, nB=nB,
                               reference=PO, eigenvalues0=d0, eigenvalues=lam, inertia=inert,
                               coordinates=[coords[i] for i in range(n)], qualities=[qual[i] for i in range(n)],
                               row_names=row_names or [f"I{i + 1}" for i in range(n)])


def bootstrap_mds(D, nB=200, dim=2, type="ratio", method="sampling", seed=None, itmax=1000, eps=1e-10,
                  row_names=None) -> OrdinationBootstrap:
    """Bootstrap of SMACOF multidimensional scaling (``mds``).

    The residuals between the disparities and the configuration distances of
    the MDS (both on the normalized scale of SMACOF) are resampled and added to
    the configuration distances; negative values are set to 0. Each replicate
    is refitted from the original configuration and aligned to it by
    orthogonal Procrustes (rotation and reflection only). ``seed`` uses R's
    generator. MultBiplotR's ``BootstrapSmacof`` cannot be used as a reference
    (it reads ``Sol$Dhat``, which its ``SMACOF`` does not return); the
    reference is the same scheme run in R on ``smacof::mds``."""
    if isinstance(D, pd.DataFrame):
        row_names = row_names or [str(i) for i in D.index]
    D = np.asarray(D, float)
    n = D.shape[0]
    m0 = mds(D, dim=dim, type=type, itmax=itmax, eps=eps)
    X0 = m0.row_coordinates
    Dh = m0.config_distances
    Er = _dist_vec(m0.disparities - Dh)
    rng = RRNG(seed) if seed is not None else None
    g = np.random.default_rng()
    coords = np.zeros((n, nB, dim)); stress = np.zeros(nB)
    for b in range(nB):
        if rng is not None:
            ErE = _resample(Er, rng, method.lower())
        else:
            ErE = Er[g.integers(0, len(Er), len(Er))] if method.lower() == "sampling" else g.permutation(Er)
        DE = _vec_to_mat(ErE, n) + Dh
        DE[DE < 0] = 0
        mi = mds(DE, dim=dim, type=type, init=X0, itmax=itmax, eps=eps)
        U, d, Vt = np.linalg.svd(mi.row_coordinates.T @ X0)
        coords[:, b] = mi.row_coordinates @ (U @ Vt)
        stress[b] = mi.stress
    return OrdinationBootstrap(title=f"Bootstrap of MDS ({type}, {method} of residuals)", method=method, nB=nB,
                               reference=X0, stress0=m0.stress, stress=stress,
                               coordinates=[coords[i] for i in range(n)],
                               row_names=row_names or [f"I{i + 1}" for i in range(n)])
