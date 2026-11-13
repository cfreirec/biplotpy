"""DISTATIS (Abdi et al., 2005, 2012): STATIS for a set of distance matrices
on the same objects (port of ``DistatisR::distatis``)."""
from __future__ import annotations

import numpy as np


def _double_center(D):
    n = D.shape[0]
    C = np.eye(n) - 1 / n
    return -0.5 * C @ D @ C


def distatis(cube, norm="MFA", distance=True, double_centering=True, rv=True, nfact=3, design=None) -> dict:
    """``cube``: array n x n x T (or list of T n x n matrices) of distances.

    Returns the RV (or scalar product) matrix C with its eigen-analysis and
    table factor scores, the weights alpha, the compromise S+ and its factor
    scores, contributions and squared cosines, and the partial factor scores
    of every table."""
    D = np.stack(cube, axis=2) if isinstance(cube, (list, tuple)) else np.asarray(cube, float)
    if distance and not double_centering:
        raise ValueError("Distance matrices must be double centered")
    T = D.shape[2]
    CP = np.stack([_double_center(D[:, :, t] if distance else -D[:, :, t]) for t in range(T)], 2) \
        if double_centering else D.copy()
    for t in range(T):
        Y = CP[:, :, t]
        if norm == "MFA":
            CP[:, :, t] = Y / np.linalg.eigvalsh(Y)[-1]
        elif norm == "SUMPCA":
            CP[:, :, t] = Y / np.trace(Y)
        elif str(norm).upper() == "NUCLEAR":
            ev = np.linalg.eigvalsh(Y)
            CP[:, :, t] = Y / np.sum(np.sqrt(ev[ev > 2 * np.finfo(float).eps]))
    n = CP.shape[0]
    CP2 = CP.reshape(n * n, T, order="F")
    C = CP2.T @ CP2
    if rv:
        nr = np.sqrt((CP2 ** 2).sum(0))
        C = C / np.outer(nr, nr)
    ev, U = np.linalg.eigh(C)
    ev, U = ev[::-1], U[:, ::-1]
    U[:, 0] = np.abs(U[:, 0])
    G = U * np.sqrt(np.abs(ev))
    if design is None:
        alpha = U[:, 0] / U[:, 0].sum()
    else:
        design = np.asarray(design); levs = np.unique(design); alpha = U[:, 0].copy()
        for l in levs:
            m = design == l
            alpha[m] = alpha[m] / (len(levs) * alpha[m].sum())
    S = np.tensordot(CP, alpha, axes=([2], [0]))
    evS, US = np.linalg.eigh(S)
    evS, US = evS[::-1], US[:, ::-1]
    sv = np.sqrt(np.abs(evS))
    F = US * sv
    F2 = F ** 2
    d2 = F2.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ctr = F2 / np.abs(evS)
        cos2 = F2 / d2[:, None]
        Proj = US / sv
    Proj = Proj[:, :nfact]
    partial = np.stack([CP[:, :, t] @ Proj for t in range(T)], 2)
    return {"C": C, "eig_C": ev, "tau_C": np.round(100 * ev / ev.sum()), "G": G, "alpha": alpha,
            "cubeSP": CP, "Splus": S, "eig_Splus": evS, "tau_Splus": np.round(100 * evS / evS.sum()),
            "F": F[:, :nfact], "ctr": ctr[:, :nfact], "cos2": cos2[:, :nfact], "d2": d2,
            "projection": Proj, "partial_F": partial}
