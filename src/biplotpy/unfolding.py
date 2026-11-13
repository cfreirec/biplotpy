"""Metric (ratio) unfolding by SMACOF for a rectangular dissimilarity matrix
(rows: individuals, columns: objects), following ``smacof::unfolding``
(de Leeuw & Mair, 2009) for ``type = "ratio"`` and no constraints."""
from __future__ import annotations

import numpy as np


def _dist(x, y):
    return np.sqrt(np.maximum(((x[:, None, :] - y[None, :, :]) ** 2).sum(2), 0))


def _pstress(dhat, d, w, wpp, omega=1.0, lam=0.5):
    nrmd2 = np.sum(w * (dhat - d) ** 2)
    mnc1 = nrmd2 / wpp
    wsum = np.sum(w * dhat)
    nrmw2 = np.sum(w * dhat ** 2)
    nrmm2 = wsum ** 2 / wpp
    nrmv2 = nrmw2 - nrmm2
    mnc2 = (nrmv2 + omega * nrmm2) / nrmv2
    return np.sqrt(mnc1 ** lam * mnc2)


def unfolding(delta, ndim=2, weights=None, init=None, itmax=10000, eps=1e-6, relax=True) -> dict:
    """Returns row and column configurations, distances, stress-1, stress per
    point (%) and number of iterations."""
    diss = np.asarray(delta, float)
    n, m = diss.shape
    w = np.ones((n, m)) if weights is None else np.asarray(weights, float)
    w = np.where(np.isnan(diss), 0.0, w)
    wpp = w.sum()
    dhat = np.nan_to_num(diss)
    e = dhat ** 2
    e = -0.5 * (e - (e.sum(1)[:, None] / m + e.sum(0)[None, :] / n) + e.sum() / (n * m))
    if init is None:
        U, s, Vt = np.linalg.svd(e)
        x = U[:, :ndim]
        y = e.T @ x
    else:
        x, y = np.asarray(init[0], float)[:, :ndim], np.asarray(init[1], float)[:, :ndim]
    dil = np.trace(x.T @ (e @ y)) / np.trace((x.T @ x) @ (y.T @ y))
    x = np.sqrt(dil) / np.sqrt((x ** 2).sum()) * x
    y = np.sqrt(dil) / np.sqrt((y ** 2).sum()) * y
    d = _dist(x, y)
    lold = _pstress(dhat, d, w, wpp)
    wr, wc = w.sum(1), w.sum(0)
    v = np.linalg.inv(np.diag(wc) + 1 / m - w.T @ (w / wr[:, None])) - 1 / m
    itel = 1
    while True:
        d0 = d < eps
        b = w * dhat * (~d0) / (d + d0)
        br, bc = b.sum(1), b.sum(0)
        xraw = br[:, None] * x - b @ y
        yraw = bc[:, None] * y - b.T @ x
        xold, yold = x, y
        y = v @ (yraw + w.T @ (xraw / wr[:, None]))
        x = (xraw + w @ y) / wr[:, None]
        if relax and itel > 100:
            x = 2 * x - xold
            y = 2 * y - yold
        d = _dist(x, y)
        lnew = _pstress(dhat, d, w, wpp)
        if itel == itmax or 2 * (lold - lnew) <= eps * (lold + lnew + 1e-15):
            break
        lold = lnew
        itel += 1
    resmat = (d - dhat) ** 2
    spp_col = (w * resmat).sum(0) / w.sum(0); spp_col = spp_col / spp_col.sum() * 100
    spp_row = (w * resmat).sum(1) / w.sum(1); spp_row = spp_row / spp_row.sum() * 100
    scale = np.sqrt(w.sum() / np.sum(w * d ** 2))
    d, x, y, dhat = scale * d, scale * x, scale * y, scale * dhat
    stress = np.sqrt(max(0.0, 1 - np.sum(w * dhat * d) ** 2 / (np.sum(w * dhat ** 2) * np.sum(w * d ** 2))))
    return {"conf_row": x, "conf_col": y, "confdist": d, "dhat": dhat, "stress": stress, "spp_row": spp_row,
            "spp_col": spp_col, "niter": itel}
