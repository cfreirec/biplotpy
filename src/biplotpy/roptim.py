"""R's ``optim(method = "CG")``: conjugate gradients of Nash (1990), as coded in
R's ``src/appl/optim.c`` (``cgmin``), so that procedures that MultBiplotR runs
with ``optim(..., method = "CG")`` follow the same path in Python."""
from __future__ import annotations

import math

import numpy as np

_STEPREDN = 0.2
_ACCTOL = 0.0001
_RELTEST = 10.0


def optim_cg(fn, gr, par, args=(), type: int = 1, maxit: int = 100, abstol: float = -math.inf,
             reltol: float = 1.4901161193847656e-08):
    """Minimize ``fn`` from ``par`` with R's CG method (``type`` 1 Fletcher-Reeves,
    2 Polak-Ribiere, 3 Beale-Sorenson). Returns a dict like ``optim``: ``par``,
    ``value``, ``counts`` and ``convergence`` (1 when ``maxit`` is reached)."""
    B = [float(v) for v in np.asarray(par, float).ravel()]
    n = len(B)
    f_ = lambda b: float(fn(np.array(b), *args))
    g_ = lambda b: [float(v) for v in np.asarray(gr(np.array(b), *args), float).ravel()]
    if maxit <= 0:
        return {"par": np.array(B), "value": f_(B), "counts": (0, 0), "convergence": 0}
    c = [0.0] * n; t = [0.0] * n; X = [0.0] * n
    setstep = 1.7
    fail = 0
    cyclimit = n
    tol = reltol * n * math.sqrt(reltol)
    f = f_(B)
    if not math.isfinite(f):
        raise ValueError("Function cannot be evaluated at initial parameters")
    Fmin = f
    funcount, gradcount = 1, 0
    steplength = 1.0
    while True:
        t = [0.0] * n; c = [0.0] * n
        cycle = 0
        oldstep = 1.0
        count = 0
        while True:
            cycle += 1
            gradcount += 1
            if gradcount > maxit:
                return {"par": np.array(X), "value": Fmin, "counts": (funcount, gradcount), "convergence": 1}
            g = g_(B)
            G1 = 0.0; G2 = 0.0
            for i in range(n):
                X[i] = B[i]
                if type == 1:
                    G1 += g[i] * g[i]; G2 += c[i] * c[i]
                elif type == 2:
                    G1 += g[i] * (g[i] - c[i]); G2 += c[i] * c[i]
                else:
                    G1 += g[i] * (g[i] - c[i]); G2 += t[i] * (g[i] - c[i])
                c[i] = g[i]
            if G1 > tol:
                G3 = G1 / G2 if G2 > 0.0 else 1.0
                gradproj = 0.0
                for i in range(n):
                    t[i] = t[i] * G3 - g[i]
                    gradproj += t[i] * g[i]
                steplength = oldstep
                accpoint = False
                while True:
                    count = 0
                    for i in range(n):
                        B[i] = X[i] + steplength * t[i]
                        if _RELTEST + X[i] == _RELTEST + B[i]:
                            count += 1
                    if count < n:
                        f = f_(B)
                        funcount += 1
                        accpoint = math.isfinite(f) and f <= Fmin + gradproj * steplength * _ACCTOL
                        if not accpoint:
                            steplength *= _STEPREDN
                        else:
                            Fmin = f
                    if count == n or accpoint:
                        break
                if count < n:
                    newstep = 2 * (f - Fmin - gradproj * steplength)
                    if newstep > 0:
                        newstep = -(gradproj * steplength * steplength / newstep)
                        for i in range(n):
                            B[i] = X[i] + newstep * t[i]
                        Fmin = f
                        f = f_(B)
                        funcount += 1
                        if f < Fmin:
                            Fmin = f
                        else:
                            for i in range(n):
                                B[i] = X[i] + steplength * t[i]
            oldstep = setstep * steplength
            if oldstep > 1.0:
                oldstep = 1.0
            if not ((count != n) and (G1 > tol) and (cycle != cyclimit)):
                break
        if not ((cycle != 1) or ((count != n) and (G1 > tol) and Fmin > abstol)):
            break
    # like R's optim, the result is the last base point X (not the last trial point)
    return {"par": np.array(X), "value": Fmin, "counts": (funcount, gradcount), "convergence": fail}
