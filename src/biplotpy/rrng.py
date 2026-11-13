"""Random numbers identical to those of R.

``RRNG(seed)`` reproduces ``set.seed(seed); rnorm(n)`` of R (Mersenne-Twister
with R's seed scrambling, normal draws by inversion with
``qnorm`` of Wichura, algorithm AS 241). With it the procedures of biplotpy that
start from random numbers (alternating least squares from a random start,
bootstrap resampling, random starts of BinaryPLSR) give, seed by seed, the
same results as their R counterparts. About 97% of the normal draws are bitwise equal
to R's and the rest differ by at most a few units in the last place (rounding
inside ``qnorm``), which does not change any p-value in practice.
"""
from __future__ import annotations

import math

import numpy as np

_I2_32M1 = 2.328306437080797e-10      # 1 / (2^32 - 1)
_BIG = 134217728.0                    # 2^27


def _r_seed_state(seed: int) -> np.ndarray:
    s = int(seed) & 0xFFFFFFFF
    for _ in range(50):
        s = (69069 * s + 1) & 0xFFFFFFFF
    st = []
    for _ in range(625):
        s = (69069 * s + 1) & 0xFFFFFFFF
        st.append(s)
    return np.array(st[1:], dtype=np.uint32)


def qnorm_r(p):
    """R's qnorm(p, 0, 1) (Wichura AS241, nmath/qnorm.c), vectorized."""
    p = np.asarray(p, float); q = p - 0.5; out = np.empty_like(p)
    c = np.abs(q) <= 0.425
    r = 0.180625 - q[c] * q[c]
    out[c] = q[c] * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r
                + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r + 133.14166789178437745) * r
                + 3.387132872796366608) / (((((((r * 5226.495278852545925 + 28729.085735721942674) * r + 39307.89580009271061) * r
                + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r + 42.313330701600911252) * r + 1.)
    nc = ~c
    pp = np.where(q[nc] < 0, p[nc], 1 - p[nc])
    r = np.sqrt(-np.array([math.log(v) for v in pp]))
    v = np.empty_like(r)
    a = r <= 5.
    ra = r[a] - 1.6
    v[a] = (((((((ra * 7.7454501427834140764e-4 + .0227238449892691845833) * ra + .24178072517745061177) * ra
              + 1.27045825245236838258) * ra + 3.64784832476320460504) * ra + 5.7694972214606914055) * ra + 4.6303378461565452959) * ra
              + 1.42343711074968357734) / (((((((ra * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * ra + .0151986665636164571966) * ra
              + .14810397642748007459) * ra + .68976733498510000455) * ra + 1.6763848301838038494) * ra + 2.05319162663775882187) * ra + 1.)
    b = ~a
    rb = r[b] - 5.
    v[b] = (((((((rb * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * rb + .0012426609473880784386) * rb
              + .026532189526576123093) * rb + .29656057182850489123) * rb + 1.7848265399172913358) * rb + 5.4637849111641143699) * rb
              + 6.6579046435011037772) / (((((((rb * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * rb + 1.8463183175100546818e-5) * rb
              + 7.868691311456132591e-4) * rb + .0148753612908506148525) * rb + .13692988092273580531) * rb + .59983220655588793769) * rb + 1.)
    v[q[nc] < 0] = -v[q[nc] < 0]
    out[nc] = v
    return out


class RRNG:
    """Generator compatible with R's default ``RNGkind`` (Mersenne-Twister, Inversion).

    It exposes ``standard_normal`` and ``integers`` so that it can be passed
    wherever the package accepts a ``numpy.random.Generator``. Draws are taken
    in the order R takes them (``matrix(rnorm(n), ...)`` fills by columns).
    """

    def __init__(self, seed: int):
        self.bg = np.random.MT19937()
        self.bg.state = {"bit_generator": "MT19937", "state": {"key": _r_seed_state(seed), "pos": 624}}

    def unif(self, n: int) -> np.ndarray:
        """``runif(n)`` of R."""
        v = self.bg.random_raw(n).astype(np.float64) * 2.3283064365386963e-10
        v[v <= 0.0] = 0.5 * _I2_32M1
        v[1.0 - v <= 0.0] = 1.0 - 0.5 * _I2_32M1
        return v

    def standard_normal(self, size=None) -> np.ndarray:
        """``rnorm(n)`` of R; ``size`` may be an int or a shape (filled in C order)."""
        n = 1 if size is None else int(np.prod(size))
        u = self.unif(2 * n).reshape(n, 2)
        z = qnorm_r((np.floor(_BIG * u[:, 0]) + u[:, 1]) / _BIG)
        return z[0] if size is None else z.reshape(size)

    def integers(self, high: int) -> int:
        """``sample.int(high, 1) - 1`` of R (sample.kind = "Rejection")."""
        bits = int(math.ceil(math.log2(high))) if high > 1 else 0
        while True:
            v = 0.0
            for _ in range(0, bits + 1, 16):
                v1 = int(math.floor(self.unif(1)[0] * 65536))
                v = 65536 * v + v1
            v = int(v) & ((1 << bits) - 1)       # R masks always (bits = 0 gives 0)
            if v < high:
                return int(v)

    def rnorm(self, n: int) -> np.ndarray:
        """``rnorm(n)`` of R."""
        return self.standard_normal(int(n))

    def sample_replace(self, n: int, size: int) -> np.ndarray:
        """0-based indices of ``sample.int(n, size, replace = TRUE)``."""
        return np.array([self.integers(n) for _ in range(int(size))], dtype=int)

    def sample_noreplace(self, n: int, size: int) -> np.ndarray:
        """0-based indices of ``sample.int(n, size)`` (no hashing, n <= 1e7)."""
        x = list(range(int(n)))
        m = int(n)
        out = []
        for _ in range(int(size)):
            j = self.integers(m)
            out.append(x[j])
            m -= 1
            x[j] = x[m]
        return np.array(out, dtype=int)
