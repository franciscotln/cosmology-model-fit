import numpy as np
from numba import njit


@njit
def _pchip_slopes(x, y):
    n = len(x)
    d = np.zeros(n)

    h = np.empty(n - 1)
    delta = np.empty(n - 1)

    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]
        delta[i] = (y[i + 1] - y[i]) / h[i]

    # interior points
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0.0:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        else:
            d[i] = 0.0

    # endpoints (one-sided, shape-preserving)
    d[0] = delta[0]
    d[-1] = delta[-1]

    return d


@njit
def _pchip_interp(xq, x, y, d):
    out = np.empty_like(xq, dtype=np.float64)

    for k in range(len(xq)):
        xi = xq[k]

        if xi <= x[0]:
            out[k] = y[0]
            continue
        if xi >= x[-1]:
            out[k] = y[-1]
            continue

        i = np.searchsorted(x, xi) - 1

        h = x[i + 1] - x[i]
        t = (xi - x[i]) / h

        t2 = t * t
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        out[k] = h00 * y[i] + h10 * h * d[i] + h01 * y[i + 1] + h11 * h * d[i + 1]

    return out


@njit
def interp_pchip(xq, x, y):
    d = _pchip_slopes(x, y)
    return _pchip_interp(xq, x, y, d)
