import numpy as np
from numba import njit


@njit
def _pchip_slopes(x, y):
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    d = np.zeros(n, dtype=np.float64)
    h = np.empty(n - 1, dtype=np.float64)
    delta = np.empty(n - 1, dtype=np.float64)

    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]
        delta[i] = (y[i + 1] - y[i]) / h[i]

    if n == 2:
        d[0] = delta[0]
        d[1] = delta[0]
        return d

    # --- Interior points (Weighted Harmonic Mean) ---
    for i in range(1, n - 1):
        delta_i_minus_1 = delta[i - 1]
        delta_i = delta[i]
        h_i_minus_1 = h[i - 1]
        h_i = h[i]
        if (
            delta_i_minus_1 != 0.0
            and delta_i != 0.0
            and delta_i_minus_1 * delta_i > 0.0
        ):
            w1 = 2.0 * h_i + h_i_minus_1
            w2 = h_i + 2.0 * h_i_minus_1
            d[i] = (w1 + w2) / (w1 / delta_i_minus_1 + w2 / delta_i)
        else:
            d[i] = 0.0

    # --- Start Point (d[0]) ---
    # Non-centered three-point formula
    d0 = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])

    # Check for sign change or overshoot
    if delta[0] == 0.0 or np.sign(d0) != np.sign(delta[0]):
        d[0] = 0.0
    elif (np.sign(delta[0]) != np.sign(delta[1])) and (abs(d0) > abs(3 * delta[0])):
        d[0] = 3 * delta[0]
    else:
        d[0] = d0

    # --- End Point (d[n-1]) ---
    # Non-centered three-point formula for the end
    dn = ((2 * h[n - 2] + h[n - 3]) * delta[n - 2] - h[n - 2] * delta[n - 3]) / (
        h[n - 2] + h[n - 3]
    )

    if delta[n - 2] == 0.0 or np.sign(dn) != np.sign(delta[n - 2]):
        d[n - 1] = 0.0
    elif (np.sign(delta[n - 2]) != np.sign(delta[n - 3])) and (
        abs(dn) > abs(3 * delta[n - 2])
    ):
        d[n - 1] = 3 * delta[n - 2]
    else:
        d[n - 1] = dn

    return d


@njit
def _pchip_interp(xq, x, y, d, exact):
    out = np.empty_like(xq, dtype=np.float64)
    h = np.diff(x)

    for k in range(len(xq)):
        xi = xq[k]

        if not exact:
            if xi <= x[0]:
                out[k] = y[0]
                continue
            if xi >= x[-1]:
                out[k] = y[-1]
                continue
        else:
            if xi <= x[0]:
                out[k] = y[0] + d[0] * (xi - x[0])
                continue
            if xi >= x[-1]:
                out[k] = y[-1] + d[-1] * (xi - x[-1])
                continue

        i = np.searchsorted(x, xi) - 1

        h_i = h[i]
        t = (xi - x[i]) / h_i

        t2 = t * t
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        out[k] = h00 * y[i] + h10 * h_i * d[i] + h01 * y[i + 1] + h11 * h_i * d[i + 1]
    return out


@njit
def interp_pchip(xq, x, y):
    d = _pchip_slopes(x, y)
    return _pchip_interp(xq, x, y, d, False)


@njit
def interp_hermite(xq, x, y, y_prime):
    return _pchip_interp(xq, x, y, y_prime, True)
