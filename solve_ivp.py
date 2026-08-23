"""
Numpy/numba-only replacement for a subset of scipy.integrate.solve_ivp.

Implements the Dormand-Prince RK45 method with adaptive step size control.
Meant to be called with @njit-decorated right-hand-side functions and can
itself be decorated with @njit.

Differences from scipy.integrate.solve_ivp:
- Only forward integration (t_span[1] >= t_span[0]) is supported.
- `t_eval` is required (not optional). Pass an empty float64 array to get
  the solver's own accepted step points back instead of interpolated ones.
- `args` is required (not optional). Pass an empty tuple `()` if `fun`
  takes no extra arguments.
- The returned object only exposes `.t` and `.y`, matching the fields used
  throughout this project.
"""

from collections import namedtuple
import numpy as np
from numba import njit
from interpolator import interp_hermite

_C2, _C3, _C4, _C5 = 1 / 5, 3 / 10, 4 / 5, 8 / 9

_A21 = 1 / 5
_A31, _A32 = 3 / 40, 9 / 40
_A41, _A42, _A43 = 44 / 45, -56 / 15, 32 / 9
_A51, _A52, _A53, _A54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
_A61, _A62, _A63, _A64, _A65 = (
    9017 / 3168,
    -355 / 33,
    46732 / 5247,
    49 / 176,
    -5103 / 18656,
)
_A71, _A73, _A74, _A75, _A76 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84

_E1, _E3, _E4, _E5, _E6, _E7 = (
    71 / 57600,
    -71 / 16695,
    71 / 1920,
    -17253 / 339200,
    22 / 525,
    -1 / 40,
)

_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_ERR_EXPONENT = -1 / 5

OdeResult = namedtuple("OdeResult", ["t", "y"])


@njit
def _rms_norm(x):
    s = 0.0
    for i in range(x.size):
        s += x[i] * x[i]
    return np.sqrt(s / x.size)


@njit
def _select_initial_step(fun, t0, y0, f0, rtol, atol, args):
    scale = atol + np.abs(y0) * rtol
    d0 = _rms_norm(y0 / scale)
    d1 = _rms_norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * f0
    f1 = fun(t0 + h0, y1, *args)
    d2 = _rms_norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / 5)

    return min(100 * h0, h1)


@njit
def solve_ivp(fun, t_span, y0, t_eval, rtol=1e-6, atol=1e-8, args=()):
    t0, tf = t_span[0], t_span[1]
    y = np.asarray(y0, dtype=np.float64).copy()
    n = y.size

    k1 = fun(t0, y, *args)
    h = _select_initial_step(fun, t0, y, k1, rtol, atol, args)

    cap = 256
    t_arr = np.empty(cap, dtype=np.float64)
    y_arr = np.empty((n, cap), dtype=np.float64)
    dy_arr = np.empty((n, cap), dtype=np.float64)

    t_arr[0] = t0
    y_arr[:, 0] = y
    dy_arr[:, 0] = k1
    count = 1

    t = t0
    while t < tf:
        if t + h > tf:
            h = tf - t

        while True:
            k2 = fun(t + _C2 * h, y + h * _A21 * k1, *args)
            k3 = fun(t + _C3 * h, y + h * (_A31 * k1 + _A32 * k2), *args)
            k4 = fun(t + _C4 * h, y + h * (_A41 * k1 + _A42 * k2 + _A43 * k3), *args)
            k5 = fun(
                t + _C5 * h,
                y + h * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4),
                *args,
            )
            k6 = fun(
                t + h,
                y + h * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5),
                *args,
            )

            y_new = y + h * (_A71 * k1 + _A73 * k3 + _A74 * k4 + _A75 * k5 + _A76 * k6)
            k7 = fun(t + h, y_new, *args)

            err = h * (_E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7)
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
            error_norm = _rms_norm(err / scale)

            if error_norm <= 1.0:
                factor = (
                    _MAX_FACTOR
                    if error_norm == 0.0
                    else min(_MAX_FACTOR, _SAFETY * error_norm**_ERR_EXPONENT)
                )
                t = t + h
                y = y_new
                k1 = k7

                if count >= cap:
                    cap *= 2
                    t_arr2 = np.empty(cap, dtype=np.float64)
                    y_arr2 = np.empty((n, cap), dtype=np.float64)
                    dy_arr2 = np.empty((n, cap), dtype=np.float64)
                    t_arr2[:count] = t_arr[:count]
                    y_arr2[:, :count] = y_arr[:, :count]
                    dy_arr2[:, :count] = dy_arr[:, :count]
                    t_arr, y_arr, dy_arr = t_arr2, y_arr2, dy_arr2

                t_arr[count] = t
                y_arr[:, count] = y
                dy_arr[:, count] = k1
                count += 1

                h = h * factor
                break
            else:
                factor = max(_MIN_FACTOR, _SAFETY * error_norm**_ERR_EXPONENT)
                h = h * factor

    t_out = t_arr[:count]
    y_out = y_arr[:, :count]
    dy_out = dy_arr[:, :count]

    if t_eval.size == 0:
        return OdeResult(np.ascontiguousarray(t_out), np.ascontiguousarray(y_out))

    y_eval = np.empty((n, t_eval.size), dtype=np.float64)
    for i in range(n):
        y_eval[i, :] = interp_hermite(t_eval, t_out, y_out[i, :], dy_out[i, :])

    return OdeResult(np.ascontiguousarray(t_eval), y_eval)
