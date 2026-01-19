import numpy as np
from numba import njit


@njit
def interp_quad(x, xp, fp):
    x = np.asarray(x)
    idx = np.searchsorted(xp, x) - 1
    i = np.minimum(np.maximum(idx, 0), len(xp) - 3)
    x0, x1, x2 = xp[i], xp[i + 1], xp[i + 2]
    y0, y1, y2 = fp[i], fp[i + 1], fp[i + 2]
    term0 = y0 * ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
    term1 = y1 * ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
    term2 = y2 * ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))

    return term0 + term1 + term2


@njit
def interp_cubic(x, xp, fp):
    x = np.asarray(x)
    idx = np.searchsorted(xp, x) - 2
    i = np.minimum(np.maximum(idx, 0), len(xp) - 4)
    x0, x1, x2, x3 = xp[i], xp[i + 1], xp[i + 2], xp[i + 3]
    y0, y1, y2, y3 = fp[i], fp[i + 1], fp[i + 2], fp[i + 3]
    t0 = y0 * ((x - x1) * (x - x2) * (x - x3)) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
    t1 = y1 * ((x - x0) * (x - x2) * (x - x3)) / ((x1 - x0) * (x1 - x2) * (x1 - x3))
    t2 = y2 * ((x - x0) * (x - x1) * (x - x3)) / ((x2 - x0) * (x2 - x1) * (x2 - x3))
    t3 = y3 * ((x - x0) * (x - x1) * (x - x2)) / ((x3 - x0) * (x3 - x1) * (x3 - x2))

    return t0 + t1 + t2 + t3
