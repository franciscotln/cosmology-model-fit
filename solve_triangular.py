import numpy as np
from numba import njit


@njit
def solve_triangular(L, b):
    """
    Solve the system of linear equations L * y = b for y, where L is a lower triangular matrix.
    """
    n = len(b)
    y = np.empty_like(b)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y
