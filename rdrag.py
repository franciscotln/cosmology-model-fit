from numba import njit
import numpy as np


@njit
def r_drag(omegabh2, omegamh2):
    """
    Eisenstein, Hu (arXiv:astro-ph/9709112)
    Re-fitted 3 coefficients
    """
    a, b, c = 46.38896452, 7.06211353, 8.85743299
    return a * np.log(b / omegamh2) / np.sqrt(1.0 + c * omegabh2**0.75)
