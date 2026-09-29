from numba import njit
import numpy as np


@njit
def r_drag(omegabh2, omegamh2):
    """
    Eisenstein, Hu (arXiv:astro-ph/9709112)
    Re-fitted 3 coefficients
    """
    return 46.38896452 * np.log(7.06211353 / omegamh2) / np.sqrt(1.0 + 8.85743299 * omegabh2**0.75)
