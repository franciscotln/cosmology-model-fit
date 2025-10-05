from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# --- PLANCK DISTANCE PRIORS (Prakhar Bansal+2025 arXiv:2502.07185v2) ---
DISTANCE_PRIORS = np.array(
    [
        1.7504,  # R
        301.77,  # lA
        0.022371,  # Ωb h^2
    ],
    dtype=np.float64,
)
cov_matrix = 10**-8 * np.array(
    [
        [1559.83, -1325.41, -36.45],
        [-1325.41, 714691.80, 269.77],
        [-36.45, 269.77, 2.10],
    ],
    dtype=np.float64,
)
inv_cov_mat = np.linalg.inv(cov_matrix)
N_EFF = 3.046
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4728e-5 * (TCMB / 2.7255) ** 4


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


def rs_z(Ez_func, z, params, H0, Ob_h2):
    Rb = 3 * Ob_h2 / (4 * O_GAMMA_H2)

    def integrand(a):
        denom = a**2 * Ez_func(1 / a - 1, params) * np.sqrt(3 * (1 + Rb * a))
        return c / denom

    return quad(integrand, 0, 1 / (1 + z))[0] / H0


def DA_z(Ez_func, z, params, H0):
    I = quad(lambda zp: c / Ez_func(zp, params), 0, z)[0]
    return (I / H0) / (1.0 + z)


def cmb_distances(Ez_func, params, H0, Om, Ob_h2):
    zstar = z_star(wb=Ob_h2, wm=Om * (H0 / 100) ** 2)
    rs_star = rs_z(Ez_func, zstar, params, H0, Ob_h2)
    DA_star = DA_z(Ez_func, zstar, params, H0)

    R = np.sqrt(Om) * H0 * (1 + zstar) * DA_star / c
    lA = (1 + zstar) * np.pi * DA_star / rs_star
    return np.array([R, lA, Ob_h2])


@njit
def r_drag(wb, wm):
    # arXiv:2106.00428v2 (eq 8)
    a1 = 0.00257366
    a2 = 0.05032
    a3 = 0.013
    a4 = 0.7720642
    a5 = 0.24346362
    a6 = 0.00641072
    a7 = 0.5350899
    a8 = 32.7525
    a9 = 0.315473

    term_A_denominator = (a1 * (wb**a2)) + (a3 * (wb**a4) * (wm**a5)) + (a6 * (wm**a7))
    term_A = 1.0 / term_A_denominator
    term_B = a8 / (wm**a9)
    return term_A - term_B


@njit
def z_star(wb, wm):
    return 1090.0  # fixed according to Prakhar Bansal+2025


@njit
def z_drag(wb, wm):
    # arXiv:astro-ph/9510117v2 (eq-2)
    factor_1 = 0.313 * wm**-0.419
    factor_2 = 1 + 0.607 * wm**0.674
    b1_val = factor_1 * factor_2
    b2_val = 0.238 * wm**0.223

    fraction_numerator = wm**0.251
    fraction_denominator = 1 + 0.659 * wm**0.828
    bracket_term = 1 + b1_val * wb**b2_val

    return 1345 * (fraction_numerator / fraction_denominator) * bracket_term
