from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# --- PLANCK DISTANCE PRIORS (Chen+2018 arXiv:1808.05724v1) ---
DISTANCE_PRIORS = np.array(
    [
        1.750235,  # R
        301.4707,  # lA
        0.02235976,  # Ωb h^2
    ],
    dtype=np.float64,
)
inv_cov_mat = np.array(
    [
        [94392.3971, -1360.4913, 1664517.2916],
        [-1360.4913, 161.4349, 3671.618],
        [1664517.2916, 3671.618, 79719182.5162],
    ],
    dtype=np.float64,
)
N_EFF = 3.046
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4728e-5 * (TCMB / 2.7255) ** 4


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


@njit
def z_star(wb, wm):
    # arXiv:2106.00428v2 (eq A4)
    return wm**-0.731631 + (
        (391.672 * wm**-0.372296 + 937.422 * wb**-0.97966) * wm**0.0192951 * wb**0.93681
    )


@njit
def z_drag(wb, wm):
    # arXiv:2106.00428v2 (eq A2)
    return (
        1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615
    ) * wm**-0.714129


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
