from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# --- PLANCK DISTANCE PRIORS (Rubin+ arXiv:2311.12098v2) ---
# θ ≡ rs(z*) / DM(z*)
# R ≡ √(Ωm H0²) * DA(z*) * (1 + z*) / c
DISTANCE_PRIORS = np.array(
    [
        1.7492768568335353,  # R
        1.039233410719115,  # 100 θ
        0.02239245,  # ωb
    ],
    dtype=np.float64,
)
inv_cov_mat = np.array(
    [
        [92701.58172970748, 348041.8137694254, 1613445.8550364415],
        [348041.8137694254, 13114681.644682042, -3019007.1687636944],
        [1613445.8550364415, -3019007.1687636944, 80842256.32398143],
    ],
    dtype=np.float64,
)
N_EFF = 3.04
TCMB = 2.72548  # K
O_GAMMA_H2 = 2.4729e-5 * (TCMB / 2.72548) ** 4


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
    theta_100 = 100 * rs_star / ((1 + zstar) * DA_star)
    return np.array([R, theta_100, Ob_h2])


@njit
def r_drag(wb, wm):
    # arXiv:2106.00428v2 (eq 6)
    numerator = 45.5337 * np.log(7.20376 / wm)
    denominator = np.sqrt(1 + 9.98592 * (wb**0.801347))
    return numerator / denominator


@njit
def z_star(wb, wm):
    # arXiv:astro-ph/9510117v2 (eq-1)
    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    factor_1 = 1 + 0.00124 * wb**-0.738
    factor_2 = 1 + g1 * wm**g2

    return 1048 * factor_1 * factor_2


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
