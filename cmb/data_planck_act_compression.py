"""
arXiv:2503.14452v2
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_lcdm_get.html
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_info.html
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_prod_table.html
"""

from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# arXiv:2503.14452v2 P-ACT baseline LCDM constraints
# R, lA = π / θ*, ωb = Ωb h^2
DISTANCE_PRIORS = np.array([1.74800862, 301.802768, 0.0224967038], dtype=np.float64)
covariance = np.array(
    [
        [1.53695593e-05, 1.06033525e-04, -2.09800310e-07],
        [1.06033525e-04, 5.56342893e-03, -1.59024331e-06],
        [-2.09800310e-07, -1.59024331e-06, 1.23736642e-08],
    ],
    dtype=np.float64,
)
inv_cov_mat = np.linalg.inv(covariance)

N_EFF = 3.044
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4729e-5


@njit
def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


def rs_z(Ez_func, z, params, H0, Ob_h2):
    """Sound horizon at redshift z."""
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
    DM_star = (1.0 + zstar) * DA_z(Ez_func, zstar, params, H0)

    R = np.sqrt(Om) * H0 * DM_star / c
    lA = np.pi * DM_star / rs_star
    return np.array([R, lA, Ob_h2])


@njit
def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""
    SCALING_FID = 1.00110357

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
    return SCALING_FID * (term_A - term_B)


@njit
def z_star(wb, wm):
    """arXiv:astro-ph/9510117v2 (eq-1)"""
    SCALING_FID = 0.9969799

    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    factor_1 = 1 + 0.00124 * wb**-0.738
    factor_2 = 1 + g1 * wm**g2
    return SCALING_FID * 1048 * factor_1 * factor_2


@njit
def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""
    SCALING_FID = 0.998385  # reproduces rdrag from integral

    return (
        SCALING_FID
        * (1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615)
        * wm**-0.714129
    )
