"""
Planck+ACT baseline LCDM constraints arXiv:2503.14452v2
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_lcdm_get.html
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_info.html
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_prod_table.html
"""

from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# R, lA = π / θ*, ωb = Ωb h^2
DISTANCE_PRIORS = np.array([1.74795802, 301.803306, 0.0224962530], dtype=np.float64)
covariance = np.array(
    [
        [1.54911112e-05, 1.03997132e-04, -2.10953275e-07],
        [1.03997132e-04, 5.43880523e-03, -1.53612827e-06],
        [-2.10953275e-07, -1.53612827e-06, 1.23574770e-08],
    ],
    dtype=np.float64,
)
inv_cov_mat = np.linalg.inv(covariance)

N_EFF = 3.044
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.472975328714087e-05

T_nu0 = (4 / 11) ** (1 / 3) * TCMB  # K
T_nu0_eV = T_nu0 * 8.617333262e-5  #  1.67639e-04 eV
mnu_tot = 0.06  # total mass [eV]
Omnu_h2 = mnu_tot / (94.0641 / (N_EFF / 3.0) ** 0.75)  # present-day Omega_nu*h^2 (CAMB)
z_nr = mnu_tot / (3.15 * T_nu0_eV)


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + Neff * (7 / 8) * (4 / 11) ** (4 / 3))


Or_h2 = Omega_r_h2(N_EFF - (N_EFF / 3))

fact0 = (
    (3.0 / N_EFF) ** (1 / 4)
    * (mnu_tot / 94.0641)
    * (8 / 7)
    * (11 / 4) ** (4 / 3)
    / O_GAMMA_H2
)


@njit
def Omnu_z(z):
    """
    Computes the appox. evolution of massive neutrino
    energy density with redshift
    """
    zp1 = 1 + z
    fact = 1.0 * fact0
    return zp1**4 * np.sqrt(1 + fact**2 / zp1**2) / np.sqrt(1 + fact**2)


def rs_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    Rb = 3 * Obh2 / (4 * O_GAMMA_H2)

    def integrand(a):
        Ez = Ez_func(1 / a - 1, H0, Obh2, Och2, w0, wa)
        denom = a**2 * Ez * np.sqrt(3 * (1 + Rb * a))
        return 1 / denom

    return (c / H0) * quad(integrand, 1e-08, 1 / (1 + z_lim))[0]


def DM_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    integral = quad(lambda z: 1 / Ez_func(z, H0, Obh2, Och2, w0, wa), 0.0, z_lim)[0]
    return integral * c / H0


def cmb_distances(Ez_func, H0, Ob_h2, Oc_h2, w0=-1, wa=0):
    """
    returns (R, lA = π / θ*, ωb)
    """
    Om_h2 = Oc_h2 + Ob_h2 + Omnu_h2
    zstar = z_star(wb=Ob_h2, wm=Om_h2)
    rs_star = rs_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)
    DM_star = DM_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)

    R = 100 * np.sqrt(Om_h2) * DM_star / c
    lA = np.pi * DM_star / rs_star
    return np.array([R, lA, Ob_h2])


@njit
def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""
    SCALING_FID = 1.0011037178265139

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
    SCALING_FID = 0.9981705753766743

    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    factor_1 = 1 + 0.00124 * wb**-0.738
    factor_2 = 1 + g1 * wm**g2
    return SCALING_FID * 1048 * factor_1 * factor_2


@njit
def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""
    SCALING_FID = 1.000031880280976

    return (
        SCALING_FID
        * (1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615)
        * wm**-0.714129
    )
