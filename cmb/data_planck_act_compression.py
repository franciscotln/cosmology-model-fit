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
import nu_evolution as neutrino

c = c0 / 1000  # km/s

DISTANCE_PRIORS = np.array([1.74795802, 301.803306, 0.0224962530], dtype=np.float64)
"""Compressed Planck + ACT DR6 priors: (R, lA = π / θ*, ωb)"""

covariance = np.array(
    [
        [1.54911112e-05, 1.03997132e-04, -2.10953275e-07],
        [1.03997132e-04, 5.43880523e-03, -1.53612827e-06],
        [-2.10953275e-07, -1.53612827e-06, 1.23574770e-08],
    ],
    dtype=np.float64,
)
inv_cov_mat = np.linalg.inv(covariance)

k_B = 8.617333262e-5  # eV/K
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.472975328714087e-05

N_EFF = 3.044
T_nu0 = (4 / 11) ** (1 / 3) * (N_EFF / 3) ** (1 / 4) * TCMB  # K
T_nu0_eV = T_nu0 * k_B  # eV
mnu_tot = 0.06  # total mass [eV]
m0 = mnu_tot / T_nu0_eV
Omnu_h2 = mnu_tot / (94.0641 / (N_EFF / 3.0) ** 0.75)


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + Neff * (7 / 8) * (4 / 11) ** (4 / 3))


Or_h2 = Omega_r_h2(2 * N_EFF / 3)


# 1 massive neutrino section
rho0 = neutrino.compute_rho0(m0)
qs = neutrino.compute_qs(m0)
qs_sq = qs**2
ws = neutrino.weights


@njit
def Omnu_z(z):
    """
    Energy density rho(z) for massive neutrinos using the 5-node approximation
    """
    zp1 = 1.0 + z
    mz_sq = (m0 / zp1) ** 2
    f0 = np.sqrt(qs_sq[0] + mz_sq)
    f1 = np.sqrt(qs_sq[1] + mz_sq)
    f2 = np.sqrt(qs_sq[2] + mz_sq)
    f3 = np.sqrt(qs_sq[3] + mz_sq)
    f4 = np.sqrt(qs_sq[4] + mz_sq)
    weighted_sum = f0 * ws[0] + f1 * ws[1] + f2 * ws[2] + f3 * ws[3] + f4 * ws[4]
    return zp1**4 * weighted_sum / rho0


@njit
def z_star(wb, wm):
    """arXiv:2106.00428v2 (eq A-4)"""

    s1, s2, b, m = (0.70130133, 1.00839438, 1.02468387, 1.18438972)

    wb = wb**b
    wm = wm**m

    return (
        wm**-0.7316314841257655
        + s1 * 391.6723594873167 * wb**0.9368102670600895 * wm**-0.35300106475765136
        + s2 * 937.4224935298015 * wm**0.0192950634264157 * wb**-0.04285000485853785
    )


@njit
def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""

    b, m = (0.99625075, 1.00593295)

    wb = wb**b
    wm = wm**m

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
def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""

    s1, s2, b, m = (1.00791144, 1.00585853, 1.05510863, 0.84044899)

    wb = wb**b
    wm = wm**m

    return (
        1 + s1 * 428.169 * wb**0.256459 * wm**0.616388 + s2 * 925.56 * wm**0.751615
    ) * wm**-0.714129


_HZ_FUNC = None


@njit
def _rs_integ(z, Obh2, params):
    Rb = (3 / 4) * (Obh2 / O_GAMMA_H2) / (1.0 + z)
    cs = c / np.sqrt(3 * (1.0 + Rb))
    Hz = _HZ_FUNC(z, params)
    return cs / Hz


def rs_z(Hz_fun, z_lim, Obh2, params):
    global _HZ_FUNC
    _HZ_FUNC = Hz_fun

    args = (Obh2, params)
    res = quad(_rs_integ, z_lim, np.inf, args=args)[0]
    _HZ_FUNC = None
    return res


@njit
def _DM_integ(z, params):
    return c / _HZ_FUNC(z, params)


def DM_z(Hz_fun, z_lim, params):
    global _HZ_FUNC
    _HZ_FUNC = Hz_fun

    res = quad(_DM_integ, 0.0, z_lim, args=(params,))[0]
    _HZ_FUNC = None
    return res


def cmb_distances(Hz_fun, Ob_h2, Oc_h2, params):
    """
    return (R, lA=π / θ*, ωb=Ωb*h^2)
    """
    Om_h2 = Oc_h2 + Ob_h2 + Omnu_h2
    zstar = z_star(Ob_h2, Om_h2)
    rs_star = rs_z(Hz_fun, zstar, Ob_h2, params)
    DM_star = DM_z(Hz_fun, zstar, params)

    R = 100 * np.sqrt(Om_h2) * DM_star / c
    lA = np.pi * DM_star / rs_star
    return np.array([R, lA, Ob_h2])
