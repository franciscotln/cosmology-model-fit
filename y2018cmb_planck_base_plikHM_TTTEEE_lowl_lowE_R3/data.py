"""
Planck PR3, 2019
plikHM TT, TE, EE + lowl + lowE
https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/
"""

import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0
from numba import njit
import nu_evolution as neutrino

c = c0 / 1000  # km/s

DISTANCE_PRIORS = np.array([1.0410875, 147.0515917], dtype=np.float64)
"""Compressed Planck priors: (100 θ*, rdrag)"""

covariance = np.array(
    [
        [9.29868316e-08, 2.01530071e-05],
        [2.01530071e-05, 8.79248605e-02],
    ]
)
inv_cov_mat = np.linalg.inv(covariance)

k_B = 8.617333262e-5  # eV/K
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4729753287140862e-05

N_EFF = 3.046
mnu_tot = 0.06  # total mass [eV]
T_nu0 = (4 / 11) ** (1 / 3) * (N_EFF / 3) ** (1 / 4) * TCMB  # K
T_nu0_eV = T_nu0 * k_B  # eV
m0 = mnu_tot / T_nu0_eV
Omnu_h2 = (mnu_tot / 94.07) * (N_EFF / 3) ** (3 / 4)


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

    s1, s2, b, m = (0.73491615, 1.00820929, 1.01709662, 1.17030559)

    wb = wb**b
    wm = wm**m
    a = s1 * 391.6723594873167
    b = s2 * 937.4224935298015

    return (
        wm**-0.7316314841257655
        + a * wb**0.9368102670600895 * wm**-0.35300106475765136
        + b * wm**0.0192950634264157 * wb**-0.04285000485853785
    )


@njit
def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""

    b, m = (1.00078696, 1.00128548)

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

    term_A_denominator = a1 * wb**a2 + a3 * wb**a4 * wm**a5 + a6 * wm**a7
    term_A = 1.0 / term_A_denominator
    term_B = a8 / (wm**a9)
    return term_A - term_B


@njit
def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""

    s1, s2, b, m = (1.00044649, 1.00006975, 1.00041899, 1.00135313)

    wb = wb**b
    wm = wm**m
    a = s1 * 428.169
    b = s2 * 925.56

    return wm**-0.714129 + a * wb**0.256459 * wm**-0.097741 + b * wm**0.037486


_HZ_FUNC = None


def set_HZ(Hz_fun):
    global _HZ_FUNC
    _HZ_FUNC = Hz_fun


@njit
def _DH(z, params):
    return c / _HZ_FUNC(z, params)


@njit
def _rs_integ(z, Obh2, params):
    Rb = (3 / 4) * (Obh2 / O_GAMMA_H2) / (1.0 + z)
    return _DH(z, params) / np.sqrt(3 * (1.0 + Rb))


def rs_z(z_lim, Obh2, params):
    args = (Obh2, params)
    return quad(_rs_integ, z_lim, np.inf, args=args)[0]


def DM_z(z_lim, params):
    return quad(_DH, 0.0, z_lim, args=(params,))[0]


def cmb_distances(Hz_fun, Ob_h2, Oc_h2, params):
    """
    returns (100 x θ*, rdrag)
    """
    Om_h2 = Oc_h2 + Ob_h2 + Omnu_h2
    rs_drag = r_drag(Ob_h2, Om_h2)
    zstar = z_star(Ob_h2, Om_h2)
    rs_star = rs_z(Hz_fun, zstar, Ob_h2, params)
    DM_star = DM_z(Hz_fun, zstar, params)
    thetastar = rs_star / DM_star
    return np.array([100 * thetastar, rs_drag], dtype=np.float64)
