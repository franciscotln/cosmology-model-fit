"""
SPT+ACT+Planck LCDM constraints arXiv:2506.20707v2
https://lambda.gsfc.nasa.gov/product/spt/spt3g_d1_bandp_liklyhood_info.html
https://lambda.gsfc.nasa.gov/product/spt/spt3g_d1_bandp_liklyhood_get.html
"""

import numpy as np
from scipy.constants import c as c0
from numba import njit
import nu_evolution as neutrino
from solve_triangular import solve_triangular

c_km_per_s = c0 / 1000  # km/s

DISTANCE_PRIORS = np.array([1.04161, 0.0223985, 0.14331818])
"""Compressed SPT+ACT+Planck priors: (100 θ*, ωb, ωm)"""

covariance = np.array([
    [ 5.20526927e-08,  1.19666450e-09, -2.79849344e-08],
    [ 1.19666450e-09,  8.96589845e-09, -1.72830047e-08],
    [-2.79849344e-08, -1.72830047e-08,  8.37535794e-07]
])
inv_cov_mat = np.linalg.inv(covariance)
L = np.linalg.cholesky(covariance)
logdet = 2 * np.sum(np.log(np.diag(L)))
prob_norm = logdet + len(DISTANCE_PRIORS) * np.log(2 * np.pi)

# ---- Physical constants ----
k_B = 8.617333262e-5  # eV/K
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.472975328714087e-05

N_EFF = 3.044
T_nu0 = (4 / 11) ** (1 / 3) * (N_EFF / 3) ** (1 / 4) * TCMB  # K
T_nu0_eV = T_nu0 * k_B  # eV
mnu_tot = 0.06  # total mass [eV]
m0 = mnu_tot / T_nu0_eV
nu_rel = 2.0308
nu_nr = N_EFF - nu_rel
Omnu_h2 = mnu_tot / (94.0641 / nu_nr ** 0.75)


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + Neff * (7 / 8) * (4 / 11) ** (4 / 3))


Or_h2 = Omega_r_h2(nu_rel)


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

    first_f = np.sqrt(qs_sq[0] + mz_sq)
    weighted_sum = ws[0] * first_f

    for i in range(1, len(qs_sq)):
        weighted_sum += ws[i] * np.sqrt(qs_sq[i] + mz_sq)

    return zp1**4 * weighted_sum / rho0


@njit
def w_nu_z(z):
    """
    Equation of state w(z) for massive neutrinos using the 5-node approximation
    """
    zp1 = 1.0 + z
    mz_sq = (m0 / zp1) ** 2

    first_f = np.sqrt(qs_sq[0] + mz_sq)
    numerator = ws[0] / first_f
    denominator = ws[0] * first_f

    for i in range(1, len(qs_sq)):
        f = np.sqrt(qs_sq[i] + mz_sq)
        numerator += ws[i] / f
        denominator += ws[i] * f

    return (1 / 3) - (1 / 3) * mz_sq * numerator / denominator


@njit
def z_star(wb, wm):
    # for SPA this is actually z_rec, peak visibility function
    """arXiv:2106.00428v2 (eq A-4)"""
    s1, s2, b, m = (1.01659306, 0.99938819, 1.00488811, 1.01260252)

    wb_eff = wb**b
    wm_eff = wm**m

    return (
        wm_eff**-0.7316314841257655
        + s1 * 391.6723594873167 * wb_eff**0.9368102670600895 * wm_eff**-0.35300106475765136
        + s2 * 937.4224935298015 * wm_eff**0.0192950634264157 * wb_eff**-0.04285000485853785
    )


@njit
def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""
    b, m = 1.01063661, 0.99183967

    wb_eff = wb**b
    wm_eff = wm**m

    a1 = 0.00257366
    a2 = 0.05032
    a3 = 0.013
    a4 = 0.7720642
    a5 = 0.24346362
    a6 = 0.00641072
    a7 = 0.5350899
    a8 = 32.7525
    a9 = 0.315473

    term_A_denominator = (a1 * (wb_eff**a2)) + (a3 * (wb_eff**a4) * (wm_eff**a5)) + (a6 * (wm_eff**a7))
    term_A = 1.0 / term_A_denominator
    term_B = a8 / (wm_eff**a9)
    return term_A - term_B


@njit
def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""
    s1, s2, b, m = (1.0003704, 0.99986385, 0.99926332, 1.00361697)

    wb_eff = wb**b
    wm_eff = wm**m

    return (
        1 + s1 * 428.169 * wb_eff**0.256459 * wm_eff**0.616388 + s2 * 925.56 * wm_eff**0.751615
    ) * wm_eff**-0.714129


_HZ_FUNC = None


def set_HZ(Hz_fun):
    global _HZ_FUNC
    _HZ_FUNC = Hz_fun


N_DM = 30
N_RS = 15
GL_X_DM, GL_W_DM = np.polynomial.legendre.leggauss(N_DM)
GL_X_RS, GL_W_RS = np.polynomial.legendre.leggauss(N_RS)
# change integration variable
# a = u^2, da = 2 * u * du


@njit
def _integ_u(u, params):
    # u = sqrt(a): constant in matter era, ~linear in radiation era
    z = 1.0 / (u * u) - 1.0
    return 2.0 * c_km_per_s / (u**3 * _HZ_FUNC(z, params))


@njit
def DM_z(z_lim, params):
    u_lo = 1.0 / np.sqrt(1.0 + z_lim)
    half = 0.5 * (1.0 - u_lo)
    mid = 0.5 * (1.0 + u_lo)
    s = 0.0
    for i in range(N_DM):
        s += GL_W_DM[i] * _integ_u(half * GL_X_DM[i] + mid, params)
    return half * s


@njit
def rs_z(z_lim, Obh2, params):
    half = 0.5 / np.sqrt(1.0 + z_lim)
    k = 0.75 * Obh2 / O_GAMMA_H2
    s = 0.0
    for i in range(N_RS):
        u = half * (GL_X_RS[i] + 1.0)
        s += GL_W_RS[i] * _integ_u(u, params) / np.sqrt(3.0 * (1.0 + k * u * u))
    return half * s


@njit
def cmb_distances(Obh2, Och2, params):
    """
    return (100 θ*, ωb h^2, ωm h^2)
    """
    Omh2 = Och2 + Obh2 + Omnu_h2
    zstar = z_star(Obh2, Omh2)
    thetastar = rs_z(zstar, Obh2, params) / DM_z(zstar, params)
    return np.array([100 * thetastar, Obh2, Omh2])


@njit
def chi2(Obh2, Och2, params):
    delta = DISTANCE_PRIORS - cmb_distances(Obh2, Och2, params)
    y = solve_triangular(L, delta)
    return np.dot(y, y)


@njit
def log_likelihood(Obh2, Och2, params):
    return -0.5 * (chi2(Obh2, Och2, params) + prob_norm)
