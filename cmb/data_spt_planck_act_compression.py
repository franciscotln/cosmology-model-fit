"""
SPT+ACT+Planck LCDM constraints arXiv:2506.20707v2
https://lambda.gsfc.nasa.gov/product/spt/spt3g_d1_bandp_liklyhood_info.html
https://lambda.gsfc.nasa.gov/product/spt/spt3g_d1_bandp_liklyhood_get.html
"""

import numpy as np
from numba import njit
import nu_evolution as neutrino
from solve_triangular import solve_triangular
import rec_hyrec as rec
import cmb.cmb_distances as cmb_dist

z_star = rec.z_star
z_drag = rec.z_drag
r_drag = rec.r_drag
rs_z = cmb_dist.rs_z
DM_z = cmb_dist.DM_z

DISTANCE_PRIORS = np.array([1.04161, 0.0223985, 0.14331818])
"""Compressed SPT+ACT+Planck priors: (100 θ*, ωb, ωm)"""

covariance = np.array([
    [5.20526927e-08, 1.19666450e-09, -2.79849344e-08],
    [1.19666450e-09, 8.96589845e-09, -1.72830047e-08],
    [-2.79849344e-08, -1.72830047e-08, 8.37535794e-07]
])
inv_cov_mat = np.linalg.inv(covariance)
L = np.linalg.cholesky(covariance)
logdet = 2 * np.sum(np.log(np.diag(L)))
prob_norm = logdet + len(DISTANCE_PRIORS) * np.log(2 * np.pi)

# ---- Physical constants ----
c_km_per_s = cmb_dist.C_KM_PER_S  # km/s
k_B = cmb_dist.K_BOLTZ  # eV/K
TCMB = cmb_dist.TCMB  # K
O_GAMMA_H2 = cmb_dist.O_GAMMA_H2
# ----------------------------

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


def set_HZ(Hz_fun):
    cmb_dist.set_HZ(Hz_fun)


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
