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
L = np.linalg.cholesky(covariance)
logdet = 2 * np.sum(np.log(np.diag(L)))
prob_norm = logdet + len(DISTANCE_PRIORS) * np.log(2 * np.pi)

# ---- Physical constants ----
c_km_per_s = cmb_dist.C_KM_PER_S  # km/s
TCMB = cmb_dist.TCMB  # K
O_GAMMA_H2 = cmb_dist.O_GAMMA_H2
N_EFF = 3.044
MNU_TOT = 0.06  # total mass [eV]
NU_REL = 2.0308
# ----------------------------


def Omega_r_h2(n_eff):
    return O_GAMMA_H2 * (1 + n_eff * (7 / 8) * (4 / 11) ** (4 / 3))


Or_h2 = Omega_r_h2(NU_REL)

Omnu_h2, Omnu_z, w_nu_z = neutrino.create_neutrino_evolution(N_EFF, NU_REL, MNU_TOT, TCMB)


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
