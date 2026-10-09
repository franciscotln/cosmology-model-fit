"""
CMB Constraints on the Early Universe Independent of Late-Time Cosmology
arXiv:2302.12911
"""

from numba import njit
import numpy as np
import nu_evolution as neutrino
from solve_triangular import solve_triangular
import rec_planck as rec
import cmb.cmb_distances as cmb_dist

z_star = rec.z_star
z_drag = rec.z_drag
r_drag = rec.r_drag
rs_z = cmb_dist.rs_z
DM_z = cmb_dist.DM_z

DISTANCE_PRIORS = np.array([1.04102739, 0.02223208, 0.14207901])
"""Compressed early-LCDM priors: (100 θ*, ωb, ωm)"""

covariance = np.array([
    [6.62099420e-08, 1.24442058e-08, -1.19287532e-07],
    [1.24442058e-08, 2.13441666e-08, -9.40008323e-08],
    [-1.19287532e-07, -9.40008323e-08, 1.48841714e-06],
])
L = np.linalg.cholesky(covariance)
logdet = 2 * np.sum(np.log(np.diag(L)))
prob_norm = logdet + len(DISTANCE_PRIORS) * np.log(2 * np.pi)

# ---- Physical constants ----
c_km_per_s = cmb_dist.C_KM_PER_S
TCMB = cmb_dist.TCMB  # K
O_GAMMA_H2 = cmb_dist.O_GAMMA_H2
N_EFF = 3.046
MNU_TOT = 0.06  # total mass [eV]
NU_REL = 2 * N_EFF / 3
# ----------------------------


def Omega_r_h2(n_eff):
    return O_GAMMA_H2 * (1 + n_eff * (7 / 8) * (4 / 11) ** (4 / 3))


Or_h2 = Omega_r_h2(NU_REL)

Omnu_h2, Omnu_z, w_nu_z = neutrino.create_neutrino_evolution(N_EFF, NU_REL, MNU_TOT, TCMB)


def set_HZ(Hz_fun):
    cmb_dist.set_HZ(Hz_fun)


@njit
def cmb_distances(Ob_h2, Oc_h2, params):
    """
    returns (100 θ*, ωb, ωm)
    """
    Om_h2 = Oc_h2 + Ob_h2 + Omnu_h2
    zstar = z_star(Ob_h2, Om_h2)
    rs_star = rs_z(zstar, Ob_h2, params)
    DM_star = DM_z(zstar, params)
    thetastar = rs_star / DM_star
    return np.array([100 * thetastar, Ob_h2, Om_h2])


@njit
def chi2(Obh2, Och2, params):
    delta = DISTANCE_PRIORS - cmb_distances(Obh2, Och2, params)
    y = solve_triangular(L, delta)
    return np.dot(y, y)


@njit
def log_likelihood(Obh2, Och2, params):
    return -0.5 * (chi2(Obh2, Och2, params) + prob_norm)
