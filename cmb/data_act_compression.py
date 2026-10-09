"""
ACT baseline LCDM constraints arXiv:2503.14452v2
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_lcdm_get.html
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_info.html
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_prod_table.html
"""

import numpy as np
from numba import njit
import nu_evolution as neutrino
from solve_triangular import solve_triangular
import rec_cosmorec as rec
import cmb.cmb_distances as cmb_dist

z_star = rec.z_star
z_drag = rec.z_drag
r_drag = rec.r_drag
rs_z = cmb_dist.rs_z
DM_z = cmb_dist.DM_z

DISTANCE_PRIORS = np.array([1.76114018, 301.858188, 0.0225906400])
"""Compressed ACT DR6 priors: (R, lA = π / θ*, ωb)"""

covariance = np.array([
    [4.21173357e-05, 2.72141593e-04, -1.81499538e-07],
    [2.72141593e-04, 8.16733306e-03, 2.41363324e-07],
    [-1.81499538e-07, 2.41363324e-07, 2.81508052e-08],
])
L = np.linalg.cholesky(covariance)
logdet = 2 * np.sum(np.log(np.diag(L)))
prob_norm = logdet + len(DISTANCE_PRIORS) * np.log(2 * np.pi)

# ---- Physical constants ----
c_km_per_s = cmb_dist.C_KM_PER_S
TCMB = cmb_dist.TCMB  # K
O_GAMMA_H2 = cmb_dist.O_GAMMA_H2
N_EFF = 3.044
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
    return (R, lA = π / θ*, ωb = Ωb*h^2)
    """
    Om_h2 = Oc_h2 + Ob_h2 + Omnu_h2
    zstar = z_star(Ob_h2, Om_h2)
    rs_star = rs_z(zstar, Ob_h2, params)
    DM_star = DM_z(zstar, params)

    R = 100 * np.sqrt(Om_h2) * DM_star / c_km_per_s
    lA = np.pi * DM_star / rs_star
    return np.array([R, lA, Ob_h2])


@njit
def chi2(Obh2, Och2, params):
    delta = DISTANCE_PRIORS - cmb_distances(Obh2, Och2, params)
    y = solve_triangular(L, delta)
    return np.dot(y, y)


@njit
def log_likelihood(Obh2, Och2, params):
    return -0.5 * (chi2(Obh2, Och2, params) + prob_norm)
