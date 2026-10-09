"""
https://pla.esac.esa.int/#cosmology
Baseline LCDM chains with baseline likelihoods
Planck PR3, 2018 plikHM TT, TE, EE + lowl + lowE
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

DISTANCE_PRIORS = np.array([1.75063846, 301.760701, 0.0223597502])
"""Compressed Planck priors: (R, lA = π / θ*, ωb)"""

covariance = np.array([
    [2.09107356e-05, 1.78419597e-04, -4.46283183e-07],
    [1.78419597e-04, 7.81249750e-03, -4.24834772e-06],
    [-4.46283183e-07, -4.24834772e-06, 2.21402189e-08],
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
