"""
CMB Constraints on the Early Universe Independent of Late-Time Cosmology
arXiv:2302.12911
"""

from numba import njit
import numpy as np
from scipy.constants import c as c0
import nu_evolution as neutrino
from solve_triangular import solve_triangular
import rec_planck as rec

z_star = rec.z_star
z_drag = rec.z_drag
r_drag = rec.r_drag

c_km_per_s = c0 / 1000  # km/s

DISTANCE_PRIORS = np.array([1.04102739, 0.02223208, 0.14207901])
"""Compressed early-LCDM priors: (100 θ*, ωb, ωm)"""

covariance = np.array([
    [6.62099420e-08, 1.24442058e-08, -1.19287532e-07],
    [1.24442058e-08, 2.13441666e-08, -9.40008323e-08],
    [-1.19287532e-07, -9.40008323e-08, 1.48841714e-06],
])
inv_cov_mat = np.linalg.inv(covariance)
L = np.linalg.cholesky(covariance)
logdet = 2 * np.sum(np.log(np.diag(L)))
prob_norm = logdet + len(DISTANCE_PRIORS) * np.log(2 * np.pi)

# ---- Physical constants ----
k_B = 8.617333262e-5  # eV/K
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.472975328714087e-05

N_EFF = 3.046
T_nu0 = (4 / 11) ** (1 / 3) * (N_EFF / 3) ** (1 / 4) * TCMB  # K
T_nu0_eV = T_nu0 * k_B  # eV
mnu_tot = 0.06  # total mass [eV]
m0 = mnu_tot / T_nu0_eV
Omnu_h2 = mnu_tot / (94.07 / (N_EFF / 3.0) ** 0.75)


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
