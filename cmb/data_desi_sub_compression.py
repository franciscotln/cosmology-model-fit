from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0
import numdifftools as nd

c = c0 / 1000  # km/s

# --- PLANCK PRIORS (arXiv:2503.14738v2 Abdul Karim+) ---
_DISTANCE_PRIORS_ = np.array(
    [
        0.01041,  # θ* ≡ rs(z*) / DM(z*)
        0.02223,  # ωb
        0.14208,  # ωm
    ],
    dtype=np.float64,
)
_covariance_init_ = 10**-9 * np.array(
    [
        [0.006621, 0.12444, -1.1929],
        [0.12444, 21.344, -94.001],
        [-1.1929, -94.001, 1488.4],
    ],
    dtype=np.float64,
)

N_EFF = 3.044
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4729e-05


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


def rs_z(Ez_func, z, params, H0, Ob_h2):
    Rb = 3 * Ob_h2 / (4 * O_GAMMA_H2)

    def integrand(a):
        denom = a**2 * Ez_func(1 / a - 1, params) * np.sqrt(3 * (1 + Rb * a))
        return c / denom

    return quad(integrand, 0, 1 / (1 + z))[0] / H0


def DA_z(Ez_func, z, params, H0):
    I = quad(lambda zp: c / Ez_func(zp, params), 0, z)[0]
    return (I / H0) / (1.0 + z)


def cmb_distances(Ez_func, params, H0, Om, Ob_h2):
    Om_h2 = Om * (H0 / 100) ** 2
    zstar = z_star(wb=Ob_h2, wm=Om_h2)
    rs_star = rs_z(Ez_func, zstar, params, H0, Ob_h2)
    DM_star = (1 + zstar) * DA_z(Ez_func, zstar, params, H0)

    theta = rs_star / DM_star
    rd = r_drag(wb=Ob_h2, wm=Om_h2)
    return np.array([100 * theta, rd], dtype=np.float64)


@njit
def z_star(wb, wm):
    """arXiv:2106.00428v2 (eq A4)"""
    return (391.672 * wm ** (-0.372296) + 937.422 * wb ** (-0.97966)) / (
        wm ** (-0.0192951) * wb ** (-0.93681)
    ) + wm ** (-0.731631)


@njit
def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""
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
    return (
        1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615
    ) * wm**-0.714129


# ----- Transform Planck's priors (θ*, ωb, ωm) into (100 x θ*, r_d) space -----
def _transform_(x):
    theta_star, wb, wm = x
    return np.array([100 * theta_star, r_drag(wb=wb, wm=wm)], dtype=np.float64)


J = nd.Jacobian(_transform_)(_DISTANCE_PRIORS_)

DISTANCE_PRIORS = _transform_(_DISTANCE_PRIORS_)  # [100 x θ*, rd]
covariance = J @ _covariance_init_ @ J.T
inv_cov_mat = np.linalg.inv(covariance)


# ----- Estimate correlation matrix for Planck's direct measurements -----
_std_devs_ = np.sqrt(np.diag(covariance))
_correlation_matrix_ = covariance / np.outer(_std_devs_, _std_devs_)
_estimated_corr_factor_ = _correlation_matrix_[0, 1]

# Planck's direct measurements with estimated correlation ρ = 0.2931563
_sigma_theta_ = 0.00031
_sigma_rd_ = 0.26

DISTANCE_PRIORS_PLANCK = np.array([1.04110, 147.09], dtype=np.float64)  # [100 x θ*, rd]

covariance_planck = np.array(
    [
        [_sigma_theta_**2, _estimated_corr_factor_ * _sigma_theta_ * _sigma_rd_],
        [_estimated_corr_factor_ * _sigma_theta_ * _sigma_rd_, _sigma_rd_**2],
    ],
    dtype=np.float64,
)

inv_cov_mat_planck = np.linalg.inv(covariance_planck)
