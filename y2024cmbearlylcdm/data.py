"""
https://github.com/cmbant/PlanckEarlyLCDM/blob/main/README.md
arXiv:2302.12911 - CMB Constraints on the Early Universe Independent of Late-Time Cosmology
"""

from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0
from getdist import loadMCSamples


samples = loadMCSamples(
    "y2024cmbearlylcdm/raw/spline_planck_PR4_TTTEEE_lowE_lensing_ISW",
    settings={"ignore_rows": 0.3},
)

DISTANCE_PRIORS = samples.mean(["thetastar", "rdrag"])
covariance = samples.cov(["thetastar", "rdrag"])
inv_cov_mat = np.linalg.inv(covariance)


c = c0 / 1000  # km/s

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
    rs_drag = r_drag(Ob_h2, Om_h2)
    rs_star = rs_z(Ez_func, zstar, params, H0, Ob_h2)
    DM_star = (1 + zstar) * DA_z(Ez_func, zstar, params, H0)
    theta = rs_star / DM_star
    return np.array([100 * theta, rs_drag], dtype=np.float64)


@njit
def z_star(wb, wm):
    """arXiv:2106.00428v2 (eq A4)"""
    return (391.672 * wm ** (-0.372296) + 937.422 * wb ** (-0.97966)) / (
        wm ** (-0.0192951) * wb ** (-0.93681)
    ) + wm ** (-0.731631)


@njit
def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""
    return (
        1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615
    ) * wm**-0.714129


rd_fid = samples.mean("rdrag")
wb_fid = samples.mean("ombh2")
wm_fid = samples.mean("omegamh2")


@njit
def r_drag(wb, wm, n_eff=N_EFF):
    """arXiv:2212.04522v2 (eq 3.4)"""
    return (
        rd_fid * (wb_fid / wb) ** 0.13 * (wm_fid / wm) ** 0.23 * (3.044 / n_eff) ** 0.1
    )
