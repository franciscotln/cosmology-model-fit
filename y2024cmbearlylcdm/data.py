"""
https://github.com/cmbant/PlanckEarlyLCDM/blob/main/README.md
arXiv:2302.12911 - CMB Constraints on the Early Universe Independent of Late-Time Cosmology
"""

from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0


# from getdist import loadMCSamples


# samples = loadMCSamples(
#     "y2024cmbearlylcdm/raw/spline_planck_PR4_TTTEEE_lowE_lensing_ISW",
#     settings={"ignore_rows": 0.3},
# )
# samples.addDerived(samples.getParams().thetastar / 100, "thetastar_unscaled")

# samples.mean(["thetastar", "rdrag"])
# samples.cov(["thetastar", "rdrag"])

DISTANCE_PRIORS = np.array([1.04103, 147.46], dtype=np.float64)
covariance = 1e-05 * np.array(
    [
        [0.00662099420, 2.10838540],
        [2.10838540, 7798.46644],
    ],
    dtype=np.float64,
)
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
    zdrag = z_drag(wb=Ob_h2, wm=Om_h2)
    rs_drag = rs_z(Ez_func, zdrag, params, H0, Ob_h2)
    rs_star = rs_z(Ez_func, zstar, params, H0, Ob_h2)
    DM_star = (1 + zstar) * DA_z(Ez_func, zstar, params, H0)
    theta = rs_star / DM_star
    return np.array([100 * theta, rs_drag], dtype=np.float64)


@njit
def z_star(wb, wm):
    """arXiv:2106.00428v2 (eq A4)"""
    SCALING_FID = 0.99993  # z_star_fid / z_star(wb_fid, wm_fid)

    return SCALING_FID * (391.672 * wm ** (-0.372296) + 937.422 * wb ** (-0.97966)) / (
        wm ** (-0.0192951) * wb ** (-0.93681)
    ) + wm ** (-0.731631)


@njit
def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""
    SCALING_FID = 0.9985  # z_drag_fid / z_drag(wb_fid, wm_fid)

    return (
        SCALING_FID
        * (1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615)
        * wm**-0.714129
    )


@njit
def z_star_HU(wb, wm):
    """arXiv:astro-ph/9510117v2 (eq-1)"""
    SCALING_FID = 0.9971  # z_star_fid / z_star_HU(wb_fid, wm_fid)

    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    factor_1 = 1 + 0.00124 * wb**-0.738
    factor_2 = 1 + g1 * wm**g2
    return SCALING_FID * 1048 * factor_1 * factor_2


@njit
def z_drag_HU(wb, wm):
    """arXiv:astro-ph/9510117v2 (eq-2)"""
    SCALING_FID = 0.99521  # z_drag_fid / z_drag_HU(wb_fid, wm_fid)

    b1 = 0.313 * wm**-0.419 * (1 + 0.607 * wm**0.674)
    b2 = 0.238 * wm**0.223
    return SCALING_FID * 1345 * wm**0.251 * (1 + b1 * wb**b2) / (1 + 0.659 * wm**0.828)


wb_fid = 0.02223  # samples.mean("ombh2")
wm_fid = 0.14208  # samples.mean("omegamh2")
rd_fid = 147.46  # samples.mean("rdrag")
H0_fid = 67.49  # samples.mean("H0")
theta_star_fid = 1.04103  # samples.mean("thetastar")
z_drag_fid = 1057.91  # from integral to achieve rd_fid in ΛCDM
z_star_fid = 1088.86  # from integral to achieve theta_star_fid in ΛCDM


@njit
def r_drag(wb, wm, n_eff=N_EFF):
    """arXiv:2212.04522v2 (eq 3.4)"""
    return (
        rd_fid * (wb_fid / wb) ** 0.13 * (wm_fid / wm) ** 0.23 * (N_EFF / n_eff) ** 0.1
    )


@njit
def r_drag1(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""
    SCALING_FID = 1.001  # rd_fid / r_drag1(wb_fid, wm_fid)

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
    return SCALING_FID * (term_A - term_B)
