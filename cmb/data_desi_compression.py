from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# --- PLANCK DISTANCE PRIORS (arXiv:2503.14738v2 Abdul Karim+) ---
# θ* ≡ rs(z*) / DM(z*)
DISTANCE_PRIORS = np.array(
    [
        0.01041,  # θ*
        0.02223,  # ωb
        0.14208,  # ωm
    ],
    dtype=np.float64,
)
covariance = 10**-9 * np.array(
    [
        [0.006621, 0.12444, -1.1929],
        [0.12444, 21.344, -94.001],
        [-1.1929, -94.001, 1488.4],
    ],
    dtype=np.float64,
)
inv_cov_mat = np.linalg.inv(covariance)

N_EFF = 3.044
TCMB = 2.7255  # K
O_GAMMA_H2 = (0.75 / 31500) * (TCMB / 2.7) ** 4


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
    DA_star = DA_z(Ez_func, zstar, params, H0)
    theta = rs_star / ((1 + zstar) * DA_star)
    return np.array([theta, Ob_h2, Om_h2])


@njit
def r_drag(wb, wm, n_eff=N_EFF):  # arXiv:2503.14738v2 (eq 2)
    return (
        147.05 * (0.02236 / wb) ** 0.13 * (0.1432 / wm) ** 0.23 * (3.04 / n_eff) ** 0.1
    )


@njit
def z_star(wb, wm):
    # arXiv:2106.00428v2 (eq A4)
    return (391.672 * wm ** (-0.372296) + 937.422 * wb ** (-0.97966)) / (
        wm ** (-0.0192951) * wb ** (-0.93681)
    ) + wm ** (-0.731631)


@njit
def z_star_HU(wb, wm):
    # arXiv:astro-ph/9510117v2 (eq-1)
    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    factor_1 = 1 + 0.00124 * wb**-0.738
    factor_2 = 1 + g1 * wm**g2
    return 1048 * factor_1 * factor_2


@njit
def z_drag_HU(wb, wm):
    # arXiv:astro-ph/9510117v2 (eq-2)
    b1 = 0.313 * (wm**-0.419) * (1 + 0.607 * (wm**0.674))
    b2 = 0.238 * (wm**0.223)

    numerator_factor = 1345 * (wm**0.251)
    denominator = 1 + 0.659 * (wm**0.828)
    correction_factor = 1 + b1 * (wb**b2)

    return (numerator_factor / denominator) * correction_factor


@njit
def z_drag(wb, wm):
    # arXiv:2106.00428v2 (eq A2)
    return (
        1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615
    ) * wm**-0.714129
