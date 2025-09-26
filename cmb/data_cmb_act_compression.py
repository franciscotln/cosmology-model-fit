from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# --- PLANCK DISTANCE PRIORS (Prakhar Bansal+2025 arXiv:2502.07185v2) ---
DISTANCE_PRIORS = np.array(
    [
        1.7504,  # R
        301.77,  # lA
        0.022371,  # Ωb h^2
    ],
    dtype=np.float64,
)
cov_matrix = 10**-8 * np.array(
    [
        [1559.83, -1325.41, -36.45],
        [-1325.41, 714691.80, 269.77],
        [-36.45, 269.77, 2.10],
    ],
    dtype=np.float64,
)
inv_cov_mat = np.linalg.inv(cov_matrix)
N_EFF = 3.046
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4728e-5 * (TCMB / 2.7255) ** 4


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


@njit
def z_star(wb, wm):
    # arXiv:2106.00428v2 (eq A4)
    return wm**-0.731631 + (
        (391.672 * wm**-0.372296 + 937.422 * wb**-0.97966) * wm**0.0192951 * wb**0.93681
    )


@njit
def z_drag(wb, wm):
    # arXiv:2106.00428v2 (eq A2)
    return (
        1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615
    ) * wm**-0.714129


def rs_z(Ez_func, z, params, H0, Ob_h2):
    Rb = 3 * Ob_h2 / (4 * O_GAMMA_H2)

    def integrand(zp):
        denom = Ez_func(zp, params) * np.sqrt(3 * (1 + Rb / (1 + zp)))
        return 1 / denom

    return c * quad(integrand, z, np.inf, limit=100)[0] / H0


@njit
def DA_z(Ez_func, z, params):
    zp = np.linspace(0.0, z, 20_000)
    I = np.trapz(y=1.0 / Ez_func(zp, params), x=zp)
    return I / (1.0 + z)


def cmb_distances(Ez_func, params, H0, Om, Ob_h2):
    zstar = z_star(wb=Ob_h2, wm=Om * (H0 / 100) ** 2)
    rs_star = rs_z(Ez_func, zstar, params, H0, Ob_h2)
    DA_star = c * DA_z(Ez_func, zstar, params) / H0

    R = np.sqrt(Om) * H0 * (1 + zstar) * DA_star / c
    lA = (1 + zstar) * np.pi * DA_star / rs_star
    return np.array([R, lA, Ob_h2])


@njit
def r_drag(wb, wm):
    # arXiv:2106.00428v2 (eq 6)
    numerator = 45.5337 * np.log(7.20376 / wm)
    denominator = np.sqrt(1 + 9.98592 * (wb**0.801347))
    return numerator / denominator
