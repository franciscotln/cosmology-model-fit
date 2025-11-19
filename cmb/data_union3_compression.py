from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# --- PLANCK PRIORS (Rubin+ arXiv:2311.12098v2) ---
# R ≡ √(Ωm H0²) * DA(z*) * (1 + z*) / c
# 100 θ* ≡ 100 rs(z*) / DM(z*)
# ωb = Ωb h²
DISTANCE_PRIORS = np.array(
    [1.7492768568335353, 1.039233410719115, 0.02239245], dtype=np.float64
)
inv_cov_mat = np.array(
    [
        [92701.58172970748, 348041.8137694254, 1613445.8550364415],
        [348041.8137694254, 13114681.644682042, -3019007.1687636944],
        [1613445.8550364415, -3019007.1687636944, 80842256.32398143],
    ],
    dtype=np.float64,
)
covariance = np.linalg.inv(inv_cov_mat)

N_EFF = 3.04
TCMB = 2.72548  # K
O_GAMMA_H2 = 2.4729e-5

T_nu0 = (4 / 11) ** (1 / 3) * TCMB  # K
T_nu0_eV = T_nu0 * 8.617333262e-5  #  1.67639e-04 eV
mnu_tot = 0.06  # total mass [eV]
Omnu_h2 = mnu_tot / 93.14  # present-day Omega_nu*h^2
z_nr = mnu_tot / (3.15 * T_nu0_eV)


@njit
def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


def rs_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Rb = 3 * Obh2 / (4 * O_GAMMA_H2)
    Obc = (Och2 + Obh2) / h**2

    def integrand(a):
        denom = a**2 * Ez_func(1 / a - 1, H0, Obc, w0, wa) * np.sqrt(3 * (1 + Rb * a))
        return 1 / denom

    return (c / H0) * quad(integrand, 1e-09, 1 / (1 + z_lim))[0]


def DM_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Obc = (Och2 + Obh2) / h**2
    integral = quad(lambda z: 1 / Ez_func(z, H0, Obc, w0, wa), 1e-8, z_lim)[0]
    return integral * c / H0


def cmb_distances(Ez_func, H0, Ob_h2, Oc_h2, w0=-1, wa=0):
    Om_h2 = Oc_h2 + Ob_h2 + Omnu_h2
    zstar = z_star(wb=Ob_h2, wm=Om_h2)
    rs_star = rs_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)
    DM_star = DM_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)

    R = 100 * np.sqrt(Om_h2) * DM_star / c
    theta_100 = 100 * rs_star / DM_star
    return np.array([R, theta_100, Ob_h2])


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
def z_star(wb, wm):
    """arXiv:astro-ph/9510117v2 (eq-1)"""
    SCALING_FACTOR = 0.9981

    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    factor_1 = 1 + 0.00124 * wb**-0.738
    factor_2 = 1 + g1 * wm**g2
    return SCALING_FACTOR * 1048 * factor_1 * factor_2


@njit
def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""
    return (
        1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615
    ) * wm**-0.714129
