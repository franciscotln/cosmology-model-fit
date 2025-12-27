"""
CMB Constraints on the Early Universe Independent of Late-Time Cosmology
arXiv:2302.12911 https://github.com/cmbant/PlanckEarlyLCDM/blob/main/README.md
"""

from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0


DISTANCE_PRIORS = np.array([1.0410274, 147.46], dtype=np.float64)
covariance = 1e-05 * np.array(
    [
        [0.00662099420, 2.10838540],
        [2.10838540, 7798.46644],
    ],
    dtype=np.float64,
)
inv_cov_mat = np.linalg.inv(covariance)


c = c0 / 1000  # km/s

k_B = 8.617333262e-5  # eV/K
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.472975328714087e-05

N_EFF = 3.044
T_nu0 = (4 / 11) ** (1 / 3) * (N_EFF / 3) ** (1 / 4) * TCMB  # K
T_nu0_eV = T_nu0 * k_B  # eV
mnu_tot = 0.06  # total mass [eV]
Omnu_h2 = mnu_tot / (94.07 / (N_EFF / 3.0) ** 0.75)


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + Neff * (7 / 8) * (4 / 11) ** (4 / 3))


Or_h2 = Omega_r_h2(N_EFF - (N_EFF / 3))


@njit
def Omnu_z(z):
    """
    Computes the appox. evolution of massive neutrino
    energy density with redshift
    """
    zp1 = 1 + z
    p = 1.8361771
    base = -9.2941 * N_EFF + 141.4673
    ratio = (1 + (base / zp1) ** p) / (1 + base**p)
    return zp1**4 * ratio ** (1 / p)


def rs_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    Rb = 3 * Obh2 / (4 * O_GAMMA_H2)

    def integrand(a):
        Ez = Ez_func(1 / a - 1, H0, Obh2, Och2, w0, wa)
        denom = a**2 * Ez * np.sqrt(3 * (1 + Rb * a))
        return 1 / denom

    return (c / H0) * quad(integrand, 1e-09, 1 / (1 + z_lim))[0]


def DM_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    integral = quad(lambda z: 1 / Ez_func(z, H0, Obh2, Och2, w0, wa), 0.0, z_lim)[0]
    return integral * c / H0


def cmb_distances(Ez_func, H0, Ob_h2, Oc_h2, w0=-1, wa=0):
    Om_h2 = Oc_h2 + Ob_h2 + Omnu_h2
    rs_drag = r_drag(Ob_h2, Om_h2)
    zstar = z_star(Ob_h2, Om_h2)

    rs_star = rs_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)
    DM_star = DM_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)
    thetastar = rs_star / DM_star
    return np.array([100 * thetastar, rs_drag], dtype=np.float64)


@njit
def z_star(wb, wm):
    """arXiv:2106.00428v2 (eq A-4)"""

    s0, s1, s2, b, m = (0.85852057, 0.85543412, 1.00112447, 1.03168648, 0.98880959)

    wb = wb**b
    wm = wm**m

    return (
        s0 * wm**-0.7316314841257655
        + s1 * 391.6723594873167 * wb**0.9368102670600895 * wm**-0.35300106475765136
        + s2 * 937.4224935298015 * wm**0.0192950634264157 * wb**-0.04285000485853785
    )


@njit
def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""

    b, m = (1.00137869, 1.0007536)

    wb = wb**b
    wm = wm**m

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
    s0, s1, s2, b, m = (0.99882683, 1.00371723, 1.00155821, 1.00215819, 1.01377142)

    wb = wb**b
    wm = wm**m

    return (
        s0
        * (1 + s1 * 428.169 * wb**0.256459 * wm**0.616388 + s2 * 925.56 * wm**0.751615)
        * wm**-0.714129
    )
