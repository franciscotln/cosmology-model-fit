"""
CMB Constraints on the Early Universe Independent of Late-Time Cosmology
arXiv:2302.12911
"""

from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# θ*, Obh2, Omh2
DISTANCE_PRIORS = np.array([0.010410274, 0.02223, 0.14208], dtype=np.float64)
covariance = 1e-9 * np.array(
    [
        [0.00662099420, 0.124442058, -1.19287532],
        [0.124442058, 21.3441666, -94.0008323],
        [-1.19287532, -94.0008323, 1488.41714],
    ],
    dtype=np.float64,
)
inv_cov_mat = np.linalg.inv(covariance)

k_B = 8.617333262e-5  # eV/K
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.472975328714087e-05

N_EFF = 3.044
T_nu0 = (4 / 11) ** (1 / 3) * (N_EFF / 3) ** (1 / 4) * TCMB  # K
T_nu0_eV = T_nu0 * k_B  # eV
mnu_tot = 0.06  # total mass [eV]
m0 = mnu_tot / T_nu0_eV
Omnu_h2 = mnu_tot / (94.07 / (N_EFF / 3.0) ** 0.75)  # present-day Omega_nu*h^2


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + Neff * (7 / 8) * (4 / 11) ** (4 / 3))


Or_h2 = Omega_r_h2(N_EFF - (N_EFF / 3))


# 1 massive neutrino section
def compute_B1(m0):
    return m0**0.99877359 / 1.13497121


def compute_B2(m0):
    return m0**0.99877359 / 3.23490176


def compute_B3(m0):
    return m0**0.99877359 / 7.13084298


def compute_W1(m0):
    return (m0 / 1000) ** 0.00377374 / 10.20460242


def compute_W2(m0):
    return (m0 / 1000) ** 0.00109726 / 1.57582916


@njit
def fluid_component(B, z):
    Bz = B / (1.0 + z)
    return np.sqrt(1 + Bz**2)


B1 = compute_B1(m0)
B2 = compute_B2(m0)
B3 = compute_B3(m0)
W1 = compute_W1(m0)
W2 = compute_W2(m0)
W3 = 1.0 - W1 - W2
f1_0 = fluid_component(B1, 0)
f2_0 = fluid_component(B2, 0)
f3_0 = fluid_component(B3, 0)
normalization = W1 * f1_0 + W2 * f2_0 + W3 * f3_0


@njit
def Omnu_z(z):
    """
    3-fluid energy density rho(z) for massive neutrinos
    """
    zp1 = 1.0 + z
    density1 = W1 * fluid_component(B1, z)
    density2 = W2 * fluid_component(B2, z)
    density3 = W3 * fluid_component(B3, z)
    return zp1**4 * (density1 + density2 + density3) / normalization


def rs_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    Rb = 3 * Obh2 / (4 * O_GAMMA_H2)

    def integrand(a):
        Ez = Ez_func(1 / a - 1, H0, Obh2, Och2, w0, wa)
        denom = a**2 * Ez * np.sqrt(3 * (1 + Rb * a))
        return 1 / denom

    return (c / H0) * quad(integrand, 1e-08, 1 / (1 + z_lim))[0]


def DM_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    integral = quad(lambda z: 1 / Ez_func(z, H0, Obh2, Och2, w0, wa), 0.0, z_lim)[0]
    return integral * c / H0


def cmb_distances(Ez_func, H0, Ob_h2, Oc_h2, w0=-1, wa=0):
    """
    returns (θ*, ωb, ωm)
    """
    Om_h2 = Oc_h2 + Ob_h2 + Omnu_h2
    zstar = z_star(wb=Ob_h2, wm=Om_h2)
    rs_star = rs_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)
    DM_star = DM_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)
    thetastar = rs_star / DM_star
    return np.array([thetastar, Ob_h2, Om_h2])


@njit
def z_star(wb, wm):
    """arXiv:2106.00428v2 (eq A-4)"""

    s1, s2, b, m = (0.72314928, 1.00880342, 1.01224091, 1.15661317)

    wb = wb**b
    wm = wm**m

    return (
        wm**-0.7316314841257655
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
    s1, s2, b, m = (1.00272401, 1.00043175, 1.00323178, 1.00764938)

    wb = wb**b
    wm = wm**m

    return (
        1 + s1 * 428.169 * wb**0.256459 * wm**0.616388 + s2 * 925.56 * wm**0.751615
    ) * wm**-0.714129
