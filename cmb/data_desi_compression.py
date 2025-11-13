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

N_EFF = 3.044
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4729e-05

T_nu0 = (4 / 11) ** (1 / 3) * TCMB  # K
T_nu0_eV = T_nu0 * 8.617333262e-5  #  1.67639e-04 eV
mnu_tot = 0.06  # total mass [eV]
Omnu_h2 = mnu_tot / 93.14  # present-day Omega_nu*h^2
z_nr = mnu_tot / (3.15 * T_nu0_eV)


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


Orh2_h_z = Omega_r_h2(3.044)
Orh2_l_z = Omega_r_h2(2.044)


def rs_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Rb = 3 * Obh2 / (4 * O_GAMMA_H2)
    Obc = (Och2 + Obh2) / h**2
    Or = Orh2_h_z / h**2

    def integrand(a):
        denom = a**2 * Ez_func(1 / a - 1, Obc, Or, w0, wa) * np.sqrt(3 * (1 + Rb * a))
        return 1 / denom

    return (c / H0) * quad(integrand, 1e-09, 1 / (1 + z_lim))[0]


def DM_z(Ez_func, z_lim, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Obc = (Och2 + Obh2) / h**2
    Omnu = Omnu_h2 / h**2
    Or_l_z = Orh2_l_z / h**2
    Or_h_z = Orh2_h_z / h**2
    int_l_z, _ = quad(lambda z: 1 / Ez_func(z, Obc + Omnu, Or_l_z, w0, wa), 0, z_nr)
    int_h_z, _ = quad(lambda z: 1 / Ez_func(z, Obc, Or_h_z, w0, wa), z_nr, z_lim)
    return (int_l_z + int_h_z) * c / H0



def cmb_distances(Ez_func, H0, Ob_h2, Oc_h2, w0=-1, wa=0):
    Om_h2 = Oc_h2 + Ob_h2 + Omnu_h2
    zstar = z_star(wb=Ob_h2, wm=Om_h2)
    rs_star = rs_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)
    DM_star = DM_z(Ez_func, zstar, H0, Ob_h2, Oc_h2, w0, wa)
    thetastar = rs_star / DM_star
    return np.array([thetastar, Ob_h2, Om_h2])


@njit
def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""
    SCALING_FID = 1.0010481824509851

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


@njit
def z_star(wb, wm):
    """arXiv:astro-ph/9510117v2 (eq-1)"""
    SCALING_FID = 0.9981350407579086

    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    factor_1 = 1 + 0.00124 * wb**-0.738
    factor_2 = 1 + g1 * wm**g2
    return SCALING_FID * 1048 * factor_1 * factor_2


@njit
def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""
    SCALING_FID = 1.0001866265459478  # reproduces rdrag from integral

    return (
        SCALING_FID
        * (1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615)
        * wm**-0.714129
    )
