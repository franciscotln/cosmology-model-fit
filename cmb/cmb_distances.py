from numba import njit
import numpy as np

# -------- physical constants and parameters --------
C_KM_PER_S = 299792.458
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4729753287140862e-05
PARSEC_IN_M = 3.085677581491367e+16
G = 6.6743e-11  # m^3 kg^-1 s^-2
H_BAR = 1.0545718176461565e-34
K_BOLTZ = 1.380649e-23
PI = 3.141592653589793
# ---------------------------------------------------


# -- integration settings and Gauss-Legendre nodes --
_HZ_FUNC = None
N_DM = 16
N_RS = 8
GL_X_DM, GL_W_DM = np.polynomial.legendre.leggauss(N_DM)
GL_X_RS, GL_W_RS = np.polynomial.legendre.leggauss(N_RS)
# change integration variable
# a = u^2, da = 2 * u * du
# ---------------------------------------------------


def O_gamma_h2(T_cmb):
    c = C_KM_PER_S * 1e+03
    rho_gamma = (PI**2 / 15.0) * (K_BOLTZ * T_cmb)**4 / (H_BAR**3 * c**3)
    H100 = 0.1 / PARSEC_IN_M
    rho_crit_h2 = 3.0 * H100**2 / (8.0 * PI * G) * c**2
    return rho_gamma / rho_crit_h2


def set_HZ(Hz_fun):
    global _HZ_FUNC
    _HZ_FUNC = Hz_fun


@njit
def _integ_u(u, params):
    # u = sqrt(a): constant in matter era, ~linear in radiation era
    z = 1.0 / (u * u) - 1.0
    return 2.0 * C_KM_PER_S / (u**3 * _HZ_FUNC(z, params))


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
