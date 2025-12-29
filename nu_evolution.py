from numba import njit
import numpy as np
from scipy.integrate import quad
from scipy import constants as sc
import matplotlib.pyplot as plt

k_B = 8.617333262e-5
N_EFF = 3.044
TCMB = 2.7255
mnu_tot = 0.06
T_nu0 = (4 / 11) ** (1 / 3) * TCMB * (N_EFF / 3) ** (1 / 4)
T_nu0_eV = T_nu0 * k_B


def get_O_gamma_h2(T_cmb):
    rho_gamma = (np.pi**2 / 15.0) * (sc.k * T_cmb) ** 4 / (sc.hbar**3 * sc.c**3)
    H100 = 100.0 * 1000.0 / (1e6 * sc.parsec)
    rho_crit_h2 = 3.0 * H100**2 / (8.0 * np.pi * sc.G) * sc.c**2
    return rho_gamma / rho_crit_h2


O_GAMMA_H2 = get_O_gamma_h2(TCMB)


@njit
def integrand(q, mz):
    return q**2 * np.sqrt(q**2 + mz**2) / (np.exp(q) + 1)


def rho_nu_full(y):
    return quad(integrand, 0, 100, args=(y,))[0]


z_range = np.logspace(-3, 7, 10_000)

rho_nu_fermi = np.vectorize(rho_nu_full)

m0 = mnu_tot / T_nu0_eV
Iy0 = rho_nu_fermi(m0)
Iyz = rho_nu_fermi(m0 / (1 + z_range))
fermi_dirac = (1 + z_range) ** 4 * (Iyz / Iy0)


@njit
def _Omnu_comp(z, b):
    p = 1.95648
    zp1 = 1 + z
    ratio = (zp1**p + b**p) / (1 + b**p)
    return zp1**3 * ratio ** (1 / p)


@njit
def Omnu_z(z):
    """
    Two-fluid model for massive neutrinos (Neff in the range 2.90-3.12 and mnu_tot=0.06 eV)
    """

    B1 = 1.38103793 * N_EFF**2 - 14.98287611 * N_EFF + 112.84492554
    B2 = 2.72486 * B1
    W1 = 0.53757

    return W1 * _Omnu_comp(z, B1) + (1 - W1) * _Omnu_comp(z, B2)


approx = Omnu_z(z_range)
max_err = np.max(np.abs((approx - fermi_dirac) / fermi_dirac)) * 100
print(f"Max rel diff: {max_err:.5f}%")

plt.semilogx(z_range, (approx / fermi_dirac - 1) * 100, lw=2)
plt.xlabel("Redshift z")
plt.ylabel("Relative Difference (%)")
plt.title(f"Two-Fluid Approximation Residuals\nMax Error: {max_err:.4f}%")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.axhline(0, color="k", lw=0.5)
plt.show()

plt.loglog(z_range, approx, label="Two-Fluid Approximation", lw=2)
plt.loglog(z_range, fermi_dirac, "--", label="Fermi-Dirac Integral", lw=2)
plt.xlabel("Redshift z")
plt.ylabel(r"$\rho_\nu(z)/\rho_{\nu,0}$")
plt.title("Massive Neutrino Energy Density Evolution")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()


def _d_Omnu_comp(z, b):
    p = 1.95648
    zp1 = 1 + z

    num = zp1**p + b**p
    ratio = num / (1 + b**p)
    f_z = zp1**3 * ratio ** (1 / p)

    term1 = 3 / zp1
    term2 = zp1 ** (p - 1) / num

    return f_z * (term1 + term2)


def d_Omnu_z(z):
    B1 = 1.38103793 * N_EFF**2 - 14.98287611 * N_EFF + 112.84492554
    B2 = 2.72486 * B1
    W1 = 0.53757

    return W1 * _d_Omnu_comp(z, B1) + (1 - W1) * _d_Omnu_comp(z, B2)


def get_w(z):
    """
    Equation of state w(z) for massive neutrinos using the two-fluid approximation
    """
    return -1 + ((1 + z) / 3.0) * d_Omnu_z(z) / Omnu_z(z)


def integrand_pressure(q, mz):
    return (q**4 / np.sqrt(q**2 + mz**2)) / (np.exp(q) + 1)


def get_w_fermi(z):
    """
    Equation of state w(z) for massive neutrinos using the Fermi-Dirac integral
    """
    m0 = mnu_tot / T_nu0_eV
    mz = m0 / (1 + z)
    I_rho = quad(integrand, 0, 100, args=(mz,))[0]
    I_press = quad(integrand_pressure, 0, 100, args=(mz,))[0]
    w = (1.0 / 3.0) * (I_press / I_rho)
    return w


get_w_fermi = np.vectorize(get_w_fermi)

w_approx = get_w(z_range)
w_fermi = get_w_fermi(z_range)

plt.loglog(z_range, w_approx)
plt.loglog(z_range, w_fermi, "--")
plt.legend(["Two-Fluid Approximation", "Fermi-Dirac Integral"])
plt.title("Equation of State w(z)")
plt.ylabel("w(z)")
plt.axhline(1e-5, color="red", lw=1, ls="--")
plt.axhline(1 / 3, color="red", lw=1, ls="--")
plt.xlabel("z")
plt.grid(True)
plt.show()

p_approx = w_approx * approx
p_fermi = w_fermi * fermi_dirac

plt.loglog(z_range, p_approx)
plt.loglog(z_range, p_fermi, "--")
plt.legend(["Two-Fluid Approximation", "Fermi-Dirac Integral"])
plt.title("Neutrino Pressure Evolution")
plt.ylabel(r"$p_\nu(z)/\rho_{\nu,0}$")
plt.xlabel("z")
plt.grid(True)
plt.show()
