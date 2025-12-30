import numpy as np
from scipy.integrate import quad
from scipy import constants as sc
from numba import njit
import matplotlib.pyplot as plt

k_B = sc.k / sc.e  # Boltzmann constant in eV/K
N_EFF = 3.044
TCMB = 2.7255
T_nu0 = (4 / 11) ** (1 / 3) * TCMB * (N_EFF / 3) ** (1 / 4)
T_nu0_eV = T_nu0 * k_B
mnu_tot = 0.06
m0 = mnu_tot / T_nu0_eV
Omnu_h2 = mnu_tot / (94.0641 / (N_EFF / 3.0) ** 0.75)


def O_gamma_h2(T_cmb):
    rho_gamma = (np.pi**2 / 15.0) * (sc.k * T_cmb) ** 4 / (sc.hbar**3 * sc.c**3)
    H100 = 100.0 * 1000.0 / (1e6 * sc.parsec)
    rho_crit_h2 = 3.0 * H100**2 / (8.0 * np.pi * sc.G) * sc.c**2
    return rho_gamma / rho_crit_h2


@njit
def integrand_rho(q, mz):
    return q**2 * np.sqrt(q**2 + mz**2) / (np.exp(q) + 1)


rho_norm0 = quad(integrand_rho, 0, 100, args=(m0,))[0]


def Rho_nu_fermi_dirac(z):
    """
    Energy density rho(z) for massive neutrinos using the Fermi-Dirac integral
    """
    mz = m0 / (1 + z)
    return (1.0 + z) ** 4 * quad(integrand_rho, 0, 100, args=(mz,))[0] / rho_norm0


@njit
def integrand_pressure(q, mz):
    return q**2 * integrand_rho(q, mz) / (q**2 + mz**2)


def pressure_nu_fermi_dirac(z):
    """
    Pressure p(z) for massive neutrinos using the Fermi-Dirac integral
    """
    mz = m0 / (1 + z)
    return (
        (1.0 / 3.0)
        * (1.0 + z) ** 4
        * quad(integrand_pressure, 0, 100, args=(mz,))[0]
        / rho_norm0
    )


Rho_nu_fermi_dirac = np.vectorize(Rho_nu_fermi_dirac)
pressure_nu_fermi_dirac = np.vectorize(pressure_nu_fermi_dirac)


def w_nu_fermi_dirac(z):
    """
    Equation of state w(z) for massive neutrinos using the Fermi-Dirac integral
    """
    return pressure_nu_fermi_dirac(z) / Rho_nu_fermi_dirac(z)


# Analytical two-fluid approximation functions
## Coefficients mnu_tot=0.06 eV and Neff in the range 2.90-3.12
B1 = 1.38103793 * N_EFF**2 - 14.98287611 * N_EFF + 112.84492554
B2 = 2.72486 * B1
W1 = 0.53757
W2 = 1.0 - W1
P = 1.95648


def Rhonu_comp(z, B):
    zp1 = 1 + z
    ratio = (zp1**P + B**P) / (1 + B**P)
    return zp1**3 * ratio ** (1 / P)


def Rhonu_nu_fluid(z):
    """
    Two-fluid model rho(z) for massive neutrinos (Neff in the range 2.90-3.12 and mnu_tot=0.06 eV)
    """
    density1 = W1 * Rhonu_comp(z, B1)
    density2 = W2 * Rhonu_comp(z, B2)
    return density1 + density2


def pressure_nu_fluid(z):
    """
    Pressure p(z) for massive neutrinos using the two-fluid approximation
    """
    density1 = W1 * Rhonu_comp(z, B1)
    density2 = W2 * Rhonu_comp(z, B2)

    zp1 = 1 + z

    coef1 = zp1**P / (zp1**P + B1**P)
    coef2 = zp1**P / (zp1**P + B2**P)
    return (1 / 3) * (coef1 * density1 + coef2 * density2)


def w_nu_fluid(z):
    """
    Equation of state w(z) for massive neutrinos using the two-fluid approximation
    """
    return pressure_nu_fluid(z) / Rhonu_nu_fluid(z)


z_range = np.logspace(-3, 7, 10_000)

rho_fermi_dirac = Rho_nu_fermi_dirac(z_range)
rho_approx = Rhonu_nu_fluid(z_range)

rel_err = 100 * (rho_approx / rho_fermi_dirac - 1)
max_err = np.max(np.abs(rel_err))
print(f"Max rel diff: {max_err:.5f}%")  # 0.02385%

plt.style.use("seaborn-v0_8-bright")

plt.semilogx(z_range, rel_err, lw=2)
plt.xlabel("Redshift z")
plt.ylabel("Relative Difference (%)")
plt.title(f"Two-Fluid Approximation Residuals\nMax Error: {max_err:.4f}%")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.axhline(0, color="k", lw=0.5)
plt.tight_layout()
plt.show()

a_range = np.linspace(1e-07, 1, 2000)
zs = 1 / a_range - 1

plt.loglog(a_range, w_nu_fluid(zs))
plt.loglog(a_range, w_nu_fermi_dirac(zs), "--")
plt.legend(["Two-Fluid Approximation", "Fermi-Dirac Integral"])
plt.title("Equation of State w(a)")
plt.ylabel("w(a)")
plt.xlabel("Scale Factor a")
plt.grid(True)
plt.show()

plt.loglog(a_range, Rhonu_nu_fluid(zs), lw=2)
plt.loglog(a_range, Rho_nu_fermi_dirac(zs), "--", lw=2)
plt.legend(["Two-Fluid Approximation", "Fermi-Dirac Integral"])
plt.xlabel("Scale Factor a")
plt.ylabel(r"$\rho_\nu(a)/\rho_{\nu,0}$")
plt.title("Massive Neutrino Energy Density Evolution")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()

plt.loglog(a_range, pressure_nu_fluid(zs))
plt.loglog(a_range, pressure_nu_fermi_dirac(zs), "--")
plt.legend(["Two-Fluid Approximation", "Fermi-Dirac Integral"])
plt.title("Neutrino Pressure Evolution")
plt.ylabel(r"$p_\nu(z)/\rho_{\nu,0}$")
plt.xlabel("Scale Factor a")
plt.grid(True)
plt.show()
