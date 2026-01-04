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
def integrand_density(q, z):
    mz = m0 / (1.0 + z)
    return q**2 * (q**2 + mz**2) ** (1 / 2) / (np.exp(q) + 1)


"""
There are two values of q (q1 = 2.0811166 and q2 = 4.5036711) for which the integrand becomes:
Bz = B / (1 + z)
return (1 + Bz**2) ** (1 / 2)

where B1 = m0/q1 and B2 = m0/q2
"""

R0 = quad(integrand_density, 0, 100, args=(0,))[0]


def Rho_nu_fermi_dirac(z):
    """
    Energy density rho(z) for massive neutrinos using the Fermi-Dirac integral
    """
    zp1 = 1.0 + z

    density = zp1**4 * quad(integrand_density, 0, 100, args=(z,))[0] / R0
    return density


@njit
def integrand_pressure(q, z):
    mz = m0 / (1.0 + z)
    coeff = q**2 / (q**2 + mz**2)
    return coeff * integrand_density(q, z)


def pressure_nu_fermi_dirac(z):
    """
    Pressure p(z) for massive neutrinos using the Fermi-Dirac integral
    """
    zp1 = 1.0 + z
    return (1 / 3) * zp1**4 * quad(integrand_pressure, 0, 100, args=(z,))[0] / R0


Rho_nu_fermi_dirac = np.vectorize(Rho_nu_fermi_dirac)
pressure_nu_fermi_dirac = np.vectorize(pressure_nu_fermi_dirac)


def w_nu_fermi_dirac(z):
    """
    Equation of state w(z) for massive neutrinos using the Fermi-Dirac integral
    """
    return pressure_nu_fermi_dirac(z) / Rho_nu_fermi_dirac(z)


# Analytical two-fluid approximation functions and coefficients
def compute_B1(m0):
    return m0**0.99918671 / 1.16318070


def compute_B2(m0):
    return m0**0.99972151 / 7.23392202


def compute_B3(m0):
    return m0**0.99960262 / 3.29578202


def compute_W1(m0):
    return m0**1.76287695 / (9.81495217 + 3.599578112 * m0**1.76168671)


def compute_W2(m0):
    return m0**2.20779182 / (-408.51056625 + 8.78920026 * m0**2.20906010)


B1 = compute_B1(m0)
B2 = compute_B2(m0)
B3 = compute_B3(m0)
W1 = compute_W1(m0)
W2 = compute_W2(m0)
W3 = 1.0 - W1 - W2
P = 2.0


def fluid_component(B, z):
    Bz = B / (1.0 + z)
    return np.sqrt(1 + Bz**2)


R01 = fluid_component(B1, 0)
R02 = fluid_component(B2, 0)
R03 = fluid_component(B3, 0)


def Rho_nu_fluid(z):
    """
    Two-fluid energy density rho(z) for massive neutrinos
    - valid for 200 <= m0 = mnu_tot / T_nu0 <= 3160
    """
    zp1 = 1.0 + z

    density1 = W1 * fluid_component(B1, z) / R01
    density2 = W2 * fluid_component(B2, z) / R02
    density3 = W3 * fluid_component(B3, z) / R03
    return zp1**4 * (density1 + density2 + density3)


def pressure_nu_fluid(z):
    """
    Pressure p(z) for massive neutrinos using the two-fluid approximation
    """
    zp1 = 1.0 + z

    B1z = B1 / zp1
    B2z = B2 / zp1
    B3z = B3 / zp1

    coeff1 = 1 / (1 + B1z**P)
    coeff2 = 1 / (1 + B2z**P)
    coeff3 = 1 / (1 + B3z**P)

    density1 = W1 * fluid_component(B1, z) / R01
    density2 = W2 * fluid_component(B2, z) / R02
    density3 = W3 * fluid_component(B3, z) / R03

    return zp1**4 * (coeff1 * density1 + coeff2 * density2 + coeff3 * density3) / 3


def w_nu_fluid(z):
    """
    Equation of state w(z) for massive neutrinos using the two-fluid approximation
    """
    return pressure_nu_fluid(z) / Rho_nu_fluid(z)


z_range = np.logspace(-3, 7, 10_000)

rho_fermi_dirac = Rho_nu_fermi_dirac(z_range)
rho_approx = Rho_nu_fluid(z_range)

rel_err = 100 * (rho_approx / rho_fermi_dirac - 1)
max_err = np.max(np.abs(rel_err))
print(f"Max rel diff: {max_err:.5f}%")  # 0.02479%
print(f"RMS rel diff: {np.sqrt(np.mean((rel_err / 100) ** 2))}")  # 5.6022e-05

plt.style.use("seaborn-v0_8-bright")

plt.semilogx(z_range, rel_err, lw=2)
plt.xlabel("Redshift z")
plt.ylabel("Relative Difference (%)")
plt.title(f"3-Fluid Approximation Residuals\nMax Error: {max_err:.4f}%")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.axhline(0, color="k", lw=0.5)
plt.tight_layout()
plt.show()

a_range = np.linspace(1e-07, 1, 2000)
zs = 1 / a_range - 1

plt.loglog(a_range, w_nu_fluid(zs))
plt.loglog(a_range, w_nu_fermi_dirac(zs), "--")
plt.legend(["3-Fluid Approximation", "Fermi-Dirac Integral"])
plt.title("Equation of State w(a)")
plt.ylabel("w(a)")
plt.xlabel("Scale Factor a")
plt.grid(True)
plt.show()

plt.loglog(a_range, Rho_nu_fluid(zs))
plt.loglog(a_range, Rho_nu_fermi_dirac(zs), "--", lw=2)
plt.legend(["3-Fluid Approximation", "Fermi-Dirac Integral"])
plt.xlabel("Scale Factor a")
plt.ylabel(r"$\rho_\nu(a)/\rho_{\nu,0}$")
plt.title("Massive Neutrino Energy Density Evolution")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()

plt.loglog(a_range, pressure_nu_fluid(zs))
plt.loglog(a_range, pressure_nu_fermi_dirac(zs), "--")
plt.legend(["3-Fluid Approximation", "Fermi-Dirac Integral"])
plt.title("Neutrino Pressure Evolution")
plt.ylabel(r"$p_\nu(z)/\rho_{\nu,0}$")
plt.xlabel("Scale Factor a")
plt.grid(True)
plt.show()
