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


# Analytical 3-fluid approximation functions and coefficients
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


def Rho_nu_fluid(z):
    """
    3-fluid energy density rho(z) for massive neutrinos
    """
    f1 = fluid_component(B1, z)
    f2 = fluid_component(B2, z)
    f3 = fluid_component(B3, z)
    density = W1 * f1 + W2 * f2 + W3 * f3
    return (1 + z) ** 4 * density / normalization


def pressure_nu_fluid(z):
    """
    Pressure p(z) for massive neutrinos using the 3-fluid approximation
    """
    f1 = fluid_component(B1, z)
    f2 = fluid_component(B2, z)
    f3 = fluid_component(B3, z)
    pressure = W1 / f1 + W2 / f2 + W3 / f3

    return (1 / 3) * (1 + z) ** 4 * pressure / normalization


def w_nu_fluid(z):
    """
    Equation of state w(z) for massive neutrinos using the two-fluid approximation
    """
    f1 = fluid_component(B1, z)
    f2 = fluid_component(B2, z)
    f3 = fluid_component(B3, z)
    return (1 / 3) * (W1 / f1 + W2 / f2 + W3 / f3) / (W1 * f1 + W2 * f2 + W3 * f3)


z_range = np.logspace(-3, 7, 10_000)

rho_fermi_dirac = Rho_nu_fermi_dirac(z_range)
rho_approx = Rho_nu_fluid(z_range)

rel_err = 100 * (rho_approx / rho_fermi_dirac - 1)
max_err = np.max(np.abs(rel_err))
print(f"Max rel diff: {max_err:.5f}%")  # 2.445e-02 %
print(f"RMS rel diff: {np.sqrt(np.mean((rel_err) ** 2)):.6f}%")  # 5.528e-03 %

plt.style.use("bmh")

plt.semilogx(z_range, rel_err, lw=2)
plt.xlabel("Redshift z")
plt.ylabel("Relative Difference (%)")
plt.title(f"3-Fluid Approximation Residuals\nMax Error: {max_err:.4f}%")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.axhline(0, color="k", lw=0.5)
plt.tight_layout()
plt.show()

a_range = np.linspace(1e-07, 1, 4000)
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


def cs2g(z):
    zp1 = 1.0 + z
    Bz1_sqr = (B1 / zp1) ** 2
    Bz2_sqr = (B2 / zp1) ** 2
    Bz3_sqr = (B3 / zp1) ** 2

    f1 = np.sqrt(1 + Bz1_sqr)
    f2 = np.sqrt(1 + Bz2_sqr)
    f3 = np.sqrt(1 + Bz3_sqr)

    drho_dz_over_zp1_cubed = (
        W1 * (4 + 3 * Bz1_sqr) / f1
        + W2 * (4 + 3 * Bz2_sqr) / f2
        + W3 * (4 + 3 * Bz3_sqr) / f3
    )

    dp_dz_over_zp1_cubed = (1 / 3) * (
        W1 * (4 + 5 * Bz1_sqr) / f1**3
        + W2 * (4 + 5 * Bz2_sqr) / f2**3
        + W3 * (4 + 5 * Bz3_sqr) / f3**3
    )

    return dp_dz_over_zp1_cubed / drho_dz_over_zp1_cubed


def cs2asp(z):
    f1 = fluid_component(B1, z)
    f2 = fluid_component(B2, z)
    f3 = fluid_component(B3, z)

    density = W1 * f1 + W2 * f2 + W3 * f3
    numerator = density + (1 / 3) * (W1 / f1 + W2 / f2 + W3 / f3)
    denominator = density + (1 / 3) * (W1 * f1**3 + W2 * f2**3 + W3 * f3**3)

    return (1 / 3) * (numerator / denominator)


plt.plot(np.log(a_range), cs2asp(zs), label="asymptotic")
plt.plot(np.log(a_range), cs2g(zs), label="adiabatic")
plt.title("Neutrino Sound Speed Squared Evolution")
plt.ylabel(r"$c_s^2(a)$")
plt.legend()
plt.xlim(-8, None)
plt.xlabel("Scale Factor ln(a)")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()
