import numpy as np
from scipy.integrate import quad
from scipy import constants as sc
import matplotlib.pyplot as plt

k_B = sc.k / sc.e  # Boltzmann constant in eV/K
N_EFF = 3.044
TCMB = 2.7255
T_nu0 = (4 / 11) ** (1 / 3) * TCMB * (N_EFF / 3) ** (1 / 4)
T_nu0_eV = T_nu0 * k_B
mnu_tot = 0.06
m0 = mnu_tot / T_nu0_eV


def O_gamma_h2(T_cmb):
    rho_gamma = (np.pi**2 / 15.0) * (sc.k * T_cmb) ** 4 / (sc.hbar**3 * sc.c**3)
    H100 = 100.0 * 1000.0 / (1e6 * sc.parsec)
    rho_crit_h2 = 3.0 * H100**2 / (8.0 * np.pi * sc.G) * sc.c**2
    return rho_gamma / rho_crit_h2


def integrand_rho(q, mz):
    return q**2 * np.sqrt(q**2 + mz**2) / (np.exp(q) + 1)


def rho_nu_fermi_dirac(mz):
    return quad(integrand_rho, 0, 100, args=(mz,))[0]


def integrand_pressure(q, mz):
    return q**2 * integrand_rho(q, mz) / (q**2 + mz**2)


def w_nu_fermi_dirac(z):
    """
    Equation of state w(z) for massive neutrinos using the Fermi-Dirac integral
    """
    mz = m0 / (1 + z)
    I_rho = quad(integrand_rho, 0, 100, args=(mz,))[0]
    I_press = quad(integrand_pressure, 0, 100, args=(mz,))[0]
    return (1.0 / 3.0) * (I_press / I_rho)


rho_nu_fermi_dirac = np.vectorize(rho_nu_fermi_dirac)
w_nu_fermi_dirac = np.vectorize(w_nu_fermi_dirac)


def Rhonu_comp(z, B):
    p = 1.95648
    zp1 = 1 + z
    ratio = (zp1**p + B**p) / (1 + B**p)
    return zp1**3 * ratio ** (1 / p)


def Rhonu_z(z):
    """
    Two-fluid model for massive neutrinos (Neff in the range 2.90-3.12 and mnu_tot=0.06 eV)
    """

    B1 = 1.38103793 * N_EFF**2 - 14.98287611 * N_EFF + 112.84492554
    B2 = 2.72486 * B1
    W1 = 0.53757

    return W1 * Rhonu_comp(z, B1) + (1.0 - W1) * Rhonu_comp(z, B2)


def d_Rhonu_comp(z, B):
    p = 1.95648
    zp1 = 1 + z

    return Rhonu_comp(z, B) * (3.0 + zp1**p / (zp1**p + B**p)) / zp1


def d_Rhonu_z(z):
    B1 = 1.38103793 * N_EFF**2 - 14.98287611 * N_EFF + 112.84492554
    B2 = 2.72486 * B1
    W1 = 0.53757

    return W1 * d_Rhonu_comp(z, B1) + (1 - W1) * d_Rhonu_comp(z, B2)


def w_nu_approx(z):
    """
    Equation of state w(z) for massive neutrinos using the two-fluid approximation
    """
    return -1.0 + ((1 + z) / 3.0) * d_Rhonu_z(z) / Rhonu_z(z)


z_range = np.logspace(-3, 7, 10_000)

Iy0 = rho_nu_fermi_dirac(m0)
Iyz = rho_nu_fermi_dirac(m0 / (1.0 + z_range))
rho_fermi_dirac = (1.0 + z_range) ** 4 * (Iyz / Iy0)

rho_approx = Rhonu_z(z_range)

rel_err = 100 * (rho_approx / rho_fermi_dirac - 1)
max_err = np.max(np.abs(rel_err))
print(f"Max rel diff: {max_err:.5f}%")

plt.style.use("seaborn-v0_8-bright")

plt.semilogx(z_range, rel_err, lw=2)
plt.xlabel("Redshift z")
plt.ylabel("Relative Difference (%)")
plt.title(f"Two-Fluid Approximation Residuals\nMax Error: {max_err:.4f}%")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.axhline(0, color="k", lw=0.5)
plt.show()

plt.loglog(z_range, rho_approx, lw=2)
plt.loglog(z_range, rho_fermi_dirac, "--", lw=2)
plt.legend(["Two-Fluid Approximation", "Fermi-Dirac Integral"])
plt.xlabel("Redshift z")
plt.ylabel(r"$\rho_\nu(z)/\rho_{\nu,0}$")
plt.title("Massive Neutrino Energy Density Evolution")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()

w_approx = w_nu_approx(z_range)
w_fermi = w_nu_fermi_dirac(z_range)

plt.semilogx(z_range, w_approx)
plt.semilogx(z_range, w_fermi, "--")
plt.legend(["Two-Fluid Approximation", "Fermi-Dirac Integral"])
plt.title("Equation of State w(z)")
plt.ylabel("w(z)")
plt.axhline(0, lw=1, ls="-.", color="gray")
plt.axhline(1 / 3, lw=1, ls="-.", color="gray")
plt.xlabel("z")
plt.grid(True)
plt.show()

a_range = np.linspace(1e-05, 1, 1000)
zs = 1 / a_range - 1
plt.loglog(a_range, w_nu_approx(zs))
plt.loglog(a_range, w_nu_fermi_dirac(zs), "--")
plt.legend(["Two-Fluid Approximation", "Fermi-Dirac Integral"])
plt.title("Equation of State w(a)")
plt.ylabel("w(a)")
plt.xlabel("a")
plt.grid(True)
plt.show()

p_approx = w_approx * rho_approx
p_fermi = w_fermi * rho_fermi_dirac

plt.loglog(z_range, p_approx)
plt.loglog(z_range, p_fermi, "--")
plt.legend(["Two-Fluid Approximation", "Fermi-Dirac Integral"])
plt.title("Neutrino Pressure Evolution")
plt.ylabel(r"$p_\nu(z)/\rho_{\nu,0}$")
plt.xlabel("z")
plt.grid(True)
plt.show()
