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


# Analytical 5-node approximation functions and coefficients
def compute_nodes(m0):
    return m0 / np.array([0.5554855, 1.52077029, 3.12302226, 5.71996287, 10.09243889])


def compute_weight(m0, coeffs):
    a, b, c, d = coeffs
    return a + b / (m0**d + c)


def compute_weights(m0):
    w1_coeffs = (7.92827793e-03, -3.19058802e-03, 178.754821, 1.61644550)
    w2_coeffs = (0.137866439, 2.27856337e-02, 173.860696, 1.60725939)
    w3_coeffs = (0.456512842, -4.76472630e-02, 165.916077, 1.59366999)
    w4_coeffs = (0.353930679, 4.07026316e-02, 154.215273, 1.57547094)
    w1 = compute_weight(m0, w1_coeffs)
    w2 = compute_weight(m0, w2_coeffs)
    w3 = compute_weight(m0, w3_coeffs)
    w4 = compute_weight(m0, w4_coeffs)
    w5 = 1.0 - w1 - w2 - w3 - w4
    return np.array([w1, w2, w3, w4, w5])


if __name__ == "__main__":

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

    B_sqr = compute_nodes(m0) ** 2
    W = compute_weights(m0)
    f_0 = np.sqrt(1 + B_sqr)
    normalization = W @ f_0

    def Rho_nu_fluid(z):
        """
        Energy density rho(z) for massive neutrinos using the 5-node approximation
        """
        zp1 = 1.0 + z
        Bz_sqr = B_sqr[:, None] / zp1**2
        f = np.sqrt(1 + Bz_sqr)
        return zp1**4 * W.dot(f) / normalization

    def pressure_nu_fluid(z):
        """
        Pressure p(z) for massive neutrinos using the 5-node approximation
        """
        zp1 = 1.0 + z
        Bz_sqr = B_sqr[:, None] / zp1**2
        f = np.sqrt(1 + Bz_sqr)
        return (1 / 3) * zp1**4 * W.dot(1 / f) / normalization

    def w_nu_fluid(z):
        """
        Equation of state w(z) for massive neutrinos using the 5-node approximation
        """
        zp1 = 1.0 + z
        Bz_sqr = B_sqr[:, None] / zp1**2
        f = np.sqrt(1 + Bz_sqr)
        return (1 / 3) * W.dot(1 / f) / W.dot(f)

    z_range = np.logspace(-3, 7, 10_000)

    rho_fermi_dirac = Rho_nu_fermi_dirac(z_range)
    rho_approx = Rho_nu_fluid(z_range)

    rel_err = 100 * (rho_approx / rho_fermi_dirac - 1)
    max_err = np.max(np.abs(rel_err))
    print(f"Max rel diff: {max_err:.5f}%")  # 1.39e-03 %
    print(f"RMS rel diff: {np.sqrt(np.mean((rel_err) ** 2)):.6f}%")  # 8.03e-04 %

    plt.style.use("bmh")

    plt.semilogx(z_range, rel_err, lw=2)
    plt.xlabel("Redshift z")
    plt.ylabel("Relative Difference (%)")
    plt.title(f"5-node Approximation Residuals\nMax Error: {max_err:.4f}%")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.show()

    a_range = np.linspace(1e-07, 1, 4000)
    zs = 1 / a_range - 1

    plt.loglog(a_range, w_nu_fluid(zs))
    plt.loglog(a_range, w_nu_fermi_dirac(zs), "--")
    plt.legend(["5-node Approximation", "Fermi-Dirac Integral"])
    plt.title("Equation of State w(a)")
    plt.ylabel("w(a)")
    plt.xlabel("Scale Factor a")
    plt.grid(True)
    plt.show()

    plt.loglog(a_range, Rho_nu_fluid(zs))
    plt.loglog(a_range, Rho_nu_fermi_dirac(zs), "--", lw=2)
    plt.legend(["5-node Approximation", "Fermi-Dirac Integral"])
    plt.xlabel("Scale Factor a")
    plt.ylabel(r"$\rho_\nu(a)/\rho_{\nu,0}$")
    plt.title("Massive Neutrino Energy Density Evolution")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.show()

    plt.loglog(a_range, pressure_nu_fluid(zs))
    plt.loglog(a_range, pressure_nu_fermi_dirac(zs), "--")
    plt.legend(["5-node Approximation", "Fermi-Dirac Integral"])
    plt.title("Neutrino Pressure Evolution")
    plt.ylabel(r"$p_\nu(z)/\rho_{\nu,0}$")
    plt.xlabel("Scale Factor a")
    plt.grid(True)
    plt.show()

    # Sound speed (fermi-dirac: adiabatic)
    @njit
    def drho_dz_integrand_fermi(q, z):
        zp1 = 1.0 + z
        mz = m0 / zp1
        dm_dz = -m0 / (zp1**2)

        term1 = q**2 * np.sqrt(q**2 + mz**2) / (np.exp(q) + 1)
        term2 = (q**2 * mz / np.sqrt(q**2 + mz**2)) * dm_dz / (np.exp(q) + 1)

        return 4 * term1 / zp1 + term2

    @njit
    def dpressure_dz_integrand_fermi(q, z):
        zp1 = 1.0 + z
        mz = m0 / zp1
        dm_dz = -m0 / (zp1**2)

        common = q**4 / (np.exp(q) + 1)
        term1 = (1 / 3) * common / np.sqrt(q**2 + mz**2)
        term2 = (1 / 3) * common * (-mz * dm_dz) / (q**2 + mz**2) ** 1.5

        return 4 * term1 / zp1 + term2

    def cs2_adiab_fermi(z):
        dp_dz = quad(dpressure_dz_integrand_fermi, 0, 100, args=(z,))[0]
        drho_dz = quad(drho_dz_integrand_fermi, 0, 100, args=(z,))[0]

        return dp_dz / drho_dz

    cs2_adiab_fermi = np.vectorize(cs2_adiab_fermi)

    # Sound speed (adiabatic)
    def cs2_adiab(z):
        zp1_sqr = (1.0 + z) ** 2
        Bz_sqr = B_sqr[:, None] / zp1_sqr
        f = np.sqrt(1 + Bz_sqr)

        dp_dz_over_zp1_cube = (1 / 3) * W.dot((4 + 5 * Bz_sqr) / f**3)
        drho_dz_over_zp1_cube = W.dot((4 + 3 * Bz_sqr) / f)
        return dp_dz_over_zp1_cube / drho_dz_over_zp1_cube

    # Sound speed (asymtotic)
    def cs2_asympt(z):
        zp1_sqr = (1.0 + z) ** 2
        Bz_sqr = Bz_sqr = B_sqr[:, None] / zp1_sqr
        f = np.sqrt(1 + Bz_sqr)

        density = W.dot(f)
        pressure = (1 / 3) * W.dot(1 / f)
        numerator = density + pressure
        denominator = density + (1 / 3) * W.dot(f**3)

        return (1 / 3) * (numerator / denominator)

    approx_cs2_asympt = cs2_asympt(zs)
    approx_cs2_adiab = cs2_adiab(zs)
    exact_cs2_adiab = cs2_adiab_fermi(zs)

    plt.plot(np.log(a_range), approx_cs2_asympt, label="asymptotic")
    plt.plot(np.log(a_range), approx_cs2_adiab, label="adiabatic")
    plt.title("Neutrino Sound Speed Squared Evolution")
    plt.ylabel(r"$c_s^2(a)$")
    plt.legend()
    plt.xlim(-8, None)
    plt.xlabel("Scale Factor ln(a)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.show()

    plt.plot(np.log(a_range), approx_cs2_adiab, label="adiabatic approximation")
    plt.plot(np.log(a_range), exact_cs2_adiab, "--", label="adiabatic exact")
    plt.title("Neutrino Sound Speed Squared Evolution")
    plt.ylabel(r"$c_s^2(a)$")
    plt.legend()
    plt.xlim(-8, None)
    plt.xlabel("Scale Factor ln(a)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
