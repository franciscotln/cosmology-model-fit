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
def compute_q(m0, coeffs):
    a, b, c, d = coeffs
    return a + b / (m0**d + c)


def compute_qs(m0):
    q1_coeff = (0.51957627, -0.32910932, 61.69757151, 1.63862403)
    q2_coeff = (1.44003027, 0.17654758, 53.81878899, 1.61796254)
    q3_coeff = (2.98730669, -0.16983842, 43.27196182, 1.58886079)
    q4_coeff = (5.51950997, 0.28214179, 27.888305, 1.54510734)
    q5_coeff = (9.82342901, -1.06024333, 13.63432832, 1.49484095)
    q1 = compute_q(m0, q1_coeff)
    q2 = compute_q(m0, q2_coeff)
    q3 = compute_q(m0, q3_coeff)
    q4 = compute_q(m0, q4_coeff)
    q5 = compute_q(m0, q5_coeff)
    return np.array([q1, q2, q3, q4, q5])


w1, w2, w3, w4 = 0.03801, 0.26266, 0.46544, 0.21715
w5 = 1.0 - w1 - w2 - w3 - w4
weights = np.array([w1, w2, w3, w4, w5])


def compute_rho0(m0, qs, ws):
    rho0 = 0.0
    for i in range(len(qs)):
        rho0 += ws[i] * np.sqrt(qs[i] ** 2 + m0**2)
    return rho0


if __name__ == "__main__":
    qs = compute_qs(m0)
    rho0 = compute_rho0(m0, qs, weights)

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

    def Rho_nu_fluid(z):
        """
        Energy density rho(z) for massive neutrinos using the 5-node approximation
        """
        zp1 = 1.0 + z
        mz_sq = (m0 / zp1) ** 2

        f1 = np.sqrt(qs[0] ** 2 + mz_sq)
        f2 = np.sqrt(qs[1] ** 2 + mz_sq)
        f3 = np.sqrt(qs[2] ** 2 + mz_sq)
        f4 = np.sqrt(qs[3] ** 2 + mz_sq)
        f5 = np.sqrt(qs[4] ** 2 + mz_sq)
        weighted_sum = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5
        return zp1**4 * weighted_sum / rho0

    def pressure_nu_fluid(z):
        """
        Pressure p(z) for massive neutrinos using the 5-node approximation
        """
        zp1 = 1.0 + z
        mz_sq = (m0 / zp1) ** 2
        f1 = np.sqrt(qs[0] ** 2 + mz_sq)
        f2 = np.sqrt(qs[1] ** 2 + mz_sq)
        f3 = np.sqrt(qs[2] ** 2 + mz_sq)
        f4 = np.sqrt(qs[3] ** 2 + mz_sq)
        f5 = np.sqrt(qs[4] ** 2 + mz_sq)
        weighted_sum = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5
        weighted_sum_inv = mz_sq * (w1 / f1 + w2 / f2 + w3 / f3 + w4 / f4 + w5 / f5)
        return (1 / 3) * zp1**4 * (weighted_sum - weighted_sum_inv) / rho0

    def w_nu_fluid(z):
        """
        Equation of state w(z) for massive neutrinos using the 5-node approximation
        """
        mz_sq = (m0 / (1.0 + z)) ** 2
        f1 = np.sqrt(qs[0] ** 2 + mz_sq)
        f2 = np.sqrt(qs[1] ** 2 + mz_sq)
        f3 = np.sqrt(qs[2] ** 2 + mz_sq)
        f4 = np.sqrt(qs[3] ** 2 + mz_sq)
        f5 = np.sqrt(qs[4] ** 2 + mz_sq)
        numerator = w1 / f1 + w2 / f2 + w3 / f3 + w4 / f4 + w5 / f5
        denominator = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5
        return (1 / 3) - (1 / 3) * mz_sq * numerator / denominator

    z_range = np.logspace(-3, 7, 10_000)

    rho_fermi_dirac = Rho_nu_fermi_dirac(z_range)
    rho_approx = Rho_nu_fluid(z_range)

    rel_err = 100 * (rho_approx / rho_fermi_dirac - 1)
    max_err = np.max(np.abs(rel_err))
    print(f"Max rel diff: {max_err:.5f}%")  # 9.8e-04 %
    print(f"RMS rel diff: {np.sqrt(np.mean((rel_err) ** 2)):.6f}%")  # 5.63e-04 %

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
        mz_sq = (m0 / (1.0 + z)) ** 2

        f1 = np.sqrt(qs[0] ** 2 + mz_sq)
        f2 = np.sqrt(qs[1] ** 2 + mz_sq)
        f3 = np.sqrt(qs[2] ** 2 + mz_sq)
        f4 = np.sqrt(qs[3] ** 2 + mz_sq)
        f5 = np.sqrt(qs[4] ** 2 + mz_sq)
        num1 = (4 * qs[0] ** 4 + 5 * qs[0] ** 2 * mz_sq) / f1**3
        num2 = (4 * qs[1] ** 4 + 5 * qs[1] ** 2 * mz_sq) / f2**3
        num3 = (4 * qs[2] ** 4 + 5 * qs[2] ** 2 * mz_sq) / f3**3
        num4 = (4 * qs[3] ** 4 + 5 * qs[3] ** 2 * mz_sq) / f4**3
        num5 = (4 * qs[4] ** 4 + 5 * qs[4] ** 2 * mz_sq) / f5**3

        numerator = w1 * num1 + w2 * num2 + w3 * num3 + w4 * num4 + w5 * num5

        den1 = (4 * qs[0] ** 2 + 3 * mz_sq) / f1
        den2 = (4 * qs[1] ** 2 + 3 * mz_sq) / f2
        den3 = (4 * qs[2] ** 2 + 3 * mz_sq) / f3
        den4 = (4 * qs[3] ** 2 + 3 * mz_sq) / f4
        den5 = (4 * qs[4] ** 2 + 3 * mz_sq) / f5

        denominator = w1 * den1 + w2 * den2 + w3 * den3 + w4 * den4 + w5 * den5

        return (1 / 3) * (numerator / denominator)

    # Sound speed (asymtotic)
    def cs2_asympt(z):
        mz_sq = (m0 / (1.0 + z)) ** 2
        f1 = np.sqrt(qs[0] ** 2 + mz_sq)
        f2 = np.sqrt(qs[1] ** 2 + mz_sq)
        f3 = np.sqrt(qs[2] ** 2 + mz_sq)
        f4 = np.sqrt(qs[3] ** 2 + mz_sq)
        f5 = np.sqrt(qs[4] ** 2 + mz_sq)
        sum_rho = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5
        sum2 = w1 / f1 + w2 / f2 + w3 / f3 + w4 / f4 + w5 / f5
        sum_pressure = (1 / 3) * (sum_rho - mz_sq * sum2)
        sum3 = (
            w1 * f1**3 / qs[0] ** 2
            + w2 * f2**3 / qs[1] ** 2
            + w3 * f3**3 / qs[2] ** 2
            + w4 * f4**3 / qs[3] ** 2
            + w5 * f5**3 / qs[4] ** 2
        )
        return (sum_rho + sum_pressure) / (3 * sum_rho + sum3)

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
