from multiprocessing import Pool
import numpy as np
import emcee
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import corner
import matplotlib.pyplot as plt
from y2018quasars.data import get_binned_data as get_quasar_data
from y2024DES.data import get_data as get_des5y_data

legend, z, mu, sigma_mu = get_quasar_data(50)
sn_legend, sn_z, sn_z_hel, sn_mu, sn_cov = get_des5y_data()
cho_sn = cho_factor(sn_cov)

c = 299792.458  # Speed of light (km/s)
H0 = 70  # Hubble constant (km/s/Mpc)

z_grid = np.linspace(0, np.max(z), num=3000)
one_plus_z = 1 + z_grid
z_unique = np.unique(np.concatenate((z, sn_z)))


def Ez(params):
    Omega_m, w0 = params[3], params[4]
    return np.sqrt(Omega_m * one_plus_z**3 + (1 - Omega_m) * ((2 * one_plus_z**2) / (1 + one_plus_z**2))**(3 * (1 + w0)))

def integral_Ez(z, params):
    integral_values = cumulative_trapezoid(1 / Ez(params), z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)


def mu_model(z, theta):
    luminosity_distance = (1 + z) * c / H0 * integral_Ez(z, theta)
    return 25 + 5 * np.log10(luminosity_distance)


def chi_squared_sn(theta):
    deltaM = theta[2]
    mu_theory = mu_model(z_unique, theta)
    delta_sn = sn_mu - deltaM - np.interp(sn_z, z_unique, mu_theory)
    chi_2_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))
    return (chi_2_sn, mu_theory)


def chi_squared_quasar(mu_theory, theta):
    deltaM, s = theta[0], theta[1]
    delta_quasars = mu - deltaM - np.interp(z, z_unique, mu_theory)
    chi_2_quasars = np.sum(delta_quasars**2 / (sigma_mu**2 + s**2))
    return (chi_2_quasars, mu)


def log_likelihood(theta):
    chi_squared_sn_val, mu_theory = chi_squared_sn(theta)
    chi_squared_val, _ = chi_squared_quasar(mu_theory, theta)
    return -0.5 * chi_squared_sn_val - 0.5 * (
        chi_squared_val + np.sum(np.log(sigma_mu**2 + theta[1] ** 2))
    )


bounds = np.array(
    [
        (-0.5, 0.5),  # ΔM_qsr
        (0, 3),  # s
        (-0.6, 0.6),  # ΔM_sn
        (0, 0.8),  # Ωm
        (-2, 0),  # w0
    ]
)


def log_prior(theta):
    if np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return 0.0
    else:
        return -np.inf


def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def main():
    ndim = len(bounds)
    nwalkers = 8 * ndim
    nsteps = 6000
    initial_state = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        sampler.run_mcmc(initial_state, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=1000, flat=True)

    for i, param in enumerate(["ΔM_qsr", "s", "ΔM_sn", "Omega_m", "w0"]):
        mcmc_val = np.percentile(samples[:, i], [16, 50, 84])
        print(
            f"{param}: {mcmc_val[1]:.3f} +{mcmc_val[2]-mcmc_val[1]:.3f} -{mcmc_val[1]-mcmc_val[0]:.3f}"
        )

    deltaM_qsr_50 = np.median(samples[:, 0])
    s_50 = np.median(samples[:, 1])
    deltaM_50 = np.median(samples[:, 2])
    omega_m_50 = np.median(samples[:, 3])
    w0_50 = np.median(samples[:, 4])
    best_fit_params = [deltaM_qsr_50, s_50, deltaM_50, omega_m_50, w0_50]

    chi_squared_sn_val, mu_theory = chi_squared_sn(best_fit_params)
    chi_squared_qsr_val, _ = chi_squared_quasar(mu_theory, best_fit_params)
    print(f"chi squared SN: {chi_squared_sn_val:.2f}")
    print(f"chi squared quasars: {chi_squared_qsr_val:.2f}")
    print(f"chi squared total: {chi_squared_sn_val + chi_squared_qsr_val:.2f}")

    corner.corner(
        samples,
        labels=["$\\Delta_{Mqsr}$", "$s$", "$\\Delta_{Msn}$", "$\\Omega_m$", "$w_0$"],
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    z_plot = np.linspace(0.02, np.max(z), num=1000)
    theory = mu_model(z_plot, best_fit_params)

    plt.plot(z_plot, theory, color="green", label="$\mu_T$", alpha=0.8)
    plt.errorbar(
        z,
        mu - deltaM_qsr_50,
        yerr=np.sqrt(sigma_mu**2 + s_50**2),
        fmt=".",
        color="blue",
        label=legend,
        alpha=0.4,
        lw=0.5,
    )
    plt.scatter(sn_z, sn_mu - deltaM_50, s=5, color="red", alpha=0.5, label=sn_legend)
    plt.ylim(32.5, 54)
    plt.xlabel("Redshift ($z$)")
    plt.ylabel("$\mu$")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
ΔM_qsr: -0.078 +0.038 -0.038 mag
s: 1.518 +0.023 -0.023 mag
ΔM_sn: 0.040 +0.011 -0.011 mag
Omega_m: 0.382 +0.017 -0.017
w0: -1
chi squared SN: 1643.80
chi squared quasars: 2392.71
chi squared total: 4036.51

==================================

Flat wCDM
ΔM_qsr: -0.091 +0.061 -0.066 mag
s: 0.405 +0.050 -0.042 mag
ΔM_sn: 0.023 +0.013 -0.013 mag
Omega_m: 0.385 +0.041 -0.049
w0: -1.087 +0.146 -0.153
chi squared SN: 1642.01
chi squared quasars: 48.62
chi squared total: 1690.63

==================================

Flat wzCDM
ΔM_qsr: -0.093 +0.063 -0.065 mag
s: 0.408 +0.050 -0.042 mag
ΔM_sn: 0.024 +0.013 -0.013 mag
Omega_m: 0.373 +0.038 -0.041
w0: -1.052 +0.136 -0.153
chi squared SN: 1641.56
chi squared quasars: 48.63
chi squared total: 1690.19
"""
