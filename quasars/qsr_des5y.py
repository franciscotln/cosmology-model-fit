from multiprocessing import Pool
import numpy as np
import emcee
from scipy.integrate import cumulative_trapezoid
import corner
import matplotlib.pyplot as plt
from y2018quasars.data import get_binned_data, factor
from y2024DES.data import get_data as get_des5y_data

legend, z, mu, sigma_mu = get_binned_data(30)
sn_legend, sn_z, sn_mu, sn_cov  = get_des5y_data()
inv_cov_sn = np.linalg.inv(sn_cov)

c = 299792.458 # Speed of light (km/s)
H0 = 70 # Hubble constant (km/s/Mpc)

z_grid = np.linspace(0, np.max(z), num=3000)
one_plus_z = 1 + z_grid
z_unique = np.unique(np.concatenate((z, sn_z)))

def Ez(params):
    Omega_m, w0 = params[3], params[4]
    return np.sqrt(Omega_m * one_plus_z**3 + (1 - Omega_m) * ((2 * one_plus_z**2) / (1 + one_plus_z**2))**(3 * (1 + w0)))

def integral_Ez(z, params):
    integral_values = cumulative_trapezoid(1/Ez(params), z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)

def mu_model_sn(z, theta):
    luminosity_distance = (1 + z) * c / H0 * integral_Ez(z, theta)
    return 25 + 5 * np.log10(luminosity_distance)

def chi_squared_sn(theta):
    deltaM = theta[2]
    mu_theory = mu_model_sn(z_unique, theta)
    delta_sn = sn_mu - deltaM - np.interp(sn_z, z_unique, mu_theory)
    chi_2_sn = np.dot(delta_sn, np.dot(inv_cov_sn, delta_sn))
    return (chi_2_sn, mu_theory)

def chi_squared_quasar(mu_theory, theta):
    beta_prime, s = theta[0], theta[1]
    mu_model = mu + factor * beta_prime
    delta_quasars = np.interp(z, z_unique, mu_theory) - mu_model
    chi_2_quasars = np.sum(delta_quasars**2 / (sigma_mu**2 + s**2))
    return (chi_2_quasars, mu_model)

def log_likelihood(theta):
    s = theta[1]
    chi_squared_sn_val, mu_theory = chi_squared_sn(theta)
    chi_squared_val, _ = chi_squared_quasar(mu_theory, theta)
    return -0.5 * chi_squared_sn_val -0.5 * (chi_squared_val + np.sum(np.log(sigma_mu**2 + s**2)))

def log_prior(theta):
    beta_prime, s, deltaM, omega_m, w0 = theta
    if (-8 < beta_prime < -6 and 0 < s < 3 and -0.6 < deltaM < 0.6 and 0 < omega_m < 0.6 and -1.5 < w0 < 0):
        return 0.0
    return -np.inf

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

def main():
    ndim = 5
    nwalkers = 8 * ndim
    nsteps = 8000
    initial_state = np.array([-7, 1, 0, 0.3, -1]) + 0.01 * np.random.randn(nwalkers, ndim)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        sampler.run_mcmc(initial_state, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=1000, flat=True)

    for i, param in enumerate(["beta'", "s", "ΔM", "Omega_m", "w0"]):
        mcmc_val = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{param}: {mcmc_val[1]:.3f} +{mcmc_val[2]-mcmc_val[1]:.3f} -{mcmc_val[1]-mcmc_val[0]:.3f}")

    beta_50 = np.median(samples[:, 0])
    s_50 = np.median(samples[:, 1])
    deltaM_50 = np.median(samples[:, 2])
    omega_m_50 = np.median(samples[:, 3])
    w0_50 = np.median(samples[:, 4])
    best_fit_params = [beta_50, s_50, deltaM_50, omega_m_50, w0_50]

    chi_squared_sn_val, mu_theory = chi_squared_sn(best_fit_params)
    chi_squared_qsr_val, _ = chi_squared_quasar(mu_theory, best_fit_params)
    print(f"chi squared SN: {chi_squared_sn_val:.2f}")
    print(f"chi squared quasars: {chi_squared_qsr_val:.2f}")
    print(f"chi squared total: {chi_squared_sn_val + chi_squared_qsr_val:.2f}")

    corner.corner(
        samples,
        labels=["$\\beta'$", "$s$", "$\\Delta_M$", "$\\Omega_m$", "$w_0$"],
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864), # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    z_plot = np.linspace(0.02, np.max(z), num=1000)
    theory = mu_model_sn(z_plot, best_fit_params)

    plt.errorbar(
        z,
        mu + factor * beta_50,
        yerr=np.sqrt(sigma_mu**2 + s_50**2),
        fmt='.',
        color='blue',
        label=legend,
        alpha=0.4,
        lw=0.5,
    )
    plt.plot(sn_z, sn_mu - deltaM_50, '.', color='red', alpha=0.5, label=sn_legend)
    plt.ylim(32.5, 54)
    plt.plot(z_plot, theory, color='green', label="$\mu_T$", alpha=0.8)
    plt.xlabel('Redshift ($z$)')
    plt.ylabel('$\mu$')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


"""
Flat ΛCDM
beta': -7.074 +0.012 -0.012
s: 0.613 +0.067 -0.056
ΔM: 0.020 +0.011 -0.011 mag
Omega_m: 0.347 +0.017 -0.017
w0: -1
chi squared SN: 1640.33
chi squared quasars: 52.43
chi squared total: 1692.76

==================================

Flat wCDM
beta': -7.071 +0.012 -0.012
s: 0.608 +0.065 -0.054
ΔM: 0.032 +0.013 -0.013 mag
Omega_m: 0.244 +0.077 -0.098
w0: -0.767 +0.136 -0.154
chi squared SN: 1639.12
chi squared quasars: 52.26
chi squared total: 1691.38

==================================

Flat wzCDM
beta': -7.071 +0.012 -0.012
s: 0.610 +0.065 -0.057
ΔM: 0.032 +0.013 -0.013 mag
Omega_m: 0.281 +0.049 -0.053
w0: -0.799 +0.117 -0.131
chi squared SN: 1638.74
chi squared quasars: 52.09
chi squared total: 1690.82
"""
