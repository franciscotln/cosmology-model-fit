from multiprocessing import Pool
import numpy as np
import emcee
from scipy.integrate import cumulative_trapezoid
import corner
import matplotlib.pyplot as plt
from y2018quasars.data import get_data, factor

legend, z, mu, sigma_mu = get_data()

c = 299792.458 # Speed of light (km/s)
H0 = 70 # Hubble constant (km/s/Mpc)

z_grid = np.linspace(0, np.max(z), num=3000)
one_plus_z = 1 + z_grid

def Ez(params):
    Omega_m = params[2] # fixing to LCDM because of inability to fit w0
    Ez = np.sqrt(Omega_m * one_plus_z**3 + (1 - Omega_m))
    return Ez

def integral_Ez(z, params):
    integral_values = cumulative_trapezoid(1/Ez(params), z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)

def mu_model_sn(z, theta):
    return 25 + 5 * np.log10((1 + z) * c / H0 * integral_Ez(z, theta))

def chi_squared_quasar(theta):
    beta_prime, s = theta[0], theta[1]
    mu_obs = mu + factor * beta_prime
    delta_quasars =  mu_obs - mu_model_sn(z, theta)
    return np.sum(delta_quasars**2 / (sigma_mu**2 + s**2))

def log_likelihood(theta):
    s = theta[1]
    return -0.5 * (chi_squared_quasar(theta) + np.sum(np.log(sigma_mu**2 + s**2)))

def log_prior(theta):
    beta_prime, s, omega_m = theta
    if (-8 < beta_prime < -6 and 0 < s < 3 and 0 < omega_m < 0.6):
        return 0.0
    return -np.inf

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

def main():
    ndim = 3
    nwalkers = 8 * ndim
    nsteps = 15000
    p0 = np.array([-7, 1, 0.3]) + 0.01 * np.random.randn(nwalkers, ndim)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=1000, flat=True)

    for i, param in enumerate(["beta'", "s", "Omega_m"]):
        mcmc_val = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{param}: {mcmc_val[1]:.3f} +{mcmc_val[2]-mcmc_val[1]:.3f} -{mcmc_val[1]-mcmc_val[0]:.3f}")

    beta_50 = np.median(samples[:, 0])
    s_50 = np.median(samples[:, 1])
    omega_m_50 = np.median(samples[:, 2])
    best_fit_params = [beta_50, s_50, omega_m_50]
    print(f"chi squared total: {chi_squared_quasar(best_fit_params):.2f}")

    corner.corner(
        samples,
        labels=["$\\beta'$", "$s$", "$\\Omega_m$"],
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

    z_plot = np.linspace(0.01, np.max(z), num=1000)

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
    plt.plot(z_plot, mu_model_sn(z_plot, best_fit_params), color='orange', label="$\mu_T$", alpha=0.8)
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
beta': -7.154 +0.016 -0.018
s: 1.711 +0.032 -0.031 mag
Omega_m: 0.251 +0.088 -0.067
w0: -1 (fixed)
H0: 70 km/s/Mpc (fixed)
chi squared total: 1600.89
reduced chi squared: 1600.89 / (1598 - 3) = 1.004
"""
