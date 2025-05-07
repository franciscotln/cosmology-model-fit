from multiprocessing import Pool
import numpy as np
import emcee
from scipy.integrate import cumulative_trapezoid
import corner
import matplotlib.pyplot as plt
from y2018quasars.data import get_binned_data, factor
from y2005cc.data import get_data as get_cc_data

legend, z, mu, sigma_mu = get_binned_data(bin_size=30)
legend_cc, z_cc, H_values_cc, cov_mat_cc  = get_cc_data()
inv_cov_cc = np.linalg.inv(cov_mat_cc)

c = 299792.458 # Speed of light (km/s)

z_grid = np.linspace(0, np.max(z), num=3000)
one_plus_z = 1 + z_grid

def Ez(params):
    O_m, w0 = params[3], params[4]
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * ((2 * one_plus_z**2) / (1 + one_plus_z**2))**(3 * (1 + w0)))

def integral_Ez(z, params):
    integral_values = cumulative_trapezoid(1/Ez(params), z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)

def mu_model_sn(z, theta):
    H0 = theta[2]
    return 25 + 5 * np.log10((1 + z) * c / H0 * integral_Ez(z, theta))

def chi_squared_cc(theta):
    H0 = theta[2]
    delta = H_values_cc - H0 * np.interp(z_cc, z_grid, Ez(theta))
    return np.dot(delta, np.dot(inv_cov_cc, delta))

def chi_squared_quasar(theta):
    beta_prime, s = theta[0], theta[1]
    delta_quasars = mu + factor * beta_prime - mu_model_sn(z, theta)
    return np.sum(delta_quasars**2 / (sigma_mu**2 + s**2))

def log_likelihood(theta):
    s = theta[1]
    chi_squared_cc_val = chi_squared_cc(theta)
    chi_squared_val = chi_squared_quasar(theta)
    return -0.5 * chi_squared_cc_val -0.5 * (chi_squared_val + np.sum(np.log(sigma_mu**2 + s**2)))

def log_prior(theta):
    beta_prime, s, H0, omega_m, w0 = theta
    if (-8 < beta_prime < -6 and 0 < s < 3 and 50 < H0 < 100 and 0 < omega_m < 0.6 and -4.5 < w0 < 0.5):
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
    nsteps = 15000
    p0 = np.array([-7, 1, 70, 0.3, -1]) + 0.01 * np.random.randn(nwalkers, ndim)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=1000, flat=True)

    for i, param in enumerate(["beta'", "s", "H0", "Omega_m", "w0"]):
        mcmc_val = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{param}: {mcmc_val[1]:.3f} +{mcmc_val[2]-mcmc_val[1]:.3f} -{mcmc_val[1]-mcmc_val[0]:.3f}")

    beta_50 = np.median(samples[:, 0])
    s_50 = np.median(samples[:, 1])
    deltaM_50 = np.median(samples[:, 2])
    omega_m_50 = np.median(samples[:, 3])
    w0_50 = np.median(samples[:, 4])

    best_fit_params = [beta_50, s_50, deltaM_50, omega_m_50, w0_50]
    chi_squared_cc_val = chi_squared_cc(best_fit_params)
    chi_squared_qsr_val = chi_squared_quasar(best_fit_params)
    print(f"chi squared CC: {chi_squared_cc_val:.2f}")
    print(f"chi squared quasars: {chi_squared_qsr_val:.2f}")
    print(f"chi squared total: {chi_squared_cc_val + chi_squared_qsr_val:.2f}")

    corner.corner(
        samples,
        labels=["$\\beta'$", "$s$", "$H_0$", "$\\Omega_m$", "$w_0$"],
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

    theory = mu_model_sn(z, best_fit_params)
    observed_qsr = mu + factor * beta_50
    residuals = np.abs(observed_qsr - theory)
    err = np.sqrt(sigma_mu**2 + s_50**2)
    mask1 = residuals <= 1.0 * err
    mask2 = residuals > 1.0 * err

    print(f"Points within 1 sigma: {100 * np.sum(mask1) / (np.sum(mask1) + np.sum(mask2)):.1f}%")
    print(f"Points above 1 sigma: {100 * np.sum(mask2) / (np.sum(mask1) + np.sum(mask2)):.1f}%")

    plt.errorbar(
        z[mask1],
        observed_qsr[mask1],
        yerr=np.sqrt(sigma_mu[mask1]**2 + s_50**2),
        fmt='.',
        color='green',
        label=f"$<= 1\sigma$",
        alpha=0.6,
        lw=0.5,
    )
    plt.errorbar(
        z[mask2],
        observed_qsr[mask2],
        yerr=np.sqrt(sigma_mu[mask2]**2 + s_50**2),
        fmt='.',
        color='orange',
        label=f"$> 1\sigma$",
        alpha=0.5,
        lw=0.5,
    )
    plt.plot(z_plot, mu_model_sn(z_plot, best_fit_params), color='black', label="$\mu_T$", alpha=0.8)
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
beta': -7.061 +0.019 -0.019
s: 0.608 +0.067 -0.056
H0: 69.002 +5.118 -5.126
Omega_m: 0.299 +0.065 -0.052
w0: -1.996 +1.690 -1.689
chi squared CC: 14.66
chi squared quasars: 52.44
chi squared total: 67.10

===============================

Flat wCDM
beta': -7.058 +0.019 -0.018
s: 0.603 +0.067 -0.056
H0: 75.987 +11.868 -9.107
Omega_m: 0.270 +0.065 -0.057
w0: -1.748 +0.783 -0.941
chi squared CC: 15.80
chi squared quasars: 52.41
chi squared total: 68.20

==============================

Flat wzCDM
beta': -7.058 +0.019 -0.019
s: 0.604 +0.065 -0.056
H0: 77.467 +11.663 -9.819
Omega_m: 0.264 +0.064 -0.051
w0: -1.917 +0.844 -0.986
chi squared CC: 15.14
chi squared quasars: 52.05
chi squared total: 67.19
"""