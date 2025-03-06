import emcee
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2024DES.data import get_data

legend, z_values, distance_modulus_values, cov_matrix = get_data()

# Speed of light (km/s)
C = 299792.458

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Hubble constant km/s/Mpc as assumed in the dataset
h0 = 70

# Flat
def integral_of_e_z(zs, params):
    [w0] = params
    def integrand(z):
        correction = np.exp(((1 - 3 * w0) / 2) * (-1 + 1/(1 + z)))
        return 1 / (correction * (1 + z) ** 2)

    return np.array([quad(integrand, 0, z_item)[0] for z_item in zs])


def wcdm_distance_modulus(z, params):
    a0_over_ae = 1 + z
    comoving_distance = (C / h0) * integral_of_e_z(z, params)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


# Flat ΛCDM
def lcdm_e_z(zs, params):
    [omega_m] = params
    def integrand(z):
        return 1 / np.sqrt(omega_m * (1 + z)**3 + 1 - omega_m)
    return np.array([quad(integrand, 0, z_item)[0] for z_item in zs])


def lcdm_distance_modulus(z, params):
    a0_over_ae = 1 + z
    comoving_distance = (C / h0) * lcdm_e_z(z, params)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def chi_squared(params):
    """ 
    Computes modified likelihood to marginalize over M 
    (Wood-Vasey et al. 2001, Appendix A9-A12)
    """
    delta = distance_modulus_values - wcdm_distance_modulus(z_values, params)
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov_matrix @ deltaT)      # First term: (Δ^T C^-1 Δ)
    B = np.sum(delta @ inv_cov_matrix)                   # Second term: B
    C = np.sum(inv_cov_matrix)                           # Third term: C
    chi2 = chit2 - (B**2 / C) + np.log(C / (2 * np.pi))  # Full modified chi2
    return chi2


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (-1.2, 0), # w0
])


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    else:
        return -np.inf


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    steps_to_discard = 100
    ndim = len(bounds)
    nwalkers = 20
    n_steps = steps_to_discard + 2000
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
      initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Effective number of independent samples", len(samples) / tau)
    except Exception as e:
        print("Autocorrelation time could not be computed")

    w0_samples = samples[:, 0]

    one_sigma_quantile = np.array([16, 50, 84])
    [w0_16, w0_50, w0_84] = np.percentile(w0_samples, one_sigma_quantile)

    best_fit_params = [w0_50]

    predicted_distance_modulus_values = wcdm_distance_modulus(z_values, best_fit_params)
    residuals = distance_modulus_values - predicted_distance_modulus_values

    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(distance_modulus_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((distance_modulus_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    # Print the values in the console
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f}/-{w0_50-w0_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Chi squared", chi_squared(best_fit_params))

    # plot posterior distribution from samples
    fig, ax = plt.subplots()
    ax.hist(w0_samples, bins=50, density=True, alpha=0.6, color='g')
    mu, std = np.mean(w0_samples), np.std(w0_samples)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.axvline(x=w0_50, color='red', linestyle='--', alpha=0.8, label="Median (50%)")
    ax.fill_betweenx(y=[0, max(p)], x1=w0_16, x2=w0_84, color='red', alpha=0.2, label="68% CI")
    ax.legend()
    plt.show()

    # Plot chains for each parameter
    labels = [r"$w_0$"]
    fig, axes = plt.subplots(ndim, figsize=(10, 7))
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit_params[i], color='white', linestyle='--', alpha=0.5)
    plt.show()

    y_err = np.sqrt(cov_matrix.diagonal())

    # Plot the predictions
    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=y_err,
        y_model=predicted_distance_modulus_values,
        label=f"w0={w0_label}",
        x_scale="log"
    )

    # Plot the residual analysis
    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=y_err,
        bins=40
    )

if __name__ == '__main__':
    main()

"""
********************************
Dataset: Union2.1
z range: 0.015 - 1.414
Sample size: 580
********************************

Flat ΛCDM
Chi squared: 550.9463
Ωm: 0.2993 +0.0427/-0.0401
R-squared (%): 99.30
RMSD (mag): 0.268
Skewness of residuals: 1.406
kurtosis of residuals: 8.236

********************************
Dataset: DES-SN5YR
z range: 0.025 - 1.121
Sample size: 1829
********************************

Flat ΛCDM
Chi squared: 1649.4738
Ωm: 0.3520 +0.0169/-0.0170
R-squared (%): 98.35
RMSD (mag): 0.268
Skewness of residuals: 3.408
kurtosis of residuals: 25.913

==============================

Flat wCDM
Chi squared: 1648.1531
Ωm: 0.2769 +0.0722/-0.0915
w: -0.8236 +0.1450/-0.1692
R-squared (%): 98.32
RMSD (mag): 0.270
Skewness of residuals: 3.416
kurtosis of residuals: 25.958

==============================

Fluid model
Chi squared: 1650.7003
w0: -0.6807 +0.0215/-0.0216
R-squared (%): 98.36
RMSD (mag): 0.267
Skewness of residuals: 3.404
kurtosis of residuals: 25.883
"""
