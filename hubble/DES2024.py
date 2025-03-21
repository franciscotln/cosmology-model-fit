import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.integrate import cumulative_trapezoid
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
def e_z(z, omega_m, w0):
    z_grid = np.linspace(0, np.max(z), num=1500)
    alpha = 4/(1 - 3*w0) # W(z=-1) == 1/3
    H_over_H0 = np.sqrt(omega_m*(1 + z_grid)**3 + (1 - omega_m) * (alpha*(1 + z_grid)/(alpha + z_grid))**4)
    integral_values = cumulative_trapezoid(1/H_over_H0, z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)

def wcdm_distance_modulus(z, params):
    a0_over_ae = 1 + z
    comoving_distance = (C / h0) * e_z(z, *params)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def lcdm_distance_modulus(z, params):
    [omega_m] = params
    a0_over_ae = 1 + z
    comoving_distance = (C / h0) * e_z(z, omega_m, -1)
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
    (0, 1), # Ωm
    (-3, 0.5), # w0
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
    nwalkers = 50
    n_steps = steps_to_discard + 2000
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

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

    [
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [16, 50, 84], axis=0).T

    best_fit_params = [omega_50, w0_50]

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
    omega_label = f"{omega_50:.4f} +{omega_84-omega_50:.4f}/-{omega_50-omega_16:.4f}"
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f}/-{w0_50-w0_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Chi squared", chi_squared(best_fit_params))

    # plot posterior distribution from samples
    labels = [f"$\Omega_M$", f"$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 12},
        smooth=1.5,
        smooth1d=1.5,
        bins=50,
    )
    plt.show()

    # Plot chains for each parameter
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
        label=f"$\Omega_m$={omega_label}",
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
w0: -1
R-squared (%): 99.30
RMSD (mag): 0.268
Skewness of residuals: 1.406
kurtosis of residuals: 8.236

==============================

Flat wCDM
Chi squared: 550.9743
Ωm: 0.3170 +0.0926/-0.1373
w0: -1.0515 +0.3141/-0.4072
R-squared (%): 99.30
RMSD (mag): 0.268
Skewness of residuals: 1.407
kurtosis of residuals: 8.213

==============================

Modified waw0CDM
Chi squared: 550.9693
Ωm: 0.3134 +0.0834/-0.1075
w0: -1.0595 +0.3134/-0.4038
R-squared (%): 99.30
RMSD (mag): 0.267
Skewness of residuals: 1.401
kurtosis of residuals: 8.208

********************************
Dataset: DES-SN5YR
z range: 0.025 - 1.121
Sample size: 1829
********************************

Flat ΛCDM
Chi squared: 1649.4584
Ωm: 0.3505 +0.0169/-0.0164
w0: -1
R-squared (%): 98.35
RMSD (mag): 0.268
Skewness of residuals: 3.409
kurtosis of residuals: 25.929

==============================

Flat wCDM
Chi squared: 1648.1532
Ωm: 0.2706 +0.0730/-0.0950
w0: -0.8115 +0.1455/-0.1619
R-squared (%): 98.32
RMSD (mag): 0.271
Skewness of residuals: 3.417
kurtosis of residuals: 25.959

==============================

Modified Flat waw0CDM
Chi squared: 1647.8959
Ωm: 0.2906 +0.0527/-0.0610
w0: -0.8243 +0.1240/-0.1420
R-squared (%): 98.32
RMSD (mag): 0.271
Skewness of residuals: 3.418
kurtosis of residuals: 25.968
"""
