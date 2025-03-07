import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
import scipy.stats as stats
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data_shoes import get_data

legend, z_values, distance_modulus_values, cepheid_distances, cov_matrix = get_data()
sigma_distance_moduli = np.sqrt(cov_matrix.diagonal())

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Speed of light (km/s)
C = 299792.458


#  Flat fluid model
def integral_of_e_z(z, w0):
    z_grid = np.linspace(0, np.max(z), num=1000)
    e_inv = 1 / (np.exp(((1 - 3 * w0) / 2) * (-1 + 1/(1 + z_grid))) * (1 + z_grid) ** 2)
    integral_values = cumulative_trapezoid(e_inv, z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)


def fluid_distance_modulus(z, params):
    [h0, w0] = params
    normalized_h0 = 100 * h0
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(z, w0)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


# Flat ΛCDM
def lcdm_e_z(z, Omega_m):
    z_grid = np.linspace(0, np.max(z), num=1000)
    e_inv = 1 / np.sqrt(Omega_m * (1 + z_grid)**3 + 1 - Omega_m)
    integral_values = cumulative_trapezoid(e_inv, z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)


def lcdm_distance_modulus(z, params):
    [h0, omega_m] = params
    normalized_h0 = 100 * h0
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * lcdm_e_z(z, omega_m)
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params):
    mu_theory = np.where(cepheid_distances != -9, cepheid_distances, fluid_distance_modulus(z_values, params))
    delta = distance_modulus_values - mu_theory
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (0.4, 1.0), # h0
    (-1, 0), # w0
])


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    steps_to_discard = 100
    n_dim = len(bounds)
    n_walkers = 40
    n_steps = steps_to_discard + 4000
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    h0_samples = samples[:, 0]
    w0_samples = samples[:, 1]

    [h0_16, h0_50, h0_84] = np.percentile(h0_samples, [16, 50, 84])
    [w_16, w_50, w_84] = np.percentile(w0_samples, [16, 50, 84])

    best_fit_params = [h0_50, w_50]

    # Compute residuals
    predicted_distance_modulus_values = fluid_distance_modulus(z_values, best_fit_params)
    residuals = np.where(cepheid_distances != -9, distance_modulus_values - cepheid_distances , distance_modulus_values - predicted_distance_modulus_values)

    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Compute R-squared
    average_distance_modulus = np.mean(distance_modulus_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((distance_modulus_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compute root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    # Compute correlations
    spearman_corr, _ = stats.spearmanr(h0_samples, w0_samples)

    # Print the values in the console
    h0_label = f"{h0_50:.4f} +{h0_84-h0_50:.4f}/-{h0_50-h0_16:.4f}"
    w0_label = f"{w_50:.4f} +{w_84-w_50:.4f}/-{w_50-w_16:.4f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.4f} - {z_values[-1]:.4f}")
    print_color("Sample size", len(z_values))
    print_color("h = H0 / 100 (km/s/Mpc)", h0_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Spearman correlation", f"{spearman_corr:.3f}")
    print_color("Chi squared", chi_squared(best_fit_params))

    # Plot the data and the fit
    labels = [r"$h_0$", r"$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
        smooth=1.5,
        smooth1d=1.5,
    )
    plt.show()

    # Plot results: chains for each parameter
    fig, axes = plt.subplots(n_dim, figsize=(10, 7))
    if n_dim == 1:
        axes = [axes]
    for i in range(n_dim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit_params[i], color='white', linestyle='--', alpha=0.5)
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=sigma_distance_moduli,
        y_model=predicted_distance_modulus_values,
        label=f"H0={(100 * h0_50):.4f} km/s/Mpc, w0={w_50:.4f}",
        x_scale="log"
    )

    # Plot the residual analysis
    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=sigma_distance_moduli,
        bins=40
    )

if __name__ == '__main__':
    main()

"""
*****************************
Dataset: Pantheon+ and SH0ES
z range: 0.0012 - 2.2614
Sample size: 1657
*****************************

ΛCDM
H0: 73.23 +0.23/-0.23 km/s/Mpc
Ωm: 0.3307 +0.0178/-0.0176
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.086
kurtosis of residuals: 1.559
Spearman correlation: -0.823
Chi squared: 1452.7

=============================

wCDM
H0: 73.12 +0.32/-0.29 km/s/Mpc
Ωm: 0.3050 +0.0600/-0.0746
w: -0.9306 +0.1447/-0.1602
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.078
kurtosis of residuals: 1.565
Chi squared: 1452.5

=============================

Fluid model
H0: 73.33 +0.25/-0.25 km/s/Mpc
w0: -0.7049 +0.0228/-0.0234
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.093
kurtosis of residuals: 1.554
Spearman correlation: -0.845
Chi squared: 1453.3
"""
