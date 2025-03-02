import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
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


# Flat ΛCDM
def integral_of_e_z(zs, w0, wa):
    def integrand(z):
        radiation_limit = 1 / 3
        w_z = radiation_limit + (w0 - radiation_limit) * np.exp(-z * wa)
        return 1 / np.sqrt((1 + z) ** (3 * (1 + w_z)))

    return np.array([quad(integrand, 0, z_item)[0] for z_item in zs])


# Flat wCDM
def lcdm_distance_modulus(z, params):
    [h0, omega_m] = params
    normalized_h0 = 100 * h0
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(zs = z, omega_m=omega_m, w=-1)
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)


def wcdm_distance_modulus(z, params):
    [h0, w0, wa] = params
    normalized_h0 = 100 * h0
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(z, w0, wa)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def chi_squared(params):
    mu_theory = np.where(cepheid_distances != -9, cepheid_distances, wcdm_distance_modulus(z_values, params))
    delta = distance_modulus_values - mu_theory
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (0.4, 1.0), # h0
    (-2, 1), # w0
    (-0.5, 1), # wa
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
    n_walkers = 18
    n_steps = steps_to_discard + 1000
    initial_pos = np.zeros((n_walkers, n_dim))

    for dim, (lower, upper) in enumerate(bounds):
      initial_pos[:, dim] = np.random.uniform(lower, upper, n_walkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    h0_samples = samples[:, 0]
    p_samples = samples[:, 1]
    wa_samples = samples[:, 2]

    [h0_16, h0_50, h0_84] = np.percentile(h0_samples, [16, 50, 84])
    [p_16, p_50, p_84] = np.percentile(p_samples, [16, 50, 84])
    [wa_16, wa_50, wa_84] = np.percentile(wa_samples, [16, 50, 84])

    best_fit_params = [h0_50, p_50, wa_50]

    # Compute residuals
    predicted_distance_modulus_values = wcdm_distance_modulus(z_values, best_fit_params)
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
    spearman_corr, _ = stats.spearmanr(h0_samples, p_samples)

    # Print the values in the console
    h0_label = f"{h0_50:.4f} +{h0_84-h0_50:.4f}/-{h0_50-h0_16:.4f}"
    p_label = f"{p_50:.4f} +{p_84-p_50:.4f}/-{p_50-p_16:.4f}"
    wa_label = f"{wa_50:.4f} +{wa_84-wa_50:.4f}/-{wa_50-wa_16:.4f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.4f} - {z_values[-1]:.4f}")
    print_color("Sample size", len(z_values))
    print_color("Estimated h = H0 / 100 (km/s/Mpc)", h0_label)
    print_color("Estimated w0", p_label)
    print_color("Estimated wa", wa_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Spearman correlation", f"{spearman_corr:.3f}")
    print_color("Chi squared", chi_squared(best_fit_params))

    # Plot the data and the fit
    corner.corner(
        samples,
        labels=[r"$h_0$", r"$w_0$", r"$w_a$"],
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
    axes[0].plot(chains_samples[:, :, 0], color='black', alpha=0.3)
    axes[0].set_ylabel(r"$h_0$")
    axes[0].set_xlabel("chain step")
    axes[0].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
    axes[1].plot(chains_samples[:, :, 1], color='black', alpha=0.3)
    axes[1].set_ylabel(r"$w_0$")
    axes[1].set_xlabel("chain step")
    axes[1].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
    axes[2].plot(chains_samples[:, :, 2], color='black', alpha=0.3)
    axes[2].set_ylabel(r"$w_a$")
    axes[2].set_xlabel("chain step")
    axes[2].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=sigma_distance_moduli,
        y_model=predicted_distance_modulus_values,
        label=f"H0={(100 * h0_50):.4f} km/s/Mpc & w0={p_50:.4f}",
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
Estimated H0: 73.25 +0.22/-0.24 km/s/Mpc
Estimated Ωm: 0.3295 +0.0183/-0.0170
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.086
kurtosis of residuals: 1.558
Spearman correlation: -0.832
Chi squared: 1452.8

=============================

wCDM
Estimated H0: 73.12 +0.32/-0.29 km/s/Mpc
Estimated Ωm: 0.3050 +0.0600/-0.0746
Estimated w: -0.9306 +0.1447/-0.1602
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.078
kurtosis of residuals: 1.565
Chi squared: 1452.5

=============================

Fluid model
Estimated H0: 73.08 +0.28/-0.27 km/s/Mpc
Estimated w0: -0.6319 +0.0370/-0.0388
Estimated wa: 0.2210 +0.0770/-0.0689
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.074
kurtosis of residuals: 1.565
Chi squared: 1452.61
"""
