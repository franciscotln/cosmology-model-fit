import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2024DES.data import get_data

legend, z_values, distance_modulus_values, cov_matrix = get_data()

# Speed of light (km/s)
C = 299792.458

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)


# ΛCDM - flat
def integral_of_e_z(zs, omega_m):
    i = 0
    res = np.empty((len(zs),))
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([(1 / np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m))) for z in z_axis], x=z_axis)
        res[i] = integ
        i = i + 1
    return res

def lcdm_distance_modulus(z, h0, omega_m):
    normalized_h0 = 100 * h0 # (km/s/Mpc)
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(zs = z, omega_m=omega_m)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


# Distance modulus for modified matter-dominated, flat universe:
def model_distance_modulus(z, h0, p):
    normalized_h0 = 100 * h0 # (km/s/Mpc)
    a0_over_ae = (1 + z)**(1 / (1 - p))
    luminosity_distance = (2 * C * (1 - p) / normalized_h0) * (a0_over_ae - np.sqrt(a0_over_ae))
    return 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params, z, observed_mu):
    [h0, p] = params
    delta = observed_mu + 5 * np.log10(h0 / 0.70) - model_distance_modulus(z=z, h0=h0, p=p)
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params, z, observed_mu):
    return -0.5 * chi_squared(params, z, observed_mu)


h0_bounds = (0.001, 1.5)
p_bounds = (0.1, 0.6)


def log_prior(params):
    [h0, p] = params
    if h0_bounds[0] < h0 < h0_bounds[1] and p_bounds[0] < p < p_bounds[1]:
        return 0.0
    return -np.inf


def log_probability(params, z, observed_mu):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params, z, observed_mu)


def main():
    steps_to_discard = 100
    n_dim = 2
    n_walkers = 16
    n_steps = steps_to_discard + 1000

    initial_pos = np.zeros((n_walkers, n_dim))
    initial_pos[:, 0] = np.random.uniform(*h0_bounds, n_walkers)
    initial_pos[:, 1] = np.random.uniform(*p_bounds, n_walkers)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            log_probability,
            args=(z_values, distance_modulus_values),
            pool=pool
        )
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    # Extract samples
    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        effective_samples = n_steps * n_walkers / np.max(tau)
        print(f"Estimated autocorrelation time: {tau}")
        print(f"Effective samples: {effective_samples:.2f}")
    except Exception as e:
        print("Could not calculate the autocorrelation time")

    # Calculate the posterior means and uncertainties
    h0_samples = samples[:, 0]
    p_samples = samples[:, 1]

    one_sigma_quantile = np.array([16, 50, 84])
    [h0_16, h0_50, h0_84] = np.percentile(h0_samples, one_sigma_quantile)
    [p_16, p_50, p_84] = np.percentile(p_samples, one_sigma_quantile)

    corner.corner(
        samples,
        labels=[r"$h_0$", r"$p$"],
        truths=[h0_50, p_50],
        show_titles=True,
        title_fmt=".5f",
        title_kwargs={"fontsize": 12},
        quantiles=one_sigma_quantile/100,
    )
    plt.show()

    # Plot results: chains for each parameter
    fig, axes = plt.subplots(2, figsize=(10, 7))
    axes[0].plot(chains_samples[:, :, 0], color='black', alpha=0.3)
    axes[0].set_ylabel(r"$h_0$")
    axes[0].set_xlabel("chain step")
    axes[0].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
    axes[1].plot(chains_samples[:, :, 1], color='black', alpha=0.3)
    axes[1].set_ylabel(r"$p$")
    axes[1].set_xlabel("chain step")
    axes[1].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
    plt.show()

    # Calculate residuals
    predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0_50, p=p_50)
    residuals = distance_modulus_values + 5 * np.log10(h0_50 / 0.70) - predicted_distance_modulus_values

    # Compute skewness
    skewness = stats.skew(residuals)

    # Compute kurtosis
    kurtosis = stats.kurtosis(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(distance_modulus_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((distance_modulus_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    # Print the values in the console
    h0_label = f"{h0_50:.5f} +{h0_84-h0_50:.5f}/-{h0_50-h0_16:.5f}"
    p_label = f"{p_50:.5f} +{p_84-p_50:.5f}/-{p_50-p_16:.5f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("p", p_label)
    print_color("h0", h0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Chi squared", chi_squared([h0_50, p_50], z_values, distance_modulus_values))

    # Plot the data and the fit
    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=np.sqrt(cov_matrix.diagonal()),
        y_model=predicted_distance_modulus_values,
        label=f"Model: p = {p_label}",
        x_scale="log"
    )

    # Plot the residual analysis
    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=np.sqrt(cov_matrix.diagonal()),
        bins=40
    )

if __name__ == '__main__':
    main()

"""
Rescaled considering that H0 = 70 km/s/Mpc for the distance modulus data
5 * np.log10(h0 / 0.70)

==============================
Dataset: Union2.1
z range: 0.015 - 1.414
Sample size:  580
Chi squared:  547.465
p: 0.3409 +0.0194/-0.0207
H0: 69.40 +0.45/-0.46
R-squared (%): 99.29
RMSD (mag): 0.270
Skewness of residuals: 1.424
kurtosis of residuals: 8.377

==============================
Dataset: DES-SN5YR
z range: 0.025 - 1.121
Sample size: 1829
Chi squared:  1645.596
p: 0.3269 +0.0078/-0.0079
h0: 69.23 +0.17-0.17
R-squared (%): 98.39
RMSD (mag): 0.265
Skewness of residuals: 3.416
kurtosis of residuals: 25.863
"""
