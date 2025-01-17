import sys
sys.path.append('path/cosmology-model-fit')

import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
from multiprocessing import Pool
from plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data import get_data

legend, z_values, distance_modulus_values, cov_matrix = get_data()
sigma_distance_moduli = np.sqrt(cov_matrix.diagonal())

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Speed of light (km/s)
C = 299792.458


# Theoretical distance modulus for matter-dominated, flat universe:
def model_distance_modulus(z, h0, p):
    """
    fixed p=0.675 yields stable h0~0.72 at different z bins. p and h0 are degenerate, highly correlated with coefficient ~-0.84
    """
    normalized_h0 = 100 * h0 # (km/s/Mpc)
    a0_over_ae = (1 + z) ** (1 / p)
    comoving_distance = (2 * C * p/ normalized_h0) * (1 - 1 / np.sqrt(a0_over_ae))
    luminosity_distance = a0_over_ae * comoving_distance
    return 25 + 5 * np.log10(luminosity_distance)


# Define chi-squared function
def chi_squared(params, z, observed_mu):
    [h0, p] = params
    delta = observed_mu - model_distance_modulus(z=z, h0=h0, p=p)
    return delta.T @ inv_cov_matrix @ delta


# Log likelihood for MCMC sampling (negative chi-squared)
def log_likelihood(params, z, observed_mu):
    return -0.5 * chi_squared(params, z, observed_mu)


# Log prior function (uniform prior within bounds)
def log_prior(params):
    [h0, p] = params
    if 0.4 < h0 < 1 and 0.4 < p < 1:
        return 0.0  # Uniform prior
    return -np.inf  # Outside of bounds


# Log probability function (posterior = likelihood * prior)
def log_probability(params, z, observed_mu):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params, z, observed_mu)


def main():
    # Set up MCMC sampler
    n_dim = 2
    n_walkers = 20
    n_steps = 1200

    # Initial positions for walkers (random within bounds)
    initial_pos = np.random.rand(n_walkers, n_dim)
    initial_pos[:, 0] = initial_pos[:, 0] * 0.60 + 0.40  # h0 between 0.40 and 1.0
    initial_pos[:, 1] = initial_pos[:, 1] * 0.60 + 0.40  # p between 0.40 and 1.0

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
    samples = sampler.get_chain(discard=200, flat=True)

    tau = sampler.get_autocorr_time()
    effective_samples = n_steps * n_walkers / np.max(tau)
    print(f"Estimated autocorrelation time: {tau}")
    print(f"Effective samples: {effective_samples:.2f}")

    # Calculate the posterior means and uncertainties
    h0_samples = samples[:, 0]
    p_samples = samples[:, 1]

    [h0_16, h0_50, h0_84] = np.percentile(h0_samples, [16, 50, 84])
    [p_16, p_50, p_84] = np.percentile(p_samples, [16, 50, 84])

    h0 = h0_50
    p = p_50

    h0_std = np.std(h0_samples)
    p_std = np.std(p_samples)

    correlation_coefficient = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
    print(f"Correlation coefficient between h0 and p: {correlation_coefficient:.5f}")

    fig1 = corner.corner(
        samples,
        labels=["h0", "p"],
        truths=[h0, p],
        show_titles=True,
        title_fmt=".5f",
        title_kwargs={"fontsize": 12},
    )
    plt.show()

    # Plot results: chains for each parameter
    fig, axes = plt.subplots(2, figsize=(10, 7))
    axes[0].plot(chains_samples[:, :, 0], color='black', alpha=0.3)
    axes[0].set_ylabel(r"$h_0$")
    axes[1].plot(chains_samples[:, :, 1], color='black', alpha=0.3)
    axes[1].set_ylabel(r"$p$")
    plt.show()

    # Calculate residuals
    predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0, p=p)
    residuals = predicted_distance_modulus_values - distance_modulus_values

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
    h0_label = f"{h0:.5f} ± {h0_std:.5f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("Estimated h = H0 / 100 (km/s/Mpc)", h0_label)
    print_color("Estimated p", f"{p:.5f} ± {p_std:.5f}")
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")

    # Plot the data and the fit
    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=sigma_distance_moduli,
        y_model=predicted_distance_modulus_values,
        label=f"Model: h = {h0_label}",
        x_scale="log"
    )

    # Plot the residual analysis
    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=sigma_distance_moduli,
        bins=40
    )


    # correlation between h0 and p
    def nonlinear_model(p, k, a):
        return a + 2/(3*p**k)

    # params, covariance = curve_fit(nonlinear_model, p_samples, h0_samples)
    # [k, a] = params

    # print(f"Fitted relationship: k = {k:.4f} and a = {a:.4f}")
    # print("std", np.sqrt(np.diag(covariance)))

    # x = np.linspace(min(p_samples), max(p_samples), 100)
    # y = nonlinear_model(x, k=k, a=a)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(p_samples, h0_samples, alpha=0.5, s=10, label="Samples")
    # plt.plot(x, y, color="orange", label=f"Fit: k={k:.4f}")
    # plt.xlabel(r"$h_0$")
    # plt.ylabel(r"$p$")
    # plt.title("non-linear fit of $p$ vs $h_0$")
    # plt.legend()
    # plt.grid()
    # plt.show()

if __name__ == '__main__':
    main()
