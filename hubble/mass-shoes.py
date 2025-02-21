import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data import get_data

legend, z_values, distance_modulus_values, cov_matrix = get_data()
sigma_distance_moduli = np.sqrt(cov_matrix.diagonal())

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Speed of light (km/s)
C = 299792.458


# Theoretical distance modulus for matter-dominated, flat universe:
# Fixed p=0.325 yields stable h0~0.72 at different z bins. p and h0 are highly correlated
def model_distance_modulus(z, h0, p):
    normalized_h0 = 100 * h0 # (km/s/Mpc)
    a0_over_ae = (1 + z) ** (1 / (1 - p))
    comoving_distance = (2 * C * (1 - p)/ normalized_h0) * (1 - 1 / np.sqrt(a0_over_ae))
    luminosity_distance = a0_over_ae * comoving_distance
    return 25 + 5 * np.log10(luminosity_distance)


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
    if 0.4 < h0 < 1 and 0 < p < 0.6:
        return 0.0
    return -np.inf


# Log probability function (posterior = likelihood * prior)
def log_probability(params, z, observed_mu):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params, z, observed_mu)


def main():
    steps_to_discard = 100
    n_dim = 2
    n_walkers = 40
    n_steps = steps_to_discard + 2000
    initial_pos = np.zeros((n_walkers, n_dim))
    initial_pos[:, 0] = np.random.uniform(0.4, 1, n_walkers)
    initial_pos[:, 1] = np.random.uniform(0, 0.6, n_walkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            log_probability,
            args=(z_values, distance_modulus_values),
            pool=pool
        )
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    h0_samples = samples[:, 0]
    p_samples = samples[:, 1]

    [h0_16, h0_50, h0_84] = np.percentile(h0_samples, [16, 50, 84])
    [p_16, p_50, p_84] = np.percentile(p_samples, [16, 50, 84])

    # Compute residuals
    predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0_50, p=p_50)
    residuals = distance_modulus_values - predicted_distance_modulus_values

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
    h0_label = f"{h0_50:.5f} +{h0_84-h0_50:.5f}/-{h0_50-h0_16:.5f}"
    p_label = f"{p_50:.5f} +{p_84-p_50:.5f}/-{p_50-p_16:.5f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("Estimated h = H0 / 100 (km/s/Mpc)", h0_label)
    print_color("Estimated p", p_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Spearman correlation", f"{spearman_corr:.3f}")
    print_color("Chi squared", chi_squared([h0_50, p_50], z_values, distance_modulus_values))

    # Plot the data and the fit
    corner.corner(
        samples,
        labels=[r"$h_0$", r"$p$"],
        truths=[h0_50, p_50],
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
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

    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=sigma_distance_moduli,
        y_model=predicted_distance_modulus_values,
        label=f"H0={(100 * h0_50):.4f} km/s/Mpc & p={p_50:.4f}",
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
-- RESULTS WITH SHOES --
Estimated autocorrelation time: [17.30 21.28]
Effective samples: 3947.03
Spearman correlation: 0.836
Pearson correlation: 0.846
Chi squared: 1753.49

Dataset: Pantheon+SHOES
Redshift range: 0.001 - 2.261
Sample size: 1701
Estimated H0: 72.16 ± 0.23 km/s/Mpc
Estimated p: 0.325 ± 0.009
R-squared: 99.74 %
RMSD: 0.173 mag
Skewness of residuals: -0.005
kurtosis of residuals: 4.206
"""
