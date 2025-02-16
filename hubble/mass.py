import sys
sys.path.append('/Users/francisco.neto/Documents/private/cosmology-model-fit')

import emcee
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from multiprocessing import Pool
from plotting import plot_predictions, print_color, plot_residuals
from y2011union.data import get_data

legend, z_values, distance_modulus_values, _, cov_matrix = get_data()

# Speed of light (km/s)
C = 299792.458

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)


# Theoretical distance modulus for matter-dominated, flat universe:
def model_distance_modulus(z, h0, p):
    a0_over_ae = (1 + z)**(1/(1-p))
    comoving_distance = (2 * C * (1-p) / h0) * (1 - 1 / np.sqrt(a0_over_ae))
    luminosity_distance = a0_over_ae * comoving_distance
    return 25 + 5 * np.log10(luminosity_distance)

h0 = 70 # Hubble constant (km/s/Mpc)

def chi_squared(params, z, observed_mu):
    [p] = params
    delta = observed_mu - model_distance_modulus(z=z, h0=h0, p=p)
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params, z, observed_mu):
    return -0.5 * chi_squared(params, z, observed_mu)


def log_prior(params):
    [p] = params
    if 0.2 < p < 0.5:
        return 0.0
    return -np.inf


def log_probability(params, z, observed_mu):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params, z, observed_mu)


def main():
    n_dim = 1
    n_walkers = 50
    n_steps = 4100

    initial_pos = np.random.rand(n_walkers, n_dim)
    initial_pos[:, 0] = initial_pos[:, 0] * 0.30 + 0.20  # p between 0.2 and 0.5

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
    samples = sampler.get_chain(discard=100, flat=True)

    tau = sampler.get_autocorr_time()
    effective_samples = n_steps * n_walkers / np.max(tau)
    print(f"Estimated autocorrelation time: {tau}")
    print(f"Effective samples: {effective_samples:.2f}")

    # Calculate the posterior means and uncertainties
    p_samples = samples[:, 0]

    [p_16, p_50, p_84] = np.percentile(p_samples, [16, 50, 84])

    plt.hist(p_samples, bins=40, color='blue', alpha=0.3, density=True)
    plt.axvline(x=p_50, color='red', linestyle='--', label=r"$p$_true")
    plt.xlabel(r"$p$")
    plt.ylabel('Density')
    plt.title('Posterior Distribution')
    plt.legend()
    plt.show()

    # Plot results: chains for each parameter
    fig, axes = plt.subplots(1, figsize=(10, 7))
    axes.plot(chains_samples[:, :, 0], color='blue', alpha=0.2)
    axes.set_ylabel(r"$p$")
    plt.show()

    # Calculate residuals
    predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0, p=p_50)
    residuals = distance_modulus_values - predicted_distance_modulus_values

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
    p_label = f"{p_50:.5f}  +{p_84-p_50:.5f}/-{p_50-p_16:.5f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("p", p_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")

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
Dataset:  Union2.1
z range:  0.015 - 1.414
Sample size:  580

Effective chain samples: 8242
Chi squared:  549.09

p:  0.3579  +0.0139/-0.0145
R-squared (%):  99.28
RMSD (mag):  0.272
Skewness of residuals:  1.343
kurtosis of residuals:  8.323

==============================
Dataset:  DES-SN5YR
z range:  0.025 - 1.121
Sample size:  1757

Effective chain samples: 4778
Chi squared: 1664.81
p:  0.35831  +0.00269/-0.00279
R-squared (%):  99.32
RMSD (mag):  0.174
Skewness of residuals:  0.148
kurtosis of residuals:  0.670
"""
