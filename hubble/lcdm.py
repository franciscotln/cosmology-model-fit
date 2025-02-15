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


# ΛCDM
def integral_of_e_z(zs, omega_m, w):
    i = 0
    res = np.empty((len(zs),), dtype=np.float64)
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([(1 / np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m) * (1 + z)**(3 * (1 + w)))) for z in z_axis], x=z_axis)
        res[i] = integ
        i = i + 1
    return res

h0 = 70 # Hubble constant (km/s/Mpc)

def model_distance_modulus(z, w, omega_m):
    a0_over_ae = 1 + z
    comoving_distance = (C / h0) * integral_of_e_z(zs = z, omega_m=omega_m, w=w)
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params, z, observed_mu):
    [omega_m] = params
    delta = observed_mu - model_distance_modulus(z=z, w=-1, omega_m=omega_m)
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params, z, observed_mu):
    return -0.5 * chi_squared(params, z, observed_mu)


def log_prior(params):
    [omega_m] = params
    if 0.1 < omega_m < 0.5:
        return 0.0
    return -np.inf


def log_probability(params, z, observed_mu):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params, z, observed_mu)


def main():
    n_dim = 1
    n_walkers = 20
    n_steps = 1600

    initial_pos = np.random.rand(n_walkers, n_dim)
    initial_pos[:, 0] = initial_pos[:, 0] * 0.40 + 0.10  # omega_m between 0.1 and 0.5

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

    try:
        tau = sampler.get_autocorr_time()
        effective_samples = n_steps * n_walkers / np.max(tau)
        print(f"Estimated autocorrelation time: {tau}")
        print(f"Effective samples: {effective_samples:.2f}")
    except Exception as e:
        print("could not calculate the autocorrelation time")

    # Calculate the posterior means and uncertainties
    omega_m_samples = samples[:, 0]
    [omega_m_16, omega_m_50, omega_m_84] = np.percentile(omega_m_samples, [16, 50, 84])

    plt.hist(omega_m_samples, bins=40, color='blue', alpha=0.3, density=True)
    plt.axvline(x=omega_m_50, color='red', linestyle='--', label=r"$Ω_M$_true")
    plt.xlabel(r"$Ω_M$")
    plt.ylabel('Density')
    plt.title('Posterior Distribution')
    plt.legend()
    plt.show()

    # Plot results: chains for each parameter
    fig, axes = plt.subplots(1, figsize=(10, 7))
    axes.plot(chains_samples[:, :, 0], color='black', alpha=0.3)
    axes.set_ylabel(r"$Ω_M$")
    plt.show()

    # Calculate residuals
    predicted_distance_modulus_values = model_distance_modulus(z=z_values, w=-1, omega_m=omega_m_50)
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
    omega_label = f"{omega_m_50:.5f} +{omega_m_84 - omega_m_50:.5f} / -{omega_m_50 - omega_m_16:.5f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("Ωm", omega_label)
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
        label=f"Model: Ωm = {omega_label}",
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
Effective samples: 1450.22

Dataset:  Union2.1
z range:  0.015 - 1.414
Sample size:  580
Ωm:  0.28736 +0.03118 / -0.02969
R-squared (%):  99.30
RMSD (mag):  0.267
Skewness of residuals:  -1.380
kurtosis of residuals:  8.225
"""
