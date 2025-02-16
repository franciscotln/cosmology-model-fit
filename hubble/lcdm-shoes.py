import sys
sys.path.append('/Users/francisco.neto/Documents/private/cosmology-model-fit')

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


# ΛCDM
def integral_of_e_z(zs, omega_m):
    i = 0
    res = np.empty((len(zs),), dtype=np.float64)
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([(1 / np.sqrt(omega_m * (1 + z) ** 3 + 1 - omega_m)) for z in z_axis], x=z_axis)
        res[i] = integ
        i = i + 1
    return res


# Define a theoretical distance modulus:
def model_distance_modulus(z, h0, omega_m):
    normalized_h0 = 100 * h0
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(zs = z, omega_m=omega_m)
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)

# Define chi-squared function
def chi_squared(params, z, observed_mu):
    h0, omega_m = params
    delta = observed_mu - model_distance_modulus(z, h0, omega_m)
    return delta.T @ inv_cov_matrix @ delta

# Log likelihood for MCMC sampling (negative chi-squared)
def log_likelihood(params, z, observed_mu):
    return -0.5 * chi_squared(params, z, observed_mu)

# Log prior function (uniform prior within bounds)
def log_prior(params):
    h0, omega_m = params
    if 0.65 < h0 < 0.8 and 0 < omega_m < 1:
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
    n_walkers = 10
    n_steps = 1050

    # Initial positions for walkers (random within bounds)
    initial_pos = np.random.rand(n_walkers, n_dim)
    initial_pos[:, 0] = initial_pos[:, 0] * 0.15 + 0.65  # h0 between 0.65 and 0.8
    initial_pos[:, 1] = initial_pos[:, 1] * 1.00 + 0.00  # omega_m between 0 and 1.0

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability,
            args=(z_values, distance_modulus_values),
            pool=pool
        )
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    # Extract samples
    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=50, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        effective_samples = n_steps * n_walkers / np.max(tau)
        print(f"Estimated autocorrelation time: {tau}")
        print(f"Effective samples: {effective_samples:.2f}")
    except Exception as e:
        print("could not calculate the autocorrelation time")

    # Calculate the posterior means and uncertainties
    h0_samples = samples[:, 0]
    omega_m_samples = samples[:, 1]

    [h0_16, h0_50, h0_84] = np.percentile(h0_samples, [16, 50, 84])
    [omega_m_16, omega_m_50, omega_m_84] = np.percentile(omega_m_samples, [16, 50, 84])

    h0 = h0_50
    omega_m = omega_m_50

    h0_std = np.std(h0_samples)
    omega_m_std = np.std(omega_m_samples)

    correlation_coefficient = np.corrcoef(np.log(samples[:, 0]), np.log(samples[:, 1]))[0, 1]
    print(f"Correlation coefficient between h0 and omega_m: {correlation_coefficient:.5f}")

    fig1 = corner.corner(
        samples,
        labels=["h0", "omega_m"],
        truths=[h0, omega_m],
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
    axes[1].set_ylabel(r"$omega_m$")
    plt.show()

    # Calculate residuals
    predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0, omega_m=omega_m)
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
    h0_label = f"{h0:.5f} ± {h0_std:.5f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("Estimated h = H0 / 100 (km/s/Mpc)", h0_label)
    print_color("omega_m", f"{omega_m:.5f} ± {omega_m_std:.5f}")
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

if __name__ == '__main__':
    main()

"""
-- RESULTS WITH SHOES - LCDM --
Estimated autocorrelation time: [28.29 27.12]
Effective samples: 724
Pearson correlation: -0.86033
Chi squared:  1752.51

Dataset:  Pantheon+SHOES
z range:  0.001 - 2.261
Sample size:  1701
Estimated h = H0 / 100 (km/s/Mpc):  0.72841 ± 0.00230
omega_m:  0.36127 ± 0.01863
R-squared (%):  99.74
RMSD (mag):  0.172
Skewness of residuals:  0.108
kurtosis of residuals:  4.308

-- RESULTS WITHOUT SHOES - LCDM --
Estimated autocorrelation time: [27.19 22.39]
Effective samples: 735.60
Pearson correlation: -0.84528

Dataset:  Pantheon+
z range:  0.010 - 2.261
Sample size:  1590
Estimated h = H0 / 100 (km/s/Mpc):  0.73235 ± 0.00246
omega_m:  0.33167 ± 0.01842
R-squared (%):  99.74
RMSD (mag):  0.154
Skewness of residuals:  -0.091
kurtosis of residuals:  1.584
"""
