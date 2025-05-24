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


# Flat
def integral_of_e_z(z, params):
    _, Omega_m, w0, _ = params
    z_grid = np.linspace(0, np.max(z), num=2000)
    sum = 1 + z_grid
    H_over_H0 = np.sqrt(
        Omega_m * sum**3
        + (1 - Omega_m) * ((2 * sum**2) / (1 + sum**2)) ** (3 * (1 + w0))
    )
    integral_values = cumulative_trapezoid(1 / H_over_H0, z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)


def wcdm_distance_modulus(z, params):
    h0 = params[0]
    normalized_h0 = 100 * h0
    comoving_distance = (C / normalized_h0) * integral_of_e_z(z, params)
    luminosity_distance = comoving_distance * (1 + z)
    return 25 + 5 * np.log10(luminosity_distance)


def lcdm_distance_modulus(z, params):
    [h0, Omega_m] = params
    normalized_h0 = 100 * h0
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(z, [h0, Omega_m, -1, 0])
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params):
    mu_theory = np.where(
        cepheid_distances != -9,
        cepheid_distances,
        wcdm_distance_modulus(z_values, params),
    )
    delta = distance_modulus_values - mu_theory
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (0.4, 1.0),  # h0
        (0, 1),  # Ωm
        (-2, 0),  # w0
        (-3, 3),  # wa
    ]
)


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
    n_walkers = 100
    n_steps = steps_to_discard + 4000
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Average autocorrelation time", np.mean(tau))
    except:
        print_color("Autocorrelation time", "Not available")

    [
        [h0_16, h0_50, h0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit_params = [h0_50, omega_50, w0_50, wa_50]

    # Compute residuals
    predicted_distance_modulus_values = wcdm_distance_modulus(z_values, best_fit_params)
    residuals = np.where(
        cepheid_distances != -9,
        distance_modulus_values - cepheid_distances,
        distance_modulus_values - predicted_distance_modulus_values,
    )

    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Compute R-squared
    average_distance_modulus = np.mean(distance_modulus_values)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((distance_modulus_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compute root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    # Print the values in the console
    h0_label = f"{h0_50:.4f} +{h0_84-h0_50:.4f}/-{h0_50-h0_16:.4f}"
    omega_label = f"{omega_50:.4f} +{omega_84-omega_50:.4f}/-{omega_50-omega_16:.4f}"
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f}/-{w0_50-w0_16:.4f}"
    wa_label = f"{wa_50:.4f} +{wa_84-wa_50:.4f}/-{wa_50-wa_16:.4f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.4f} - {z_values[-1]:.4f}")
    print_color("Sample size", len(z_values))
    print_color("h = H0 / 100 (km/s/Mpc)", h0_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("wa", wa_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Chi squared", chi_squared(best_fit_params))

    # Plot the data and the fit
    labels = [r"$h_0$", f"$\Omega_M$", r"$w_0$", r"$w_a$"]
    corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
        smooth=1.5,
        smooth1d=1.5,
        bins=50,
    )
    plt.show()

    # Plot results: chains for each parameter
    _, axes = plt.subplots(n_dim, figsize=(10, 7))
    if n_dim == 1:
        axes = [axes]
    for i in range(n_dim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=steps_to_discard, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit_params[i], color="white", linestyle="--", alpha=0.5)
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=sigma_distance_moduli,
        y_model=predicted_distance_modulus_values,
        label=f"H0={(100 * h0_50):.4f} km/s/Mpc, w0={w0_50:.4f}",
        x_scale="log",
    )

    # Plot the residual analysis
    plot_residuals(
        z_values=z_values, residuals=residuals, y_err=sigma_distance_moduli, bins=40
    )


if __name__ == "__main__":
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
w0: -1
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.086
kurtosis of residuals: 1.559
Chi squared: 1452.7

=============================

wCDM
H0: 73.12 +0.32/-0.29 km/s/Mpc
Ωm: 0.3050 +0.0600/-0.0746
w0: -0.9306 +0.1447/-0.1602
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.078
kurtosis of residuals: 1.565
Chi squared: 1452.5

=============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 73.14 +0.31/-0.30 km/s/Mpc
Ωm: 0.3144 +0.0478/-0.0500
w0: -0.9467 +0.1288/-0.1489
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.079
kurtosis of residuals: 1.564
Chi squared: 1452.5
"""
