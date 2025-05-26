import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
import scipy.stats as stats
from scipy.linalg import cho_factor, cho_solve
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data_shoes import get_data

legend, z_values, z_hel_values, apparent_mag_values, cepheid_distances, cov_matrix = (
    get_data()
)

cepheids_mask = cepheid_distances != -9
sigma_distance_moduli = np.sqrt(cov_matrix.diagonal())
cho = cho_factor(cov_matrix)

c = 299792.458  # Speed of light (km/s)

z_grid = np.linspace(0, np.max(z_values), num=2500)
one_plus_z = 1 + z_grid


# Flat
def integral_E_z(params):
    O_m, w0 = params[2], params[3]
    O_de = 1 - O_m
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    H_over_H0 = np.sqrt(O_m * one_plus_z**3 + O_de * evolving_de)
    integral_values = cumulative_trapezoid(1 / H_over_H0, z_grid, initial=0)
    return np.interp(z_values, z_grid, integral_values)


def model_mu(params):
    h0 = params[1]
    luminosity_distance = (c / h0) * (1 + z_hel_values) * integral_E_z(params)
    return 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params):
    M = params[0]
    mu_theory = np.where(cepheids_mask, cepheid_distances, model_mu(params))
    apparent_mag_theory = mu_theory + M
    delta = apparent_mag_values - apparent_mag_theory
    return np.dot(delta, cho_solve(cho, delta))


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-20, -19),  # M
        (40, 100),  # h0
        (0, 1),  # Ωm
        (-2, 0),  # w0
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
    steps_to_discard = 200
    n_dim = len(bounds)
    n_walkers = n_dim * 16
    n_steps = steps_to_discard + 8000
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
        [M_16, M_50, M_84],
        [h0_16, h0_50, h0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit_params = [M_50, h0_50, omega_50, w0_50]

    predicted_mu_values = model_mu(best_fit_params)
    residuals = (
        apparent_mag_values
        - M_50
        - np.where(cepheids_mask, cepheid_distances, predicted_mu_values)
    )

    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Compute R-squared
    average_distance_modulus = np.mean(apparent_mag_values)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((apparent_mag_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compute root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    M_label = f"{M_50:.3f} +{M_84-M_50:.3f}/-{M_50-M_16:.3f}"
    h0_label = f"{h0_50:.2f} +{h0_84-h0_50:.2f}/-{h0_50-h0_16:.2f}"
    omega_label = f"{omega_50:.3f} +{omega_84-omega_50:.3f}/-{omega_50-omega_16:.3f}"
    w0_label = f"{w0_50:.3f} +{w0_84-w0_50:.3f}/-{w0_50-w0_16:.3f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.4f} - {z_values[-1]:.4f}")
    print_color("Sample size", len(z_values))
    print_color("M", M_label)
    print_color("H0 (km/s/Mpc)", h0_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit_params):.2f}")

    labels = ["M", "$H_0$", "$\Omega_M$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
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
        y=apparent_mag_values - M_50,
        y_err=sigma_distance_moduli,
        y_model=predicted_mu_values,
        label=f"H0={h0_50:.2f} km/s/Mpc, w0={w0_50:.4f}",
        x_scale="log",
    )

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
M: -19.2437 +0.0292/-0.0297
H0 (km/s/Mpc): 73.5 +- 1.0
Ωm: 0.332 +0.018/-0.018
w0: -1
wa: 0
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.085
kurtosis of residuals: 1.555
Chi squared: 1452.02

=============================

wCDM
M: -19.243 +0.029/-0.029
H0 (km/s/Mpc): 73.46 +1.02/-1.00
Ωm: 0.298 +0.062/-0.077
w0: -0.91 +0.14/-0.16
wa: 0
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.076
kurtosis of residuals: 1.561
Chi squared: 1451.69

=============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
M: -19.243 +0.030/-0.030
H0 (km/s/Mpc): 73.5 +- 1.0
Ωm: 0.312 +0.047/-0.050
w0: -0.93 +0.12/-0.15
wa: 0
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.077
kurtosis of residuals: 1.561
Chi squared: 1451.71
"""
