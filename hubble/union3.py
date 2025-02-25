import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.integrate import quad
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2023union3.data import get_data

legend, z_values, distance_moduli_values, cov_matrix = get_data()

inverse_cov = np.linalg.inv(cov_matrix)

# Speed of light (km/s)
C = 299792.458

# Flat ΛCDM
def integral_of_e_z(zs, omega_m, w):
    def integrand(z):
        return 1 / np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m) * (1 + z) ** (3 * (1 + w)))

    return np.array([quad(integrand, 0, z_item)[0] for z_item in zs])


def wcdm_distance_modulus(z, params):
    [h0, omega_m, w] = params
    normalized_h0 = h0 * 100
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(zs=z, omega_m=omega_m, w=w)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def lcdm_distance_modulus(z, params):
    [h0, omega_m] = params
    normalized_h0 = h0 * 100
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(zs=z, omega_m=omega_m, w=-1)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)

# Distance modulus for alternative matter-dominated, flat universe:
def distance_modulus(z, params):
    [h0, p] = params
    normalized_h0 = h0 * 100
    a0_over_ae = (1 + z)**(1 / (1 - p))
    luminosity_distance = 2 * (C / normalized_h0) * (1 - p) * (a0_over_ae - np.sqrt(a0_over_ae))
    return 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params):
    delta = distance_moduli_values - distance_modulus(z_values, params)
    return delta.T @ inverse_cov @ delta


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (0.4, 0.9), # h0
    (0.1, 0.6), # p
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
    discarded_steps = 100
    n_dim = len(bounds)
    n_walkers = 400
    n_steps = discarded_steps + 1250
    initial_pos = np.zeros((n_walkers, n_dim))

    for dim, (lower, upper) in enumerate(bounds):
      initial_pos[:, dim] = np.random.uniform(lower, upper, n_walkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=discarded_steps, flat=True)

    h0_samples = samples[:, 0]
    p_samples = samples[:, 1]

    [h0_16, h0_50, h0_84] = np.percentile(h0_samples, [16, 50, 84])
    [p_16, p_50, p_84] = np.percentile(p_samples, [16, 50, 84])

    best_fit_params = [h0_50, p_50]

    predicted_mag = distance_modulus(z_values, best_fit_params)
    residuals = distance_moduli_values - predicted_mag

    skewness = skew(residuals)
    kurt = kurtosis(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(distance_moduli_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((distance_moduli_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    # Print the values in the console
    h0_label = f"{h0_50:.4f} +{h0_84-h0_50:.4f}/-{h0_50-h0_16:.4f}"
    p_label = f"{p_50:.4f} +{p_84-p_50:.4f}/-{p_50-p_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("h0", h0_label)
    print_color("p", p_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurt:.3f}")
    print_color("Reduced chi squared", chi_squared(best_fit_params)/ (len(z_values) - 2))

    # Plot the data and the fit
    corner.corner(
        samples,
        labels=[r"$h_0$", r"$p$"],
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
    axes[0].axvline(x=discarded_steps, color='red', linestyle='--', alpha=0.5)
    axes[1].plot(chains_samples[:, :, 1], color='black', alpha=0.3)
    axes[1].set_ylabel(r"$p$")
    axes[1].set_xlabel("chain step")
    axes[1].axvline(x=discarded_steps, color='red', linestyle='--', alpha=0.5)
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=predicted_mag,
        label=f"Best fit: $p$={p_50:.4f} & $h_0$={h0_50:.4f}",
        x_scale="log"
    )

    # Plot the residual analysis
    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=np.sqrt(np.diag(cov_matrix)),
        bins=40
    )

if __name__ == '__main__':
    main()

"""
*****************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*****************************

Alternative

p: 0.3103 +0.0127/-0.0133
h0: 0.7150 +0.0299/-0.0285
R-squared (%): 99.87
RMSD (mag): 0.079
Skewness of residuals: -3.298
kurtosis of residuals: 11.505
Reduced chi squared: 1.300

=============================

ΛCDM

Ωm: 0.3568 +0.0276/-0.0262
h0: 0.7239 +0.0301/-0.0287
R-squared (%): 99.95
RMSD (mag): 0.049
Skewness of residuals: 0.578
kurtosis of residuals: 0.691
Reduced chi squared: 1.198

=============================

wCDM

Ωm: 0.2526 +0.0884/-0.1101
h0: 0.7201 +0.0301/-0.0285
w: -0.7477 +0.1553/-0.1865
R-squared (%): 99.94
RMSD (mag): 0.053
Skewness of residuals: -1.257
kurtosis of residuals: 3.921
Reduced chi squared: 1.107
"""
