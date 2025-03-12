import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.integrate import cumulative_trapezoid
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2023union3.data import get_data

legend, z_values, distance_moduli_values, cov_matrix = get_data()

inverse_cov = np.linalg.inv(cov_matrix)

# Speed of light (km/s)
C = 299792.458

# Flat model
def integral_of_e_z(zs, Omega_m, w0):
    z_grid = np.linspace(0, np.max(zs), num=1500)
    alpha = 4/(1 - 3*w0) # W(z=-1) == 1/3
    H_over_H0 = np.sqrt(Omega_m*(1 + z_grid)**3 + (1 - Omega_m) * (alpha*(1 + z_grid)/(alpha + z_grid))**4)
    integral_values = cumulative_trapezoid(1/H_over_H0, z_grid, initial=0)
    return np.interp(zs, z_grid, integral_values)


def wcdm_distance_modulus(z, params):
    [h0, Omega_m, w0] = params
    normalized_h0 = h0 * 100
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(zs=z, Omega_m=Omega_m, w0=w0)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


# Flat ΛCDM
def lcdm_distance_modulus(z, params):
    [h0, Omega_m] = params
    normalized_h0 = h0 * 100
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(zs=z, Omega_m=Omega_m, w0=-1)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def chi_squared(params):
    delta = distance_moduli_values - wcdm_distance_modulus(z_values, params)
    return delta.T @ inverse_cov @ delta


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (0.4, 0.9), # h0
    (0, 0.7),
    (-1.5, 0), # w0
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
    n_walkers = 200
    n_steps = discarded_steps + 2000
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=discarded_steps, flat=True)

    [
        [h0_16, h0_50, h0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84], 
    ] = np.percentile(samples, [16, 50, 84], axis=0).T

    best_fit_params = [h0_50, omega_50, w0_50]

    predicted_mag = wcdm_distance_modulus(z_values, best_fit_params)
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
    omega_label = f"{omega_50:.4f} +{omega_84-omega_50:.4f}/-{omega_50-omega_16:.4f}"
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f}/-{w0_50-w0_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("h0", h0_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurt:.3f}")
    print_color("Reduced chi squared", chi_squared(best_fit_params)/ (len(z_values) - 2))

    # Plot the data and the fit
    labels = [r"$h_0$", f"$\Omega_m$", r"$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
        smooth=1.5,
        smooth1d=1.5,
        bins=50
    )
    plt.show()

    # Plot results: chains for each parameter
    fig, axes = plt.subplots(n_dim, figsize=(10, 7))
    if n_dim == 1:
        axes = [axes]
    for i in range(n_dim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=discarded_steps, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit_params[i], color='white', linestyle='--', alpha=0.5)
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=predicted_mag,
        label=f"Best fit: $w_0$={w0_50:.4f}, $h_0$={h0_50:.4f}",
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

ΛCDM

Ωm: 0.3571 +0.0274/-0.0264
w0: -1
H0: 72.42 +3.00/-2.88 km/s/Mpc
R-squared (%): 99.95
RMSD (mag): 0.048
Skewness of residuals: 0.587
kurtosis of residuals: 0.692
Reduced chi squared: 1.20

=============================

wCDM

Ωm: 0.2539 +0.0873/-0.1107
w0: -0.7494 +0.1560/-0.1857
H0: 72.04 +3.00/-2.92 km/s/Mpc
R-squared (%): 99.94
RMSD (mag): 0.053
Skewness of residuals: -1.230
kurtosis of residuals: 3.852
Reduced chi squared: 1.11

=============================

Modified waw0CDM

Ωm: 0.2801 +0.0666/-0.0769
w0: -0.7588 +0.1474/-0.1770
H0: 71.98 +2.97/-2.86
R-squared (%): 99.94
RMSD (mag): 0.052
Skewness of residuals: -1.148
kurtosis of residuals: 3.571
Reduced chi squared: 1.10
"""
