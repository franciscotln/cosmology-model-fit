import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data import get_data

legend, z_values, z_hel_values, apparent_mag_values, cov_matrix = get_data()

C = 299792.458  # Speed of light (km/s)
H0 = 70.0  # Hubble constant (km/s/Mpc)

cho = cho_factor(cov_matrix)
z_grid = np.linspace(0, np.max(z_values), num=1500)
one_plus_z = 1 + z_grid


def integral_E_z(params):
    O_m, w0 = params[1], params[2]
    O_de = 1 - O_m
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))
    Ez = np.sqrt(O_m * one_plus_z**3 + O_de * rho_de)
    integral_values = cumulative_trapezoid(1 / Ez, z_grid, initial=0)
    return np.interp(z_values, z_grid, integral_values)


# Flat
def model_apparent_mag(params):
    M = params[0]  # absolute magnitude
    a0_over_ae = 1 + z_hel_values
    luminosity_distance = a0_over_ae * (C / H0) * integral_E_z(params)
    return M + 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params):
    delta = apparent_mag_values - model_apparent_mag(params)
    return np.dot(delta, cho_solve(cho, delta))


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-20, -19),  # M
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
        [M0_16, M0_50, M0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [16, 50, 84], axis=0).T

    best_fit_params = [M0_50, omega_50, w0_50]

    # Calculate residuals
    predicted_apparent_mag = model_apparent_mag(best_fit_params)
    residuals = apparent_mag_values - predicted_apparent_mag

    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(apparent_mag_values)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((apparent_mag_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    # Print the values in the console
    M_label = f"{M0_50:.3f} +{M0_84-M0_50:.3f}/-{M0_50-M0_16:.3f}"
    omega_label = f"{omega_50:.3f} +{omega_84-omega_50:.3f}/-{omega_50-omega_16:.3f}"
    w0_label = f"{w0_50:.3f} +{w0_84-w0_50:.3f}/-{w0_50-w0_16:.3f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.4f} - {z_values[-1]:.4f}")
    print_color("Sample size", len(z_values))
    print_color("M", M_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color(
        "Reduced chi squared",
        f"{chi_squared(best_fit_params) / (len(z_values) - len(best_fit_params)):.4f}",
    )

    labels = ["$M_0$", "$\Omega_m$", "$w_0$"]
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

    # Plot chains for each parameter
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
        y=apparent_mag_values - M0_50,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=predicted_apparent_mag - M0_50,
        label=f"Best fit: $\Omega_m$={omega_50:.4f}, $M$={M0_50:.4f}",
        x_scale="log",
    )

    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=np.sqrt(np.diag(cov_matrix)),
        bins=40,
    )


if __name__ == "__main__":
    main()


"""
*****************************
Dataset: Pantheon+ (2022)
z range: 0.0102 - 2.2614
Sample size: 1590
*****************************

ΛCDM
M: -19.351 +0.007/-0.007
Ωm: 0.332 +0.018/-0.018
w0: -1
wa: 0
R-squared: 99.74 %
RMSD (mag): 0.153
Skewness of residuals: 0.090
kurtosis of residuals: 1.582
Reduced chi squared: 0.8840

=============================

wCDM
M: -19.348 +0.009/-0.009
Ωm: 0.291 +0.064/-0.078
w0: -0.90 +0.14/-0.16
wa: 0
R-squared: 99.74 %
RMSD (mag): 0.154
Skewness of residuals: 0.079
kurtosis of residuals: 1.590
Reduced chi squared: 0.8837

=============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
M: -19.348 +0.009/-0.010
Ωm: 0.312 +0.044/-0.045
w0: -0.927 +0.126/-0.145
wa: 0
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.081
kurtosis of residuals: 1.589
Reduced chi squared: 0.8838

=============================

Flat w0waCDM
M0: 19.348 +0.010/-0.010
Ωm: 0.337 +0.082/-0.148
w0: -0.919 +0.146/-0.162 (0.53 sigma)
wa: -0.3614 +1.0279/-1.8010 (0.26 sigma)
R-squared: 99.74 %
RMSD (mag): 0.154
Skewness of residuals: 0.076
kurtosis of residuals: 1.601
Reduced chi squared: 0.8868
"""
