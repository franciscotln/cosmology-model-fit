import emcee
import matplotlib.pyplot as plt
import numpy as np
import corner
from scipy.stats import skew
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2023union3.data import get_data

legend, z_values, mu_vals, cov_matrix = get_data()

cho = cho_factor(cov_matrix)

C = 299792.458  # Speed of light (km/s)
H0 = 70  # Hubble constant (km/s/Mpc)


z = np.linspace(0, np.max(z_values), num=4000)
one_plus_z = 1 + z


def h_over_h0(params):
    omega_m, w0 = params[1], params[2]
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(omega_m * one_plus_z**3 + (1 - omega_m) * evolving_de)


def integral_of_e_z(params):
    integral_values = cumulative_trapezoid(1 / h_over_h0(params), z, initial=0)
    return np.interp(z_values, z, integral_values)


# Flat model
def distance_modulus(params):
    a0_over_ae = 1 + z_values
    comoving_distance = (C / H0) * integral_of_e_z(params)
    return params[0] + 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def chi_squared(params):
    delta = mu_vals - distance_modulus(params)
    return np.dot(delta, cho_solve(cho, delta))


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([(-0.6, 0.6), (0, 1), (-2, 0)])  # ΔM  # Ωm  # w0


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
    n_dim = len(bounds)
    n_walkers = 16 * n_dim
    discarded_steps = 500
    n_steps = discarded_steps + 20000
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=discarded_steps, flat=True)

    [
        [dM_16, dM_50, dM_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [16, 50, 84], axis=0).T

    best_fit_params = [dM_50, omega_50, w0_50]

    predicted_distances = distance_modulus(best_fit_params)
    residuals = mu_vals - predicted_distances

    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mu_vals - np.mean(mu_vals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    dM_label = f"{dM_50:.4f} +{dM_84-dM_50:.4f}/-{dM_50-dM_16:.4f}"
    omega_label = f"{omega_50:.4f} +{omega_84-omega_50:.4f}/-{omega_50-omega_16:.4f}"
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f}/-{w0_50-w0_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("ΔM", dM_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r2:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skew(residuals):.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit_params):.4f}")
    print_color("Degs of freedom", len(z_values) - len(best_fit_params))

    labels = ["ΔM", "Ωm", "$w_0$"]
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

    sigma_mu = np.sqrt(np.diag(cov_matrix))

    plot_predictions(
        legend=legend,
        x=z_values,
        y=mu_vals,
        y_err=sigma_mu,
        y_model=predicted_distances,
        label=f"Best fit: $\Omega_m$={omega_50:.4f}",
        x_scale="log",
    )

    plot_residuals(z_values=z_values, residuals=residuals, y_err=sigma_mu, bins=40)


if __name__ == "__main__":
    main()

"""
*****************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*****************************

Flat ΛCDM

ΔM: -0.0698 +0.0878/-0.0874 mag
Ωm: 0.3570 +0.0274/-0.0266
w0: -1
wa: 0
R-squared (%): 99.95
RMSD (mag): 0.050
Skewness of residuals: 0.582
Chi squared: 23.9594
degrees of freedom: 20

=============================

Flat wCDM

ΔM: -0.0601 +0.0876/-0.0876 mag
Ωm: 0.2532 +0.0883/-0.1113
w0: -0.7483 +0.1568/-0.1893 (1.45 sigma)
wa: 0
R-squared (%): 99.94
RMSD (mag): 0.054
Skewness of residuals: -1.244
Chi squared: 22.1323
degrees of freedom: 19

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.0561 +0.0888/-0.0888 mag
Ωm: 0.2872 +0.0605/-0.0646
w0: -0.7601 +0.1448/-0.1744 (1.38 - 1.66 sigma)
wa: 0
R-squared (%): 99.94
RMSD (mag): 0.054
Skewness of residuals: -1.107
Chi squared: 21.8765
Degs of freedom: 19

=============================

Flat w0waCDM
ΔM: -0.0326 +0.0913/-0.0907 mag
Ωm: 0.4449 +0.0523/-0.0800
w0: -0.5456 +0.2772/-0.2322 (1.78 sigma)
wa: -4.3188 +2.6906/-3.1919 (1.47 sigma)
R-squared (%): 99.96
RMSD (mag): 0.042
Skewness of residuals: 0.724
Chi squared: 20.5888
degrees of freedom: 18
"""
