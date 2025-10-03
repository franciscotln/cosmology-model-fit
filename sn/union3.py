from numba import njit
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


z = np.linspace(0, np.max(z_values), num=1000)
cubed = (1 + z) ** 3


@njit
def Ez(params):
    omega_m, w0 = params[1], params[2]
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(omega_m * cubed + (1 - omega_m) * rho_de)


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(params), z, initial=0)
    return np.interp(z_values, z, integral_values)


# Flat model
def mu_theory(params):
    a0_over_ae = 1 + z_values
    comoving_distance = (C / H0) * integral_Ez(params)
    return params[0] + 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def chi_squared(params):
    delta = mu_vals - mu_theory(params)
    return delta.dot(cho_solve(cho, delta, check_finite=False))


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([(-0.6, 0.6), (0, 1), (-2.5, 0.5)])  # ΔM  # Ωm  # w0


@njit
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
    n_walkers = 150
    burn_in = 200
    n_steps = burn_in + 2000
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.30),
                (emcee.moves.DEMove(), 0.56),
                (emcee.moves.DESnookerMove(), 0.14),
            ],
        )
        sampler.run_mcmc(initial_pos, n_steps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", n_walkers * n_steps * n_dim / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [dM_16, dM_50, dM_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit_params = np.array([dM_50, omega_50, w0_50], dtype=np.float64)

    predicted_distances = mu_theory(best_fit_params)
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
        smooth=2.0,
        smooth1d=2.0,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    chain_samples = sampler.get_chain(discard=burn_in, flat=False)
    plt.figure(figsize=(16, 1.5 * n_dim))
    for n in range(n_dim):
        plt.subplot2grid((n_dim, 1), (n, 0))
        plt.plot(chain_samples[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
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
*******************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*******************************

Flat ΛCDM: w(z) = -1

ΔM: -0.069 +0.088/-0.088 mag
Ωm: 0.357 +0.027/-0.027
w0: -1
wa: 0
R-squared (%): 99.95
RMSD (mag): 0.050
Skewness of residuals: 0.587
Chi squared: 24.0
degrees of freedom: 20

===============================

Flat wCDM: w(z) = w0

ΔM: -0.058 +0.088/-0.088 mag
Ωm: 0.253 +0.088/-0.110
w0: -0.748 +0.155/-0.187 (1.35 - 1.63 sigma)
wa: 0
R-squared (%): 99.94
RMSD (mag): 0.055
Skewness of residuals: -1.259
Chi squared: 22.1
degrees of freedom: 19

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)

ΔM: -0.055 +0.088/-0.088 mag
Ωm: 0.296 +0.053/-0.054
w0: -0.751 +0.144/-0.172 (1.45 - 1.73 sigma)
wa: 0
R-squared (%): 99.94
RMSD (mag): 0.053
Skewness of residuals: -1.068
Chi squared: 21.7
Degs of freedom: 19

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)

ΔM: -0.0333 +0.090/-0.090 mag
Ωm: 0.437 +0.057/-0.086
w0: -0.570 +0.271/-0.229
wa: -3.942 +2.859/-3.194
R-squared (%): 99.96
RMSD (mag): 0.043
Skewness of residuals: 0.629
Chi squared: 20.6
degrees of freedom: 18
"""
