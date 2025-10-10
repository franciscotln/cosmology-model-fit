from numba import njit
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
z_grid = np.linspace(0, np.max(z_values), num=1000)
cubed = (1 + z_grid) ** 3
one_plus_z_hel = 1 + z_hel_values


@njit
def Ez(params):
    O_m, w0 = params[1], params[2]
    O_de = 1 - O_m
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubed + O_de * rho_de)


def apparent_mag(params):
    integral_vals = cumulative_trapezoid(1 / Ez(params), z_grid, initial=0)
    I = np.interp(z_values, z_grid, integral_vals)
    return params[0] + 25 + 5 * np.log10(one_plus_z_hel * (C / H0) * I)


def chi_squared(params):
    delta = apparent_mag_values - apparent_mag(params)
    return np.dot(delta, cho_solve(cho, delta, check_finite=False))


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-20, -19),  # M
        (0, 1),  # Ωm
        (-2, 0),  # w0
    ]
)


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
    burn_in = 200
    n_dim = len(bounds)
    n_walkers = 150
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

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Acceptance fraction", np.mean(sampler.acceptance_fraction))
        print_color(
            "effective samples", n_walkers * (n_steps - burn_in) * n_dim / np.max(tau)
        )
    except:
        print_color("Autocorrelation time", "Not available")

    [
        [M0_16, M0_50, M0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([M0_50, omega_50, w0_50], dtype=np.float64)

    # Calculate residuals
    predicted_apparent_mag = apparent_mag(best_fit)
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

    M_label = f"{M0_50:.3f} +{M0_84-M0_50:.3f}/-{M0_50-M0_16:.3f}"
    omega_label = f"{omega_50:.3f} +{omega_84-omega_50:.3f}/-{omega_50-omega_16:.3f}"
    w0_label = f"{w0_50:.3f} +{w0_84-w0_50:.3f}/-{w0_50-w0_16:.3f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.4f} - {z_values[-1]:.4f}")
    print_color("M", M_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Degs of freedom", len(z_values) - len(best_fit))
    print_color("Chi squared", f"{chi_squared(best_fit):.2f}")

    labels = ["$M_0$", "$Ω_m$", "$w_0$"]
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

    _, axes = plt.subplots(n_dim, figsize=(10, 7), sharex=True)
    for i in range(n_dim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    axes[n_dim - 1].set_xlabel("walker step")
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_values,
        y=apparent_mag_values - M0_50,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=predicted_apparent_mag - M0_50,
        label=f"Best fit: $Ω_m$={omega_50:.4f}, $M$={M0_50:.4f}",
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
Degs of freedom: 1588
chi squared: 1402.92

=============================

wCDM
M: -19.347 +0.009/-0.009
Ωm: 0.292 +0.064/-0.077
w0: -0.900 +0.141/-0.159
wa: 0
R-squared: 99.74 %
RMSD (mag): 0.154
Skewness of residuals: 0.079
kurtosis of residuals: 1.589
Degs of freedom: 1587
chi squared: 1402.47

=============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
M: -19.348 +0.009/-0.009
Ωm: 0.311 +0.044/-0.045
w0: -0.925 +0.123/-0.144
wa: 0
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.081
kurtosis of residuals: 1.589
Degs of freedom: 1587
Chi squared: 1402.53

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
Degs of freedom: 1586
"""
