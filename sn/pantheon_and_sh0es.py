from numba import njit
import numpy as np
import scipy.stats as stats
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2022pantheonSHOES.data_shoes import get_data

legend, z_values, z_hel_values, apparent_mag_values, cepheid_distances, cov_matrix = (
    get_data()
)

cepheids_mask = cepheid_distances != -9
cho = cho_factor(cov_matrix, lower=True)[0]

c = 299792.458  # Speed of light (km/s)

z_grid = np.linspace(0, np.max(z_values) + 0.1, num=3000)
dz = np.diff(z_grid)

zp1 = 1.0 + z_grid
zp1_hel = 1.0 + z_hel_values


@njit
def Ez(params):
    O_m, w0 = params[2], params[3]
    rho_de = (2 * zp1**3 / ((1.0 + w0) + (1.0 - w0) * zp1**3)) ** 2
    return np.sqrt(O_m * zp1**3 + (1.0 - O_m) * rho_de)


@njit
def DM_z(theta):
    dh_grid = (c / theta[1]) / Ez(theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z_values, z_grid, cum_dm, dh_grid)


@njit
def model_mu(params):
    return 25.0 + 5.0 * np.log10(zp1_hel * DM_z(params))


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    mu_theory = np.where(cepheids_mask, cepheid_distances, model_mu(params))
    apparent_mag_theory = mu_theory + params[0]
    delta = apparent_mag_values - apparent_mag_theory
    return solve_triang(cho, delta)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-19.5, -19.0),  # M
        (60.0, 85.0),  # H0
        (0.1, 0.6),  # Ωm
        (-1.0, 0.0),  # w0
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions, print_color, plot_residuals

    burn_in = 500
    n_dim = len(bounds)
    n_walkers = 150
    n_steps = burn_in + 2500
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.2),
        (emcee.moves.DEMove(), 0.8),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, n_steps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Acceptance fraction", np.mean(sampler.acceptance_fraction))
        print_color(
            "effective samples", n_dim * n_walkers * (n_steps - burn_in) / np.max(tau)
        )
    except:
        print_color("Autocorrelation time", "Not available")

    [
        (M_16, M_50, M_84),
        (H0_16, H0_50, H0_84),
        (omega_16, omega_50, omega_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    predicted_mu_values = model_mu(best_fit)
    residuals = (
        apparent_mag_values
        - M_50
        - np.where(cepheids_mask, cepheid_distances, predicted_mu_values)
    )

    # Compute R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((apparent_mag_values - np.mean(apparent_mag_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compute root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    M_label = f"{M_50:.3f} +{M_84-M_50:.3f}/-{M_50-M_16:.3f}"
    H0_label = f"{H0_50:.2f} +{H0_84-H0_50:.2f}/-{H0_50-H0_16:.2f}"
    omega_label = f"{omega_50:.3f} +{omega_84-omega_50:.3f}/-{omega_50-omega_16:.3f}"
    w0_label = f"{w0_50:.3f} +{w0_84-w0_50:.3f}/-{w0_50-w0_16:.3f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.4f} - {z_values[-1]:.4f}")
    print_color("Sample size", len(z_values))
    print_color("M", M_label)
    print_color("H0 (km/s/Mpc)", H0_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{stats.skew(residuals):.3f}")
    print_color("kurtosis of residuals", f"{stats.kurtosis(residuals):.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit):.2f}")

    sigma_mu = np.sqrt(cov_matrix.diagonal())

    labels = ["M", "$H_0$", "$Ω_m$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_predictions(
        legend=legend,
        x=z_values,
        y=apparent_mag_values - M_50,
        y_err=sigma_mu,
        y_model=predicted_mu_values,
        label=f"H0={H0_50:.2f} km/s/Mpc",
        x_scale="log",
    )
    plot_residuals(z_values=z_values, residuals=residuals, y_err=sigma_mu, bins=40)


if __name__ == "__main__":
    main()

"""
*****************************
Dataset: Pantheon+ and SH0ES
z range: 0.0012 - 2.2614
Sample size: 1657
*****************************

ΛCDM w(z) = -1
M: -19.24 +0.03/-0.03 mag
H0 (km/s/Mpc): 73.5 +- 1.0
Ωm: 0.332 +0.018/-0.018
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.085
kurtosis of residuals: 1.555
Chi squared: 1452.02

=============================

ΛCDM w(z) = -1
Evolving absolute mag of SNe M(z) = M_max + p * [1 - (z / (0.1 + z))^0.05]

p: 0.269 +0.279/-0.282 mag
M_max: -19.279 +0.048/-0.048 mag
H0 (km/s/Mpc): 72.87 +1.24/-1.22
Ωm: 0.315 +0.025/-0.024
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.065
kurtosis of residuals: 1.555
Chi squared: 1451.02

=============================

wCDM w(z) = w0
M: -19.24 +0.03/-0.03 mag
H0 (km/s/Mpc): 73.5 +1.0/-1.0 km/s/Mpc
Ωm: 0.301 +0.062/-0.075
w0: -0.92 +0.14/-0.16
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.076
kurtosis of residuals: 1.561
Chi squared: 1451.70

=============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
M: -19.243 +0.030/-0.029
H0 (km/s/Mpc): 73.34 +1.04/-1.01
Ωm: 0.300 +0.028/-0.033
w0: -0.877 +0.101/-0.080
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.070
kurtosis of residuals: 1.564
Chi squared: 1451.86
"""
