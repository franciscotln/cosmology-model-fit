from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from scipy.constants import c as c0
from y2023union3.data import get_data

legend, z_values, mu_vals, cov_matrix = get_data()

cho = cho_factor(cov_matrix, lower=True)[0]

C = c0 / 1000  # Speed of light (km/s)
H0 = 70  # Hubble constant (km/s/Mpc)

# params indices
OFFSET = 0
OM = 1
W0 = 2

bounds = np.array([(-0.6, 0.6), (0.0, 1.0), (-1.5, 0.0)])  # ΔM, Ωm, w0

z_grid = np.linspace(0, np.max(z_values), num=1000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    Om, w0 = params[OM], params[W0]
    cubed = (1 + z) ** 3
    rho_de = np.exp((1 + w0) * (1 - 1 / cubed))
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


@njit
def DM_z(z, params):
    dh_grid = (C / H0) / Ez(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def mu_theory(params):
    dL = (1 + z_values) * DM_z(z_values, params)
    return params[OFFSET] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    return solve_triang(cho, mu_vals - mu_theory(params))


def log_likelihood(params):
    return -0.5 * chi_squared(params)


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
    from scipy.stats import skew
    from multiprocessing import Pool
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from .plotting import plot_predictions, print_color, plot_residuals

    n_dim = len(bounds)
    n_walkers = 150
    burn_in = 200
    n_steps = burn_in + 2000
    np.random.seed(42)
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
        print(
            "effective samples", n_walkers * (n_steps - burn_in) * n_dim / np.max(tau)
        )
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chain_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    print_color("Gelman-Rubin R-hat:", gelman_rubin(chain_samples))

    one_sigma_ci = [15.9, 50, 84.1]
    dM_16, dM_50, dM_84 = np.percentile(samples[:, OFFSET], one_sigma_ci)
    Om_16, Om_50, Om_84 = np.percentile(samples[:, OM], one_sigma_ci)
    w0_16, w0_50, w0_84 = np.percentile(samples[:, W0], one_sigma_ci)

    best_fit_params = np.array([dM_50, Om_50, w0_50], dtype=np.float64)

    predicted_distances = mu_theory(best_fit_params)
    residuals = mu_vals - predicted_distances

    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mu_vals - np.mean(mu_vals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    rmsd = np.sqrt(np.mean(residuals**2))

    dM_label = f"{dM_50:.3f} +{dM_84-dM_50:.3f}/-{dM_50-dM_16:.3f}"
    Om_label = f"{Om_50:.3f} +{Om_84-Om_50:.3f}/-{Om_50-Om_16:.3f}"
    w0_label = f"{w0_50:.3f} +{w0_84-w0_50:.3f}/-{w0_50-w0_16:.3f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("ΔM", dM_label)
    print_color("Ωm", Om_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r2:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skew(residuals):.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit_params):.1f}")
    print_color(
        "Log evidence",
        f"{log_evidence(samples, log_probs, log_probability, bounds):.1f}",
    )
    print_color("Degs of freedom", len(z_values) - len(best_fit_params))

    sigma_mu = np.sqrt(np.diag(cov_matrix))

    plot_corner_and_chains(
        labels=["$Δ_M$", "$Ω_m$", "$w_0$"],
        flat_samples=samples,
        samples=chain_samples,
    )
    plot_predictions(
        legend=legend,
        x=z_values,
        y=mu_vals,
        y_err=sigma_mu,
        y_model=predicted_distances,
        label=f"Best fit: $Ω_m$={Om_50:.4f}",
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

Ωm: 0.357 +0.027/-0.026
w0: -1
wa: 0
R-squared (%): 99.95
RMSD (mag): 0.050
Skewness of residuals: 0.590
Chi squared: 24.0
Log evidence: -16.4
Degs of freedom: 20

===============================

Flat wCDM: w(z) = w0

Ωm: 0.252 +0.086/-0.108
w0: -0.748 +0.153/-0.183
wa: 0
R-squared (%): 99.94
RMSD (mag): 0.055
Skewness of residuals: -1.266
Chi squared: 22.1
Log evidence: -16.5
Degs of freedom: 19

===============================

Flat alternative: w(z) = -1 + (1 + w0) / (1 + z)^3

Ωm: 0.302 +0.047/-0.046
w0: -0.712 +0.159/-0.188
wa: -3 * (1 + w0)
R-squared (%): 99.94
RMSD (mag): 0.053
Skewness of residuals: -1.030
Chi squared: 21.5
Log evidence: -16.3
Degs of freedom: 19

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)

ΔM: -0.0319 +0.0904/-0.0899 mag
Ωm: 0.4406 +0.0555/-0.0923
w0: -0.5574 +0.2799/-0.2343
wa: -4.1187 +2.8976/-3.2694
R-squared (%): 99.96
RMSD (mag): 0.043
Skewness of residuals: 0.669
Chi squared: 20.6
Degs of freedom: 18
"""
