from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2026union3_1.data import get_data

legend, z_cmb, z_hel, mu_vals, cov_matrix = get_data()

inv_cov = np.linalg.inv(cov_matrix)

c = c0 / 1000  # Speed of light (km/s)
H0 = 70.0  # Hubble constant (km/s/Mpc)

# params indices
OFFSET = 0
OM = 1
P = 2

bounds = np.array([(-0.6, 0.6), (0.0, 1.0), (-1.75, 3.0)])  # ΔM, Ωm, p

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=3000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    Om = params[OM]
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_z(z, params):
    dh_grid = (c / H0) / Ez(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def mu_theory(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    Mz = params[OFFSET] + 1.0 - (z_cmb / (0.1 + z_cmb)) ** (0.1 * params[P])
    return Mz + 25.0 + 5 * np.log10(dL)


@njit
def chi_squared(params):
    delta = mu_vals - mu_theory(params)
    return delta @ inv_cov @ delta


@njit
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
    from sn.plotting import plot_predictions, print_color, plot_residuals

    n_dim = len(bounds)
    n_walkers = 200
    burn_in = 500
    n_steps = burn_in + 2000
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.25),
        (emcee.moves.DEMove(), 0.75),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, n_steps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

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
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)
    print_color("Gelman-Rubin R-hat:", gelman_rubin(chain_samples))

    one_sigma_ci = [15.9, 50, 84.1]
    dM_16, dM_50, dM_84 = np.percentile(samples[:, OFFSET], one_sigma_ci)
    Om_16, Om_50, Om_84 = np.percentile(samples[:, OM], one_sigma_ci)
    p_16, p_50, p_84 = np.percentile(samples[:, P], one_sigma_ci)

    best_fit = np.percentile(samples, 50, axis=0)

    predicted_distances = mu_theory(best_fit)
    residuals = mu_vals - predicted_distances

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mu_vals - np.mean(mu_vals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    rmsd = np.sqrt(np.mean(residuals**2))

    dM_label = f"{dM_50:.3f} +{dM_84-dM_50:.3f}/-{dM_50-dM_16:.3f}"
    Om_label = f"{Om_50:.3f} +{Om_84-Om_50:.3f}/-{Om_50-Om_16:.3f}"
    p_label = f"{p_50:.3f} +{p_84-p_50:.3f}/-{p_50-p_16:.3f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_cmb[0]:.3f} - {z_cmb[-1]:.3f}")
    print_color("Sample size", len(z_cmb))
    print_color("ΔM", dM_label)
    print_color("p", p_label)
    print_color("Ωm", Om_label)
    print_color("R-squared (%)", f"{100 * r2:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skew(residuals):.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit):.1f}")
    print_color("Log evidence", f"{log_evd:.1f}")
    print_color("Degs of freedom", len(z_cmb) - len(best_fit))

    sigma_mu = np.sqrt(np.diag(cov_matrix))

    labels = ["$ΔM$", "$Ω_m$", "$p$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chain_samples)
    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=mu_vals,
        y_err=sigma_mu,
        y_model=predicted_distances,
        label=f"$Ω_m$={Om_50:.4f}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=sigma_mu, bins=40)


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*******************************

Flat ΛCDM: w(z) = -1

ΔM: 0.027 +0.020/-0.020
Ωm: 0.335 +0.025/-0.024
R-squared (%): 99.96
RMSD (mag): 0.044
Skewness of residuals: 0.088
Chi squared: 28.8
Log evidence: -22.0
Degs of freedom: 20

===============================

Flat ΛCDM: w(z) = -1, varying absolute magnitude
M(z) = ΔM_inf + 1 - (z / (0.1 + z))^(0.1 * p)

ΔM: -0.025 +0.036/-0.036
p: 0.637 +0.394/-0.373 (prior ~ U(-1.75, 3.0))
Ωm: 0.292 +0.034/-0.032 (complete agreement with ΛCDM from BAO)
R-squared (%): 99.96
RMSD (mag): 0.046
Skewness of residuals: -0.892
Chi squared: 25.7
Log evidence: -22.1
Degs of freedom: 19

===============================

Flat wCDM: w(z) = w0

ΔM: 0.034 +0.020/-0.020
Ωm: 0.246 +0.081/-0.100
w0: -0.786 +0.152/-0.179
R-squared (%): 99.95
RMSD (mag): 0.050
Skewness of residuals: -1.534
Chi squared: 27.2
Log evidence: -22.5
Degs of freedom: 19

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)

ΔM: 0.042 +0.021/-0.021
Ωm: 0.280 +0.040/-0.045
w0: -0.745 +0.131/-0.137 (prior width 1.0: -1.0 to 0.0)
R-squared (%): 99.95
RMSD (mag): 0.049
Skewness of residuals: -1.495
Chi squared: 26.6
Log evidence: -21.8
Degs of freedom: 19

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
TODO
"""
