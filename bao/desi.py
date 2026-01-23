from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_pchip
from y2025BAO.data import get_data

c = c0 / 1000  # Speed of light in km/s
rd = 147.09  # Mpc, fixed

legend, data, cov_matrix = get_data()
inv_cov_bao = np.linalg.inv(cov_matrix)

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dx = np.diff(z_grid)


@njit
def H_z(z, params):
    h, Om, w0 = params
    OL = 1.0 - Om
    one_plus_z = 1.0 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2
    return 100 * h * np.sqrt(Om * cubed + OL * rho_de)


@njit
def DH_z(z, theta):
    return c / H_z(z, theta)


@njit
def bao_theory(z, qty, theta):
    dh_grid = DH_z(z_grid, theta)

    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    dm_grid = np.zeros(z_grid.size, dtype=np.float64)
    dm_grid[1:] = np.cumsum(dx * dy)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2

    results = np.empty(z.size, dtype=np.float64)

    results[DH_mask] = interp_pchip(z[DH_mask], z_grid, dh_grid)
    results[DM_mask] = interp_pchip(z[DM_mask], z_grid, dm_grid)

    dh_at_z = interp_pchip(z[DV_mask], z_grid, dh_grid)
    dm_at_z = interp_pchip(z[DV_mask], z_grid, dm_grid)
    results[DV_mask] = (z[DV_mask] * dh_at_z * dm_at_z**2) ** (1 / 3)
    return results / rd


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def chi_squared(theta):
    delta_bao = data["value"] - bao_theory(data["z"], quantities, theta)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao
    return chi_bao


bounds = np.array(
    [
        (0.50, 0.80),  # h
        (0.1, 0.5),  # Ωm
        (-1.0, 0.0),  # w0
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee
    from bao.plot_predictions import plot_bao_predictions, plot_bao_residuals
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains

    np.random.seed(42)
    n_dim = len(bounds)
    n_walkers = 100
    burn_in = 500
    nsteps = 5000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_walkers, n_dim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.30),
        (emcee.moves.DEMove(), 0.70),
    ]

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, moves=moves)
    sampler.run_mcmc(
        initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#FF5733"}
    )

    try:
        tau = sampler.get_autocorr_time()
        print("Auto-correlation time:", tau)
        print(
            "Effective samples:", n_dim * n_walkers * (nsteps - burn_in) / np.max(tau)
        )
        print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chain_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)
    print("Gelman-Rubin:", gelman_rubin(chain_samples))

    [
        [h_16, h_50, h_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    residuals = data["value"] - bao_theory(data["z"], quantities, best_fit)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"h: {h_50:.3f} +{(h_84 - h_50):.3f} -{(h_50 - h_16):.3f}")
    print(f"Ωm: {Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {data['value'].size  - len(best_fit)}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    labels = ["$h$", "$Ω_m$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chain_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=legend,
    )
    plot_bao_residuals(data, residuals, np.sqrt(np.diag(cov_matrix)))


if __name__ == "__main__":
    main()


"""
*******************************
Dataset: DESI DR2 2024
*******************************

Flat ΛCDM:
rd: 147.09 Mpc (fixed)
h: 0.690 +0.005 -0.005
Ωm: 0.298 +0.009 -0.008
w0: -1
wa: 0
Chi squared: 10.27
Log evidence: -12.19
Degs of freedom: 11
R^2: 0.9987
RMSD: 0.305

================================

Flat wCDM:
rd: 147.09 Mpc (fixed)
h: 0.678 +0.012 -0.011
Ωm: 0.297 +0.009 -0.009
w0: -0.915 +0.076 -0.079 (prior width 1.0: from -1.4 to -0.4)
Chi squared: 9.11
Log evidence: -13.22
Degs of freedom: 10
R^2: 0.9989
RMSD: 0.279

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
rd: 147.09 Mpc (fixed)
h: 0.665 +0.015 -0.015
Ωm: 0.312 +0.012 -0.012
w0: -0.767 +0.133 -0.131 (prior width 1.0: from -1.0 to 0.0) - left side truncated
Chi squared: 8.29
Log evidence: -12.19
Degs of freedom: 10
R^2: 0.9991
RMSD: 0.261

===============================

Flat w0waCDM:
rd: 147.09 Mpc (fixed)
h: 0.621 +0.032 -0.029
Ωm: 0.386 +0.046 -0.046
w0: -0.184 +0.443 -0.422 (prior width 2.5: -1.5 to 1.0)
wa: -2.721 +1.492 -1.535 (prior width 10.0: -8.0 to 2.0)
Chi squared: 5.63
Log evidence: -14.50
Degs of freedom: 9
R^2: 0.9994
RMSD: 0.202
"""
