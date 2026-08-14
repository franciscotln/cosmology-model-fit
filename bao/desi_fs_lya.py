from numba import njit, prange
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_pchip, interp_hermite
from y2025BAO.data_fs_lya import get_data

c = c0 / 1000  # Speed of light in km/s
rd = 147.09  # Mpc, fixed

legend, data, cov_matrix = get_data()
inv_cov_bao = np.linalg.inv(cov_matrix)

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    cubed = (1.0 + z) ** 3
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2


@njit
def H_z(z, params):
    h, Om, w0 = params
    OL = 1.0 - Om
    return 100 * h * np.sqrt(Om * (1.0 + z) ** 3 + OL * Ode_z(z, w0))


@njit
def bao_theory(z, qty, theta):
    DH_grid = c / H_z(z_grid, theta)
    dh = (DH_grid[:-1] + DH_grid[1:]) / 2
    DM_grid = np.zeros(z_grid.size, dtype=np.float64)
    DM_grid[1:] = np.cumsum(dz * dh)

    DH = interp_pchip(z, x=z_grid, y=DH_grid)
    DM = interp_hermite(z, x=z_grid, y=DM_grid, y_prime=DH_grid)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    F_AP_mask = qty == 3

    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH[DH_mask] / rd
    results[DM_mask] = DM[DM_mask] / rd
    results[DV_mask] = (z[DV_mask] * DH[DV_mask] * DM[DV_mask] ** 2) ** (1 / 3) / rd
    results[F_AP_mask] = DM[F_AP_mask] / DH[F_AP_mask]
    return results


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def chi_squared(theta):
    delta_bao = data["value"] - bao_theory(data["z"], bao_qty, theta)
    return delta_bao @ inv_cov_bao @ delta_bao


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


@njit
def log_likelihood(params):
    return -0.5 * chi_squared(params)


@njit
def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


@njit(parallel=True)
def log_probs_vectorized(batch):
    N = batch.shape[0]
    log_probs = np.empty(N, dtype=np.float32)
    for i in prange(N):
        log_probs[i] = log_probability(batch[i])
    return log_probs


def main():
    import emcee
    from bao.plot_predictions import plot_bao_predictions, plot_bao_residuals
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains

    np.random.seed(42)
    n_dim = len(bounds)
    n_walkers = 100
    burn_in = 1000
    nsteps = 9000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_walkers, n_dim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.15),
        (emcee.moves.DEMove(), 0.85),
    ]

    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_probs_vectorized, moves=moves, vectorize=True
    )
    sampler.run_mcmc(
        initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
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
        (h_16, h_50, h_84),
        (Om_16, Om_50, Om_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    degs_of_freedom = data["value"].size - len(best_fit)

    residuals = data["value"] - bao_theory(data["z"], bao_qty, best_fit)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"h * rd: {rd * h_50:.2f} +{rd * (h_84 - h_50):.2f} -{rd * (h_50 - h_16):.2f}")
    print(f"Ωm: {Om_50:.4f} +{Om_84-Om_50:.4f} -{Om_50-Om_16:.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {degs_of_freedom}")
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
Dataset: DESI DR2 2025 + FS Lyman-alpha
rd: 147.09 Mpc (fixed)
*******************************

Flat ΛCDM:
h * rd: 101.17 +0.67 -0.67
Ωm: 0.3016 +0.0078 -0.0076
Chi squared: 12.81
Degs of freedom: 12
R^2: 0.9987
RMSD: 0.298
"""

"""
Flat wCDM:
h * rd: 100.46 +1.77 -1.62
Ωm: 0.3021 +0.0083 -0.0081
w0: -0.967 +0.072 -0.075 (prior width 1: from -1.4 to -0.4)
Chi squared: 12.61
Degs of freedom: 11
R^2: 0.9988
RMSD: 0.287
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
h * rd: 98.49 +1.76 -2.08
Ωm: 0.3140 +0.0120 -0.0108
w0: -0.833 +0.124 -0.106 (prior width 1: from -1 to 0) - left side truncated
Chi squared: 12.08
Degs of freedom: 11
R^2: 0.9990
RMSD: 0.268
"""

"""
Flat w0waCDM:
TODO
w0: (prior width 2.5: -1.5 to 1.0)
wa: (prior width 10.0: -8.0 to 2.0)

"""
