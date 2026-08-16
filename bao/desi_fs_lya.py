from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_pchip, interp_hermite
from y2025BAO.data_fs_lya import get_data

c = c0 / 1000  # Speed of light in km/s
RD = 147.09  # Mpc, fixed

legend, data, cov_matrix = get_data()
inv_cov_bao = np.linalg.inv(cov_matrix)

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def ode_z(z, w0):
    cubed = (1.0 + z) ** 3
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2


@njit
def h_z(z, params):
    h, o_m, w0 = params
    o_l = 1.0 - o_m
    return 100 * h * np.sqrt(o_m * (1.0 + z) ** 3 + o_l * ode_z(z, w0))


@njit
def bao_theory(z, qty, theta):
    dh_grid = c / h_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    dm_grid = np.zeros(z_grid.size, dtype=np.float64)
    dm_grid[1:] = np.cumsum(dz * dh)

    dh_vals = interp_pchip(z, x=z_grid, y=dh_grid)
    dm_vals = interp_hermite(z, x=z_grid, y=dm_grid, y_prime=dh_grid)

    dv_mask = qty == 0
    dm_mask = qty == 1
    dh_mask = qty == 2
    f_ap_mask = qty == 3

    results = np.empty(z.size, dtype=np.float64)
    results[dh_mask] = dh_vals[dh_mask] / RD
    results[dm_mask] = dm_vals[dm_mask] / RD
    results[dv_mask] = (z[dv_mask] * dh_vals[dv_mask] * dm_vals[dv_mask] ** 2) ** (1 / 3) / RD
    results[f_ap_mask] = dm_vals[f_ap_mask] / dh_vals[f_ap_mask]
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


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    from multiprocessing import Pool
    import emcee
    from bao.plot_predictions import plot_bao_predictions, plot_bao_residuals
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 1000
    nsteps = 9000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.15),
        (emcee.moves.DEMove(), 0.85),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    try:
        tau = sampler.get_autocorr_time()
        print("Auto-correlation time:", tau)
        print(
            "Effective samples:", ndim * nwalkers * (nsteps - burn_in) / np.max(tau)
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
        (om_16, om_50, om_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    dof = len(data["value"]) - len(best_fit)

    residuals = data["value"] - bao_theory(data["z"], bao_qty, best_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"h * rd: {RD * h_50:.2f} +{RD * (h_84 - h_50):.2f} -{RD * (h_50 - h_16):.2f}")
    print(f"Ωm: {om_50:.4f} +{om_84-om_50:.4f} -{om_50-om_16:.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"χ2: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"DOF: {dof}")
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


# *******************************
# Dataset: DESI DR2 2025 + FS Lyman-alpha
# *******************************

# Flat ΛCDM:
# h * rd: 101.18 +0.67 -0.67
# Ωm: 0.3016 +0.0078 -0.0076
# χ2: 12.81
# DOF: 12
# χ2/dof: 1.07
# R^2: 0.9987
# RMSD: 0.298

# Flat wCDM:
# h * rd: 100.42 +1.78 -1.67
# Ωm: 0.3021 +0.0082 -0.0080
# w0: -0.966 +0.072 -0.075 (prior ~U(-1.4, -0.4))
# χ2: 12.61
# DOF: 11
# χ2/dof: 1.15
# R^2: 0.9988
# RMSD: 0.287

# Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
# h * rd: 98.49 +1.77 -2.09
# Ωm: 0.3141 +0.0120 -0.0108
# w0: -0.833 +0.124 -0.105 (prior ~U(-1, 0)) - left side truncated
# χ2: 12.08
# DOF: 11
# χ2/dof: 1.10
# R^2: 0.9990
# RMSD: 0.267

# Flat w0waCDM:
# h * rd: 90.02 +4.63 -4.16
# Ωm: 0.403 +0.044 -0.044 (prior ~U(0.1, 0.8))
# w0: -0.042 +0.441 -0.429 (prior ~U(-2.5, 2.5))
# wa: -3.30 +1.45 -1.48 (prior ~U(-10, 4))
# χ2: 7.21
# DOF: 10
# χ2/dof: 0.72
# R^2: 0.9995
# RMSD: 0.195
