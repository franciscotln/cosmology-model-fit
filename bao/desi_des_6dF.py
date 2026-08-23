from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import block_diag
from interpolator import interp_pchip, interp_hermite
from y2025BAO.data import get_data as ge_desi_data
from y2024DESBAO.data import get_data as get_des_data
from y20116dFBAO.data import get_data as get_6dF_data

c = c0 / 1000  # Speed of light in km/s
rd = 147.09  # Mpc, fixed

legend_desi, data_desi, cov_desi = ge_desi_data()
legend_des, data_des, cov_des = get_des_data()
legend_6dF, data_6dF, cov_6dF = get_6dF_data()

data = np.concatenate((data_desi, data_des, data_6dF))
cov_matrix = block_diag(cov_desi, cov_des, cov_6dF)

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

    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH[DH_mask]
    results[DM_mask] = DM[DM_mask]
    results[DV_mask] = (z[DV_mask] * DH[DV_mask] * DM[DV_mask] ** 2) ** (1 / 3)
    return results / rd


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
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
def log_probability_jit(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def log_probability(params):
    return log_probability_jit(params)


def main():
    import emcee
    from multiprocessing import Pool
    from bao.plot_predictions import plot_bao_predictions, plot_bao_residuals
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains

    np.random.seed(42)
    n_dim = len(bounds)
    n_walkers = 100
    burn_in = 500
    nsteps = 5000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_walkers, n_dim))
    moves = [
        (emcee.moves.KDEMove(), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability, moves=moves, pool=pool
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
    degs_of_freedom = data["value"].size + data_des["value"].size - len(best_fit)

    residuals = data["value"] - bao_theory(data["z"], bao_qty, best_fit)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"h: {h_50:.3f} +{(h_84 - h_50):.3f} -{(h_50 - h_16):.3f}")
    print(f"Ωm: {Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}")
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
        title=f"{legend_des} + {legend_desi}",
    )
    plot_bao_residuals(data, residuals, np.sqrt(np.diag(cov_matrix)))


if __name__ == "__main__":
    main()


# *********************************
# Dataset: DESI DR2 + DES6Y BAO + 6dF BAO
# *********************************


# ----------- Flat ΛCDM -----------
# rd: 147.09 Mpc (fixed)
# h: 0.691 +0.005 -0.005
# Ωm: 0.297 +0.009 -0.008
# Chi squared: 10.93
# Log evidence: -12.54
# Degs of freedom: 14
# R^2: 0.9988
# RMSD: 0.305
# ---------------------------------


# ----------- Flat wCDM -----------
# rd: 147.09 Mpc (fixed)
# h: 0.681 +0.012 -0.011
# Ωm: 0.296 +0.009 -0.009
# w0: -0.927 +0.075 -0.078 (prior U[-2, 0])
# Chi squared: 10.02
# Log evidence: -14.41
# Degs of freedom: 13
# R^2: 0.9990
# RMSD: 0.286
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
#
# rd: 147.09 Mpc (fixed)
# h: 0.670 +0.013 -0.014
# Ωm: 0.309 +0.012 -0.011
# w0: -0.799 +0.127 -0.119 (prior U[-1, 0]) - left side truncated
# Chi squared: 9.49
# Log evidence: -12.81
# Degs of freedom: 13
# R^2: 0.9991
# RMSD: 0.272
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < 0 enforced in the likelihood
#
# rd: 147.09 Mpc (fixed)
# h: 0.652 +0.027 -0.021
# Ωm: 0.343 +0.030 -0.041
# w0: -0.57 +0.27 -0.32 (prior U[-3, 1])
# wa: < -1.412 (prior U[-3, 2]) truncated posterior
# Chi squared: 8.23
# Log evidence: -14.77
# Degs of freedom: 12
# R^2: 0.9992
# RMSD: 0.246
# ---------------------------------
