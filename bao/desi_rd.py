from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from y2025BAO.data import get_data as get_bao_data

bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s

z_grid = np.linspace(0, np.max(bao_data["z"]) + 0.1, num=2000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    Om, w0 = params[2], params[3]
    inv_a = 1 + z
    cubic = inv_a**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Om * cubic + (1 - Om) * rho_de)


@njit
def H_z(z, params):
    return params[1] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[0]


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


"""
Planck prior on sound horizon at drag epoch r_d in Mpc
width increased by 100% to account for model dependence
"""
rd_planck = 147.09
rd_planck_sigma = 2 * 0.26


def chi_squared(params):
    delta_prior = params[0] - rd_planck
    chi_prior = (delta_prior / rd_planck_sigma) ** 2

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi_bao + chi_prior


bounds = np.array(
    [
        (120, 180),  # rd
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (-1.5, 0.0),  # w0
    ],
    dtype=np.float64,
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
    from multiprocessing import Pool
    from .plot_predictions import plot_bao_predictions
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
    ]
    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    best_fit = np.percentile(samples, 50, axis=0)
    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0)
    rd_err, H0_err, Om_err, w0_err = np.diff(pct, axis=0).T

    print("Gelman-Rubin:", gelman_rubin(chains_samples))
    print(f"rd: {best_fit[0]:.2f} +{rd_err[1]:.2f} -{rd_err[0]:.2f} Mpc")
    print(f"H0: {best_fit[1]:.2f} +{H0_err[1]:.2f} -{H0_err[0]:.2f} km/s/Mpc")
    print(f"Ωm: {best_fit[2]:.3f} +{Om_err[1]:.3f} -{Om_err[0]:.3f}")
    print(f"w0: {best_fit[3]:.3f} +{w0_err[1]:.3f} -{w0_err[0]:.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {1 + len(bao_data['z']) - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_corner_and_chains(
        labels=["$r_d$", "$H_0$", "$Ω_m$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
*******************************
DESI BAO DR2 2025
*******************************

Flat ΛCDM
rd: 147.09 +0.52 -0.51 Mpc
H0: 69.03 +0.54 -0.55 km/s/Mpc
Ωm: 0.298 +0.009 -0.008
w0: -1
wa: 0
Chi squared: 10.27
Log Evidence: -15.5

===============================

Flat wCDM
rd: 147.09 +0.51 -0.51 Mpc
H0: 67.86 +1.22 -1.15 km/s/Mpc
Ωm: 0.297 +0.009 -0.009
w0: -0.915 +0.075 -0.080
wa: 0
Chi squared: 9.10
Log Evidence: -16.98
Degs of freedom: 10

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)
rd: 147.09 +0.51 -0.51 Mpc
H0: 66.74 +1.78 -1.64 km/s/Mpc
Ωm: 0.310 +0.013 -0.013
w0: -0.793 +0.145 -0.154
wa: -(9/4) * (1 + w0)
Chi squared: 8.33
Log Evidence: -15.98
Degs of freedom: 10

===============================

Flat w0waCDM
TODO
"""
