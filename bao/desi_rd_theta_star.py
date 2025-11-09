from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from y2025BAO.data import get_data as get_bao_data
import y2025cmb_actbase_lcdm_camb.data as cmb

c = c0 / 1000  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2()

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

z_grid = np.linspace(0, np.max(bao_data["z"]) + 0.1, num=2000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    h, Om, w0 = params[1] / 100, params[2], params[3]
    Or = Orh2 / h**2
    Ol = 1 - Om - Or
    inv_a = 1 + z
    cubic = inv_a**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Or * inv_a**4 + Om * cubic + Ol * rho_de)


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
def bao_theory(z, qty, rd, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    distances = cmb.cmb_distances(Ez, params, params[1], params[2], params[0])
    rd = distances[1]
    delta_cmb = cmb.DISTANCE_PRIORS - distances
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, rd, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi_bao + chi2_cmb


bounds = np.array(
    [
        (0.010, 0.030),  # Obh2
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
    obh2_err, H0_err, Om_err, w0_err = np.diff(pct, axis=0).T

    Omh2_samples = samples[:, 2] * (samples[:, 1] / 100) ** 2
    Omh2_50 = np.percentile(Omh2_samples, 50)
    Omh2_err = np.diff(np.percentile(Omh2_samples, [15.9, 50, 84.1]))

    thetastar_best, rd_best = cmb.cmb_distances(
        Ez, best_fit, best_fit[1], best_fit[2], best_fit[0]
    )

    print("Gelman-Rubin:", gelman_rubin(chains_samples))
    print(f"ωb: {best_fit[0]:.5f} +{obh2_err[1]:.5f} -{obh2_err[0]:.5f}")
    print(f"H0: {best_fit[1]:.2f} +{H0_err[1]:.2f} -{H0_err[0]:.2f} km/s/Mpc")
    print(f"Ωm: {best_fit[2]:.3f} +{Om_err[1]:.3f} -{Om_err[0]:.3f}")
    print(f"w0: {best_fit[3]:.3f} +{w0_err[1]:.3f} -{w0_err[0]:.3f}")
    print(f"ωm: {Omh2_50:.5f} +{Omh2_err[1]:.5f} -{Omh2_err[0]:.5f}")
    print(f"rdrag: {rd_best:.2f} Mpc")
    print(f"100 θ*: {thetastar_best:.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {1 + len(bao_data['z']) - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, rd_best, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_corner_and_chains(
        labels=["$ω_b$", "$H_0$", "$Ω_m$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
*******************************
DESI BAO DR2 2025 + (100 θ*, rdrag)CMB
*******************************

Flat ΛCDM w(z) = -1

-- ACT DR6 --
H0: 69.82 +0.52 -0.51 km/s/Mpc
ωb: 0.02413 +0.00051 -0.00050
ωm: 0.14172 +0.00084 -0.00083
Ωm: 0.291 +0.005 -0.005
w0: -1
wa: 0
rdrag: 145.98 Mpc
100 θ*: 1.04079
Chi squared: 11.14
Log Evidence: -17.32
Degs of freedom: 11

===============================

Flat wCDM w(z) = w0

-- ACT DR6 --
H0: 68.57 +1.00 -0.96 km/s/Mpc
ωb: 0.02472 +0.00069 -0.00066
ωm: 0.14034 +0.00129 -0.00134
Ωm: 0.298 +0.007 -0.007
w0: -0.928 +0.048 -0.051 (prior width 1.5: -1.5 to 0.0)
wa: 0
rdrag: 145.87 Mpc
100 θ*: 1.04073
Chi squared: 9.10
Log Evidence: -18.78
Degs of freedom: 10

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)

-- ACT DR6 --
H0: 67.34 +1.55 -1.44 km/s/Mpc
ωb: 0.02456 +0.00058 -0.00058
ωm: 0.14084 +0.00100 -0.00100
Ωm: 0.311 +0.013 -0.013
w0: -0.798 +0.115 -0.120 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
rdrag: 145.86 Mpc
100 θ*: 1.04074
Chi squared: 8.34
Log Evidence: -17.53
Degs of freedom: 10

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)

-- ACT DR6 --
H0: 65.19 +2.63 -2.62 km/s/Mpc
ωb: 0.02372 +0.00091 -0.00076
ωm: 0.14307 +0.00167 -0.00212
Ωm: 0.337 +0.032 -0.029
w0: -0.559 +0.314 -0.278
wa: -1.206 +0.889 -1.017
rdrag: 145.96 Mpc
100 θ*: 1.04068
Chi squared: 7.88
Log Evidence: -19.90
Degs of freedom: 9
"""
