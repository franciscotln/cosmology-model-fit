from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2025BAO.data import get_data as get_bao_data
import cmb.data_planck_act_compression as cmb
import y2024BBN.prior_lcdm_schoneberg as bbn

c = cmb.c  # speed of light in km/s
Or_h2 = cmb.Omega_r_h2()

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

# arXiv:2503.14452v2 (ACT + Planck 2018)
theta_100 = 1.04094
theta_100_err = 0.00026

z_max = np.max(bao_data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (4 * one_plus_z**3 / (1 + 3 * one_plus_z**3)) ** (4 * (1 + w0))
    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


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
    rd = cmb.r_drag(wb=params[2], wm=params[1] * (params[0] / 100) ** 2)
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def theta_100_theory(params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    z_star = cmb.z_star(wb=Obh2, wm=Om * (H0 / 100) ** 2)
    rs_star = cmb.rs_z(Ez, z_star, params, H0, Obh2)
    DM_star = (1 + z_star) * cmb.DA_z(Ez, z_star, params, H0)
    return 100 * rs_star / DM_star


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_bbn = bbn.Obh2 - params[2]
    chi2_bbn = (delta_bbn / bbn.Obh2_sigma) ** 2

    delta_theta_100 = theta_100 - theta_100_theory(params)
    chi2_theta_100 = (delta_theta_100 / theta_100_err) ** 2

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi2_bbn + chi2_theta_100 + chi_bao


bounds = np.array(
    [
        (50, 80),  # H0
        (0.20, 0.50),  # Ωm
        (0.016, 0.030),  # Ωb * h^2
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
    from emcee import EnsembleSampler, autocorr, moves
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence
    from .plot_predictions import plot_bao_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 250
    nsteps = 2500 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    mvs = [
        (moves.KDEMove(), 0.30),
        (moves.DEMove(), 0.56),
        (moves.DESnookerMove(), 0.14),
    ]

    with Pool(5) as pool:
        sampler = EnsembleSampler(nwalkers, ndim, log_probability, pool, mvs)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    rd_samples = cmb.r_drag(wb=samples[:, 2], wm=Omh2_samples)
    z_star_samples = cmb.z_star(wb=samples[:, 2], wm=Omh2_samples)
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, [15.9, 50, 84.1])

    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"100 θ*: {theta_100_theory(best_fit):.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_corner_and_chains(
        labels=["$H_0$", "$Ω_m$", "$ω_b$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: DESI DR2 2024 + θ∗ + BBN
*******************************

Flat ΛCDM w(z) = -1
rd: 148.36 +0.68 -0.68 Mpc
H0: 68.48 +0.46 -0.46 km/s/Mpc
Ωm: 0.2963 +0.0045 -0.0044
ωb: 0.02217 +0.00053 -0.00053
w0: -1
wa: 0
r*: 145.56 Mpc
z*: 1088.55 +0.72 -0.69
100 θ*: 1.04096
Chi squared: 10.30
Log evidence: -16.7
Degs of freedom: 12

===============================

Flat wCDM w(z) = w0
rd: 148.56 +0.73 -0.74 Mpc
H0: 67.74 +1.11 -1.08 km/s/Mpc
Ωm: 0.3009 +0.0077 -0.0077
ωb: 0.02223 +0.00054 -0.00054
w0: -0.964 +0.046 -0.049 (prior width 1.5: -1.5 to 0..0)
wa: 0
r*: 145.77 Mpc
z*: 1088.39 +0.76 -0.73
100 θ*: 1.04092
Chi squared: 9.70
Log evidence: -18.9
Degs of freedom: 11

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)
rd: 148.51 +0.70 -0.70 Mpc
H0: 66.74 +1.57 -1.50 km/s/Mpc
Ωm: 0.3101 +0.0130 -0.0128
ωb: 0.02225 +0.00054 -0.00053
w0: -0.867 +0.115 -0.117 (prior width 1.5: -1.5 to 0..0)
r*: 145.73 Mpc
z*: 1088.36 +0.73 -0.71
100 θ*: 1.04092
Chi squared: 9.00
Log evidence: -17.6
Degs of freedom: 11

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
w0 - prior width 4.0: -2.5 to 1.5
wa - prior width 12.0: -8.0 to 4.0
"""
