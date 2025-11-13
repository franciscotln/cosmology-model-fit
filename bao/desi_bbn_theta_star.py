from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2025BAO.data import get_data as get_bao_data
import cmb.data_planck_act_compression as cmb
import y2024BBN.prior_lcdm_schoneberg as bbn

c = cmb.c  # speed of light in km/s
Orh2 = cmb.Omega_r_h2(2.044)
Omnu_h2 = cmb.Omnu_h2

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

# arXiv:2503.14452v2 (ACT + Planck 2018)
theta_100 = 1.04094
theta_100_err = 0.00026

z_max = np.max(bao_data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, Obc, Or, w0=-1, wa=0):
    Ol = 1 - Obc - Or
    inv_a = 1 + z
    cubic = inv_a**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Or * inv_a**4 + Obc * cubic + Ol * rho_de)


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0 = params
    h = H0 / 100
    Obc = (Obh2 + Och2 + Omnu_h2) / h**2
    Or = Orh2 / h**2
    return H0 * Ez(z, Obc, Or, w0)


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
    Obh2, Och2 = params[1], params[2]
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnu_h2)
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_bbn = bbn.Obh2 - params[1]
    chi2_bbn = (delta_bbn / bbn.Obh2_sigma) ** 2

    lA = cmb.cmb_distances(Ez, *params)[1]
    thetastar = 100 * np.pi / lA
    chi2_theta_100 = ((theta_100 - thetastar) / theta_100_err) ** 2

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi2_bbn + chi2_theta_100 + chi_bao


bounds = np.array(
    [
        (50, 90),  # H0
        (0.010, 0.030),  # Ωb * h^2
        (0.05, 0.30),  # Ωc * h^2
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
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 1] + samples[:, 2] + Omnu_h2
    Om_samples = Omh2_samples / (samples[:, 0] / 100) ** 2
    rd_samples = cmb.r_drag(wb=samples[:, 1], wm=Omh2_samples)
    z_star_samples = cmb.z_star(wb=samples[:, 1], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, [15.9, 50, 84.1])

    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, *best_fit):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"100 θ*: {100 * np.pi / (cmb.cmb_distances(Ez, *best_fit)[1]):.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_corner_and_chains(
        labels=["$H_0$", "$ω_b$", "$ω_c$", "$w_0$"],
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
rd: 148.35 +0.69 -0.68 Mpc
H0: 68.49 +0.46 -0.46 km/s/Mpc
ωb: 0.02217 +0.00052 -0.00053
ωc: 0.1162 +0.0008 -0.0008
ωm: 0.1390 +0.0011 -0.0011
Ωm: 0.2963 +0.0044 -0.0044
w0: -1
wa: 0
r*: 145.60 Mpc
z*: 1089.85 +0.72 -0.69
100 θ*: 1.04094
Chi squared: 10.30
Log evidence: -17.9

===============================

Flat wCDM w(z) = w0
rd: 148.53 +0.74 -0.73 Mpc
H0: 67.77 +1.11 -1.08 km/s/Mpc
ωb: 0.02224 +0.00054 -0.00054
ωc: 0.1152 +0.0015 -0.0016
ωm: 0.1381 +0.0016 -0.0017
Ωm: 0.3007 +0.0076 -0.0076
w0: -0.966 +0.047 -0.049 (prior width 1.5: -1.5 to 0.0)
wa: 0
r*: 145.79 Mpc
z*: 1089.68 +0.76 -0.73
100 θ*: 1.04091
Chi squared: 9.71
Log evidence: -20.1
Degs of freedom: 11

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)
rd: 148.48 +0.71 -0.70 Mpc
H0: 66.84 +1.60 -1.55 km/s/Mpc
ωb: 0.02225 +0.00054 -0.00054
ωc: 0.1153 +0.0012 -0.0012
ωm: 0.1382 +0.0013 -0.0013
Ωm: 0.3094 +0.0134 -0.0131
w0: -0.873 +0.118 -0.118 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
r*: 145.76 Mpc
z*: 1089.67 +0.74 -0.71
100 θ*: 1.04093
Chi squared: 9.00
Log evidence: -18.9
Degs of freedom: 11

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
w0 - prior width 4.0: -2.5 to 1.5
wa - prior width 12.0: -8.0 to 4.0
"""
