from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2025BAO.data import get_data as get_bao_data
import cmb.data_desi_compression as cmb

c = cmb.c  # speed of light in km/s

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]
cmb_cho = cho_factor(cmb.covariance, lower=True)[0]

Orh2 = cmb.Omega_r_h2()

z_grid = np.linspace(0, np.max(bao_data["z"]) + 0.1, num=1000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    h, Om, w0 = params[0] / 100, params[1], params[3]
    Or = Orh2 / h**2
    Ode = 1 - Om - Or

    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (4 * cubed / (1 + 3 * cubed)) ** (4 * (1 + w0))

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


def bao_theory(z, qty, params):
    h, Om, Obh2 = params[0] / 100, params[1], params[2]
    zd = cmb.z_drag(wb=Obh2, wm=Om * h**2)
    rd = cmb.rs_z(Ez, zd, params, params[0], Obh2)

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
    H0, Om, Ob_h2 = params[0], params[1], params[2]

    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = solve_triang(cmb_cho, delta_cmb)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi2_cmb + chi_bao


bounds = np.array(
    [
        (55, 75),  # H0
        (0.15, 0.50),  # Ωm
        (0.021, 0.023),  # ωb = Ωb * h^2
        (-1.5, 0.0),  # w0
    ],
    dtype=np.float64,
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


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
    from corner_plot import plot_corner_and_chains
    from gelman_rubin import gelman_rubin
    from .plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 200
    burn_in = 200
    nsteps = 2200 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
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
    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    one_sigma_contours = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_contours, axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Om_h2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_st_samples = cmb.z_star(samples[:, 2], Om_h2_samples)
    r_d_samples = cmb.r_drag(samples[:, 2], Om_h2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Om_h2_samples, one_sigma_contours)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, one_sigma_contours)
    rd_16, rd_50, rd_84 = np.percentile(r_d_samples, one_sigma_contours)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

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
Dataset: DESI DR2 2024 + (θ∗,ωb,ωbc)CMB
*******************************

Flat ΛCDM w(z) = -1
H0: 68.39 +0.29 -0.29 km/s/Mpc
Ωm: 0.2998 +0.0037 -0.0037
ωm: 0.14021 +0.00062 -0.00061
ωb: 0.02237 +0.00012 -0.00012
w0: -1
wa: 0
r*: 145.12 Mpc
z*: 1088.45 +0.18 -0.18
r_d: 147.84 +0.18 -0.18 Mpc
Chi squared: 13.52
Degs of freedom: 15

===============================

Flat wCDM w(z) = w0
H0: 68.85 +0.97 -0.91 km/s/Mpc
Ωm: 0.2964 +0.0073 -0.0074
ωm: 0.14055 +0.00087 -0.00088
ωb: 0.02235 +0.00013 -0.00013
w0: -1.020 +0.037 -0.040 (prior width -1.5 to 0.0)
wa: 0
r*: 145.05 Mpc
z*: 1088.52 +0.23 -0.22
r_d: 147.77 +0.22 -0.22 Mpc
Chi squared: 13.31
Degs of freedom: 14

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)
H0: 68.05 +1.54 -1.43 km/s/Mpc
Ωm: 0.3025 +0.0125 -0.0126
ωm: 0.14012 +0.00078 -0.00078
ωb: 0.02238 +0.00013 -0.00013
w0: -0.976 +0.104 -0.109 (prior width -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
r*: 145.14 Mpc
z*: 1088.43 +0.21 -0.21
r_d: 147.85 +0.21 -0.21 Mpc
Chi squared: 13.46
Degs of freedom: 14

===============================

Flat w(z) = w0 + wa * z / (1 + z)
Overfits, the uncertainties go wild and the prior are very wide
The posterior volume is also very large, making the evidence small

H0: 63.74 +2.04 -2.06 km/s/Mpc
Ωm: 0.3499 +0.0249 -0.0225
ωm: 0.14217 +0.00097 -0.00101
ωb: 0.02222 +0.00013 -0.00013
w0: -0.454 +0.249 -0.228 (prior width -2.0 to +1.5)
wa: -1.600 +0.656 -0.755 (prior width -6.0 to 2.5)
r*: 144.69 Mpc
z*: 1088.87 +0.19 -0.19
r_d: 147.45 +0.22 -0.21 Mpc
Chi squared: 7.02
Degs of freedom: 13
"""
