from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_desi_compression as cmb
from y2022pantheonSHOES.data import get_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # km/s
Or_h2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, mb_values, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]
cho_sn_T = cho_sn.T
cho_bao_T = cho_bao.T

sn_grid = np.linspace(0, np.max(z_cmb), num=1000)
dx_sn = np.diff(sn_grid)
one_plus_z_hel = 1 + z_hel

bao_grid = np.linspace(0, np.max(bao_data["z"]), num=1000)
dx_bao = np.diff(bao_grid)


@njit
def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[4]
    h = H0 / 100
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode * rho_de)


@njit
def DM(grid, zs, dx, params):
    dh_grid = DH_z(grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    dms = np.interp(zs, grid, cum_dm)
    return dms


@njit
def apparent_mag(params):
    dL = one_plus_z_hel * DM(sn_grid, z_cmb, dx_sn, params)
    return params[3] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    return DM(bao_grid, z, dx_bao, params)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


def bao_theory(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    h = H0 / 100
    z_drag = cmb.z_drag(wb=Obh2, wm=Om * h**2)
    rd = cmb.rs_z(Ez, z_drag, params, H0, Obh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def solve_triang(L, L_T, delta):
    y = solve_triangular(L, delta, lower=True, check_finite=False)
    z = solve_triangular(L_T, y, lower=False, check_finite=False)
    return delta @ z


def chi_squared(params):
    H0, Om, Obh2 = params[0], params[1], params[2]

    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Obh2)
    chi2_cmb = np.dot(delta, np.dot(cmb.inv_cov_mat, delta))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, cho_bao_T, delta_bao)

    delta_sn = mb_values - apparent_mag(params)
    chi_sn = solve_triang(cho_sn, cho_sn_T, delta_sn)

    return chi2_cmb + chi_bao + chi_sn


bounds = np.array(
    [
        (60.0, 75.0),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # ωb = Ωb * h^2
        (-20.0, -19.0),  # M
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
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, pool=pool, moves=moves
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("Auto-correlation time", tau)
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("Effective samples:", nwalkers * ndim * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    M_16, M_50, M_84 = pct[3]
    w0_16, w0_50, w0_84 = pct[4]

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    z_st = cmb.z_star(Obh2_50, Omh2_50)
    z_d = cmb.z_drag(Obh2_50, Omh2_50)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"z*: {z_st:.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"z_d: {z_d:.2f}")
    print(f"rd: {cmb.rs_z(Ez, z_d, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evidence(samples, log_probs, log_probability, bounds):.2f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mb_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag(best_fit),
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$H_0$", "$Ω_m$", "$ω_b$", "M", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
H0: 68.36 +0.28 -0.29 km/s/Mpc
Ωm: 0.300 +0.004 -0.004
ωb: 0.02237 +0.00012 -0.00012
M: -19.413 +0.009 -0.009
z*: 1088.67
r*: 145.09 Mpc
z_d: 1059.69
rd: 147.68 Mpc
Chi squared: 1420.19
Log evidence: -727.14
Degrees of freedom: 1602

===============================

Flat wCDM w(z) = w0
H0: 67.78 +0.57 -0.56 km/s/Mpc
Ωm: 0.304 +0.005 -0.005
ωb: 0.02242 +0.00012 -0.00012
w0: -0.973 +0.023 -0.023 (prior width 1.5: -1.5 to 0.0)
M: -19.426 +0.014 -0.014
z*: 1088.57
r*: 145.21 Mpc
z_d: 1059.76
rd: 147.79 Mpc
Chi squared: 1418.84
Log evidence: -729.70
Degrees of freedom: 1601

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 67.42 +0.58 -0.58 km/s/Mpc
Ωm: 0.307 +0.006 -0.005
ωb: 0.02241 +0.00012 -0.00012
w0: -0.931 +0.037 -0.037 (prior width 1.5: -1.5 to 0.0)
M: -19.431 +0.013 -0.013
z*: 1088.57
r*: 145.21 Mpc
z_d: 1059.76
rd: 147.79 Mpc
Chi squared: 1416.88
Log evidence: -728.24
Degrees of freedom: 1601

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.51 +0.58 -0.58 km/s/Mpc
Ωm: 0.310 +0.006 -0.005
ωb: 0.02230 +0.00013 -0.00013
w0: -0.853 +0.054 -0.053 (prior width 1.5: -1.5 to 0.0)
wa: -0.498 +0.201 -0.213 (prior width 3.0: -2.0 to 1.0)
M: -19.420 +0.014 -0.014
z*: 1088.79
r*: 144.91 Mpc
z_d: 1059.60
rd: 147.51 Mpc
Chi squared: 1413.05
Log evidence: -728.35
Degrees of freedom: 1600
"""
