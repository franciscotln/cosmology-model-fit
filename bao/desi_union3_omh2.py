from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from scipy.constants import c as c0
from y2023union3.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    Om, w0 = params[2], params[3]
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Om * cubic + (1 - Om) * rho_de)


@njit
def mu_theory(params):
    dL = (1 + z_sn_vals) * DM_z(z_sn_vals, params)
    return params[-1] + 25 + 5 * np.log10(dL)


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


"""
Planck prior on Ωm * h^2
Fit rs(drag) directly as a free parameter without early universe constraints
"""
Omh2_planck = 0.1430
Omh2_planck_sigma = 0.0011


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    Omh2 = params[2] * params[1] ** 2 / 100**2
    chi2_prior = ((Omh2_planck - Omh2) / Omh2_planck_sigma) ** 2

    delta_sn = mu_vals - mu_theory(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao + chi2_prior


bounds = np.array(
    [
        (120, 160),  # rd
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (-1.5, 0.0),  # w0
        (-0.7, 0.7),  # ΔM
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
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
    ]

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, pool=pool, moves=moves
        )
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

    [
        (rd_16, rd_50, rd_84),
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (w0_16, w0_50, w0_84),
        (dM_16, dM_50, dM_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    degs_of_freedom = 1 + bao_data["value"].size + z_sn_vals.size - len(best_fit)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degs of freedom: {degs_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$r_d$", "$H_0$", "$Ω_m$", "$w_0$", "$Δ_M$"],
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
rd: 147.30 +1.28 -1.27 Mpc
H0: 68.59 +0.97 -0.97 km/s/Mpc
Ωm: 0.304 +0.008 -0.008
w0: -1
Chi squared: 38.8
Log evidence: -31.2
Degs of freedom: 32

===============================

Flat wCDM
rd: 142.52 +2.40 -2.59 Mpc
H0: 69.32 +1.11 -1.07 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.865 +0.051 -0.051 (prior width 1.5: -1.5 to 0.0)
wa: 0
Chi squared: 32.2
Log evidence: -30.3 (Δ logZ = 0.9 over ΛCDM)
Degs of freedom: 31

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)
rd: 144.48 +1.60 -1.60 Mpc
H0: 67.75 +1.00 -1.00 km/s/Mpc
Ωm: 0.311 +0.009 -0.009
w0: -0.774 +0.073 -0.074
wa: -(9/4) * (1 + w0)
Chi squared: 30.1
Log evidence: -28.9 (Δ logZ = 2.3 over ΛCDM)
Degs of freedom: 31

===============================

Flat w0waCDM
rd: 148.14 +2.38 -3.00 Mpc
H0: 65.74 +1.82 -1.51 km/s/Mpc
Ωm: 0.331 +0.015 -0.017
w0: -0.696 +0.113 -0.108 (prior width 1.5: -1.5 to 0.0)
wa: -1.011 +0.546 -0.559 (prior width 4.0: -3.0 to 1.0)
Chi squared: 28.8
Log evidence: -29.8 (Δ logZ = 1.4 over ΛCDM)
Degs of freedom: 30
"""
