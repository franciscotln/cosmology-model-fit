from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s
rd = 147.09  # Mpc, fixed

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    O_m, w0 = params[2], params[3]
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubic + (1 - O_m) * rho_de)


@njit
def mu_theory(params):
    dL = (1 + z_sn_vals) * DM_z(z_sn_vals, params)
    return params[0] + 25 + 5 * np.log10(dL)


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


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

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
    return results / rd


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_sn = mu_vals - mu_theory(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi_sn + chi_bao


bounds = np.array(
    [
        (-0.7, 0.7),  # ΔM
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (-1.5, 0.0),  # w0
    ],
    dtype=np.float64,
)

normalization = -np.log(np.prod(bounds[:, 1] - bounds[:, 0]))


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
    from sn.plotting import plot_predictions as plot_sn_predictions
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

    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    [
        [dM_16, dM_50, dM_84],
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evidence(samples, log_probs, log_probability):.2f}")
    print(f"Degs of freedom: {bao_data['value'].size + z_sn_vals.size - len(best_fit)}")

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
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$Δ_M$", "$H_0$", "$Ω_m$", "$w_0$"],
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
H0: 68.69 +0.47 -0.47 km/s/Mpc
Ωm: 0.304 +0.008 -0.008
Chi squared: 38.82
Log Evidence: -27.88
Degs of freedom: 32

===============================

Flat wCDM
H0: 67.12 +0.75 -0.73 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.866 +0.051 -0.051 (prior width 1.5: -1.5 to 0.0)
Chi squared: 32.16
Log Evidence: -27.01 (Δ logZ = 0.87 over ΛCDM)
Degs of freedom: 31

===============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.66 +0.82 -0.80 km/s/Mpc
Ωm: 0.310 +0.009 -0.008
w0: -0.802 +0.066 -0.066 (prior width 1.5: -1.5 to 0.0)
Chi squared: 30.37
Log Evidence: -25.86 (Δ logZ = 2.02 over ΛCDM)
Degs of freedom: 31

===============================

Flat w0waCDM
H0: 66.21 +0.90 -0.88 km/s/Mpc
Ωm: 0.331 +0.016 -0.017
w0: -0.697 +0.112 -0.108 (prior width 1.5: -1.5 to 0.0)
wa: -1.008 +0.549 -0.551 (prior width 4.0: -3.0 to 1.0)
Chi squared: 28.79
Log Evidence: -26.40 (Δ logZ = 1.48 over ΛCDM)
Degs of freedom: 30
"""
