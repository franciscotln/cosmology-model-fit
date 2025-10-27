from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    Om, w0 = params[3], params[4]
    inv_a3 = (1 + z) ** 3
    rho_de = (2 * inv_a3 / (1 + inv_a3)) ** (2 * (1 + w0))

    return np.sqrt(Om * inv_a3 + (1 - Om) * rho_de)


@njit
def theory_mu(params):
    dL = (1 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
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
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[1]


"""
Planck prior on sound horizon at drag epoch r_d in Mpc
width increased by 75% to account for model dependence
"""
rd_planck = 147.09
rd_planck_sigma = 1.75 * 0.26


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    chi2_prior = ((rd_planck - params[1]) / rd_planck_sigma) ** 2

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao + chi2_prior


bounds = np.array(
    [
        (-0.4, 0.4),  # ΔM
        (120.0, 160.0),  # r_d
        (50.0, 90.0),  # H0
        (0.1, 0.8),  # Ωm
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
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions

    np.random.seed(42)
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

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, pool=pool, moves=moves
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    [
        (dM_16, dM_50, dM_84),
        (rd_16, rd_50, rd_84),
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {1 + len(bao_data['z']) + sn_size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$Δ_M$", "$r_d$", "$H_0$", "$Ω_M$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM w(z) = -1
ΔM: -0.056 +0.012 -0.013 mag
r_d: 147.09 +0.45 -0.44 Mpc
H0: 68.36 +0.49 -0.49 km/s/Mpc
Ωm: 0.310 +0.008 -0.008
w0: -1
Chi squared: 1659.0
Log evidence: -845.4
Degrees of freedom: 1745

===============================

Flat wCDM w(z) = w0
ΔM: -0.063 +0.013 -0.013 mag
r_d: 147.09 +0.44 -0.44 Mpc
H0: 67.21 +0.58 -0.58 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.872 +0.038 -0.037 (prior width 1.5: -1.5 to 0.0)
Chi squared: 1648.1
Log evidence: -842.7 (Δ logZ = 2.7 against ΛCDM)
Degrees of freedom: 1744

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
ΔM: -0.061 +0.012 -0.012 mag
r_d: 147.09 +0.45 -0.44 Mpc
H0: 67.05 +0.59 -0.58 km/s/Mpc
Ωm: 0.307 +0.008 -0.008
w0: -0.834 +0.044 -0.045 (prior width 1.5: -1.5 to 0.0)
Chi squared: 1646.5
Log evidence: -841.7 (Δ logZ = 3.7 against ΛCDM)
Degrees of freedom: 1744

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
ΔM: -0.057 +0.013 -0.013 mag
r_d: 147.09 +0.44 -0.45 Mpc
H0: 66.98 +0.60 -0.58 km/s/Mpc
Ωm: 0.321 +0.012 -0.015
w0: -0.783 +0.071 -0.068 (prior width 1.5: -1.5 to 0.0)
wa: -0.720 +0.446 -0.447 (prior width 5.0: -3.0 to 2.0)
Chi squared: 1645.5
Log evidence: -842.7 (Δ logZ = 2.7 against ΛCDM)
Degrees of freedom: 1743
"""
