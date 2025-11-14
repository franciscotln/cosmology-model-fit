from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
import y2025cmb_p_actbase_lcdm_camb.data as cmb
from y2025DESdovekie.data import get_data, effective_sample_size as sn_size
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
    rho_de = (4 * inv_a3 / (1 + 3 * inv_a3)) ** (4 * (1 + w0))
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
    rd = cmb.r_drag(wb=params[1], wm=params[3] * (params[2] / 100) ** 2)
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


rd_prior = cmb.DISTANCE_PRIORS[1]
rd_prior_std = cmb.covariance[1, 1] ** 0.5


def chi_squared(params):
    Omh2 = params[3] * (params[2] / 100) ** 2
    delta_rd_prior = rd_prior - cmb.r_drag(params[1], Omh2)
    chi2_prior = (delta_rd_prior / rd_prior_std) ** 2

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao + chi2_prior


bounds = np.array(
    [
        (-0.4, 0.4),  # ΔM
        (0.010, 0.030),  # Ob * h^2
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
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
        (Obh2_16, Obh2_50, Obh2_84),
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)
    Omh2_samples = samples[:, 3] * (samples[:, 2] / 100) ** 2
    rd_samples = cmb.r_drag(samples[:, 1], Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
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
        labels=["$Δ_M$", "$ω_b$", "$H_0$", "$Ω_m$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM w(z) = -1
H0: 68.55 +0.45 -0.45 km/s/Mpc
r_d: 147.14 +0.29 -0.29 Mpc
ωb: 0.02207 +0.00064 -0.00065
ωm: 0.1439 +0.0021 -0.0021
Ωm: 0.306 +0.008 -0.007
w0: -1
wa: 0
Chi squared: 1645.3
Log evidence: -838.0
Degrees of freedom: 1724

===============================

Flat wCDM w(z) = w0
H0: 67.76 +0.55 -0.55 km/s/Mpc
r_d: 147.14 +0.29 -0.29 Mpc
ωb: 0.02442 +0.00130 -0.00122
ωm: 0.1365 +0.0038 -0.0039
Ωm: 0.297 +0.008 -0.008
w0: -0.908 +0.037 -0.037 (prior width 1.5: -1.5 to 0.0)
wa: 0
Chi squared: 1639.5
Log evidence: -837.9 (Δ logZ = 0.1 against ΛCDM)
Degrees of freedom: 1723

===============================

H0: 67.62 +0.56 -0.56 km/s/Mpc
r_d: 147.14 +0.29 -0.29 Mpc
ωb: 0.02356 +0.00088 -0.00087
ωm: 0.1391 +0.0027 -0.0027
Ωm: 0.304 +0.007 -0.007
w0: -0.868 +0.048 -0.048 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/d z at z=0 = -(9/4) * (1 + w0)
Chi squared: 1638.5
Log evidence: -837.1 (Δ logZ = 0.9 against ΛCDM)
Degrees of freedom: 1723

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 67.58 +0.57 -0.56 km/s/Mpc
r_d: 147.14 +0.28 -0.29 Mpc
ωb: 0.02225 +0.00208 -0.00152
ωm: 0.1433 +0.0050 -0.0065
Ωm: 0.314 +0.013 -0.016
w0: -0.845 +0.069 -0.064 (prior width 1.5: -1.5 to 0.0)
wa: -0.521 +0.444 -0.444 (prior width 5.5: -3.5 to 2.0)
Chi squared: 1638.1
Log evidence: -838.6 (Δ logZ = -0.6 in favour of ΛCDM)
Degrees of freedom: 1722
"""
