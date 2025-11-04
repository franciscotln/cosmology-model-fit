from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_desi_sub_compression as cmb
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

c = c0 / 1000  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    h, Om, w0 = params[2] / 100, params[3], params[4]
    Or = Orh2 / h**2
    inv_a3 = (1 + z) ** 3
    rho_de = (4 * inv_a3 / (1 + 3 * inv_a3)) ** (4 * (1 + w0))
    return np.sqrt(Or * (1 + z) ** 4 + Om * inv_a3 + (1 - Om - Or) * rho_de)


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


def chi_squared(params):
    wb, H0, Om = params[1], params[2], params[3]

    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, wb)
    chi2_rd = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi_sn + chi_bao + chi2_rd


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
    rd_samples = cmb.r_drag(samples[:, 1], samples[:, 3] * (samples[:, 2] / 100) ** 2)
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
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

Planck's individual priors for 100 x θ* and r_d as independent
H0: 68.76 +0.41 -0.41 km/s/Mpc
r_d: 147.13 +0.25 -0.25 Mpc
ωb: 0.02282 +0.00026 -0.00026
Ωm: 0.298 +0.004 -0.004
w0: -1
wa: 0
Chi squared: 1662.8
Log evidence: -849.6
Degrees of freedom: 1745

----

Planck's priors for 100 x θ* and r_d with estimated correlation ρ = 0.2932
H0: 68.76 +0.41 -0.41 km/s/Mpc
r_d: 147.14 +0.25 -0.25 Mpc
ωb: 0.02282 +0.00027 -0.00027
Ωm: 0.298 +0.004 -0.004
w0: -1
wa: 0
Chi squared: 1662.9
Log evidence: -849.7
Degrees of freedom: 1745

----

DESI sub-compression (100 x θ*, r_d)CMB with covariance
H0: 68.65 +0.41 -0.41 km/s/Mpc
r_d: 147.36 +0.27 -0.27 Mpc
ωb: 0.02267 +0.00028 -0.00028
Ωm: 0.298 +0.004 -0.004
w0: -1
wa: 0
Chi squared: 1662.7
Log evidence: -849.7
Degrees of freedom: 1745
"""


"""
Flat wCDM w(z) = w0

Planck's individual priors for 100 x θ* and r_d as independent
H0: 67.34 +0.55 -0.54 km/s/Mpc
r_d: 147.07 +0.25 -0.25 Mpc
ωb: 0.02369 +0.00036 -0.00036
Ωm: 0.305 +0.005 -0.005
w0: -0.898 +0.026 -0.027 (prior width 1.5: -1.5 to 0.0)
wa: 0
Chi squared: 1649.1
Log evidence: -845.9 (Δ logZ = 3.7 against ΛCDM)
Degrees of freedom: 1744

----

Planck's priors for 100 x θ* and r_d with estimated correlation ρ = 0.2932
H0: 67.33 +0.55 -0.54 km/s/Mpc
r_d: 147.07 +0.25 -0.25 Mpc
ωb: 0.02370 +0.00037 -0.00037
Ωm: 0.305 +0.005 -0.005
w0: -0.898 +0.026 -0.027 (prior width 1.5: -1.5 to 0.0)
wa: 0
Chi squared: 1649.1
Log evidence: -845.9 (Δ logZ = 3.8 against ΛCDM)
Degrees of freedom: 1744

----

DESI sub-compression (100 x θ*, r_d)CMB with covariance
H0: 67.24 +0.54 -0.54 km/s/Mpc
r_d: 147.28 +0.27 -0.27 Mpc
ωb: 0.02354 +0.00038 -0.00037
Ωm: 0.305 +0.005 -0.005
w0: -0.899 +0.026 -0.027 (prior width 1.5: -1.5 to 0.0)
wa: 0
Chi squared: 1649.2
Log evidence: -846.0 (Δ logZ = 3.7 against ΛCDM)
Degrees of freedom: 1744
"""


"""
Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)

Planck's individual priors for 100 x θ* and r_d as independent
H0: 67.04 +0.57 -0.56 km/s/Mpc
r_d: 147.08 +0.25 -0.25 Mpc
ωb: 0.02331 +0.00028 -0.00029
Ωm: 0.310 +0.006 -0.005
w0: -0.826 +0.042 -0.042 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/d z at z=0 = -(9/4) * (1 + w0)
Chi squared: 1646.4
Log evidence: -844.0 (Δ logZ = 5.6 against ΛCDM)
Degrees of freedom: 1744

----

Planck's priors for 100 x θ* and r_d with estimated correlation ρ = 0.2932
H0: 67.04 +0.57 -0.57 km/s/Mpc
r_d: 147.08 +0.25 -0.26 Mpc
ωb: 0.02332 +0.00030 -0.00030
Ωm: 0.310 +0.006 -0.005
w0: -0.826 +0.042 -0.042 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/d z at z=0 = -(9/4) * (1 + w0)
Chi squared: 1646.4
Log evidence: -844.1 (Δ logZ = 5.6 against ΛCDM)
Degrees of freedom: 1744

----

DESI sub-compression (100 x θ*, r_d)CMB with covariance
H0: 66.94 +0.57 -0.56 km/s/Mpc
r_d: 147.30 +0.27 -0.27 Mpc
ωb: 0.02316 +0.00030 -0.00030
Ωm: 0.311 +0.006 -0.005
w0: -0.827 +0.042 -0.042 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/d z at z=0 = -(9/4) * (1 + w0)
Chi squared: 1646.4
Log evidence: -844.2 (Δ logZ = 5.5 against ΛCDM)
Degrees of freedom: 1744
"""


"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)

Planck's individual priors for 100 x θ* and r_d as independent
H0: 66.99 +0.56 -0.56 km/s/Mpc
r_d: 147.09 +0.25 -0.25 Mpc
ωb: 0.02282 +0.00050 -0.00045
Ωm: 0.314 +0.006 -0.006
w0: -0.794 +0.061 -0.059 (prior width 1.5: -1.5 to 0.0)
wa: -0.560 +0.284 -0.297 (prior width 3.5: -2.5 to 1.0)
Chi squared: 1645.9
Log evidence: -845.6 (Δ logZ = 4.0 against ΛCDM)
Degrees of freedom: 1743

----

Planck's priors for 100 x θ* and r_d with estimated correlation ρ = 0.2932
H0: 66.99 +0.58 -0.57 km/s/Mpc
r_d: 147.10 +0.25 -0.25 Mpc
ωb: 0.02284 +0.00052 -0.00046
Ωm: 0.314 +0.007 -0.007
w0: -0.795 +0.062 -0.061 (prior width 1.5: -1.5 to 0.0)
wa: -0.554 +0.289 -0.302 (prior width 3.5: -2.5 to 1.0)
Chi squared: 1645.9
Log evidence: -845.6 (Δ logZ = 4.1 against ΛCDM)
Degrees of freedom: 1743

----

DESI sub-compression (100 x θ*, r_d)CMB with covariance
H0: 66.90 +0.57 -0.57 km/s/Mpc
r_d: 147.31 +0.27 -0.27 Mpc
ωb: 0.02267 +0.00052 -0.00047
Ωm: 0.314 +0.007 -0.007
w0: -0.795 +0.063 -0.060 (prior width 1.5: -1.5 to 0.0)
wa: -0.563 +0.291 -0.304 (prior width 3.5: -2.5 to 1.0)
Chi squared: 1646.0
Log evidence: -845.7 (Δ logZ = 4.0 against ΛCDM)
Degrees of freedom: 1743
"""
