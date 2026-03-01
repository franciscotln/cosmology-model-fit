from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
import y2025cmb_p_actbase_lcdm_camb.data as cmb
from y2025DESdovekie.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    inv_a3 = (1.0 + z) ** 3
    # thawing quintessence
    return (2 * inv_a3 / (1.0 + w0 + (1.0 - w0) * inv_a3)) ** 2


@njit
def Ez(z, params):
    Om = params[3]
    inv_a3 = (1.0 + z) ** 3
    return np.sqrt(Om * inv_a3 + (1.0 - Om))


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
desi_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


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


pivot_mask = z_cmb <= 0.11


@njit
def mu_corr(params):
    z_pec = 100 * params[4] / c
    z_cosmo1 = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    z_cosmo2 = -1.0 + (1.0 + z_cmb) / (1.0 - z_pec)

    DM_ref = DM_z(z_cmb, params)

    return np.where(
        pivot_mask,
        5.0 * np.log10(DM_z(z_cosmo1, params) / DM_ref),
        5.0 * np.log10(DM_z(z_cosmo2, params) / DM_ref),
    )


@njit
def theory_mu(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


rd_prior = cmb.DISTANCE_PRIORS[1]
rd_prior_std = cmb.covariance[1, 1] ** 0.5


def chi_squared(params):
    Omh2 = params[3] * (params[2] / 100) ** 2
    delta_rd_prior = rd_prior - cmb.r_drag(params[1], Omh2)
    chi2_prior = (delta_rd_prior / rd_prior_std) ** 2

    delta_sn = mu_values - theory_mu(params) - mu_corr(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, params)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao + chi2_prior


bounds = np.array(
    [
        (-0.4, 0.4),  # ΔM
        (0.010, 0.030),  # Ob * h^2
        (50.0, 90.0),  # H0
        (0.1, 0.8),  # Ωm
        (-6.0, 2),  # v
    ]
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
    from bao.plot_predictions import plot_bao_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.25),
        (emcee.moves.DEMove(), 0.75),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
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
        (v_16, v_50, v_84),
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
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {1 + len(bao_data) + sn_size - len(best_fit)}")

    labels = ["$Δ_M$", "$ω_b$", "$H_0$", "$Ω_m$", "$v$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(best_fit),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
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
Chi squared: 1645.3
Log evidence: -838.0
Degrees of freedom: 1724
"""

"""
Flat ΛCDM
Isotropic velocity SNe observed redshifts (turning point z <= 0.11 inflow z > 0.11 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

ΔM: -0.048 +0.012 -0.012 mag
v: -1.60 +0.57 -0.57 x 100 km/s (prior ~U(-6, 2)) x 100 km/s
v / (z_cut=0.11): -1455 ± 518 km/s
H0: 68.90 +0.48 -0.48 km/s/Mpc
r_d: 147.14 +0.30 -0.30 Mpc
ωb: 0.02258 +0.00069 -0.00068
ωm: 0.1422 +0.0022 -0.0022
Ωm: 0.300 +0.008 -0.008
Chi squared: 1637.4 (2.81 sigma significance)
Log evidence: -835.8 (Δ logZ = 2.2 in favour of corrections)
Degrees of freedom: 1723
"""

"""
Flat wCDM w(z) = w0
H0: 67.76 +0.55 -0.55 km/s/Mpc
r_d: 147.14 +0.29 -0.29 Mpc
ωb: 0.02442 +0.00130 -0.00122
ωm: 0.1365 +0.0038 -0.0039
Ωm: 0.297 +0.008 -0.008
w0: -0.908 +0.037 -0.037 (prior ~U(-1.5, 0.0))
wa: 0
Chi squared: 1639.5 (2.41 sigma away from ΛCDM)
Log evidence: -837.9 (Δ logZ = 0.1 against ΛCDM)
Degrees of freedom: 1723

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
H0: 67.60 +0.57 -0.57 km/s/Mpc
r_d: 147.14 +0.30 -0.30 Mpc
ωb: 0.02350 +0.00090 -0.00087
ωm: 0.1393 +0.0028 -0.0028
Ωm: 0.305 +0.008 -0.008
w0: -0.860 +0.051 -0.052 (prior ~U(-1.0, -1/3))
Chi squared: 1638.5 (2.61 sigma away from ΛCDM)
Log evidence: -836.2 (Δ logZ = 1.8 against ΛCDM)
Degrees of freedom: 1723

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 67.58 +0.57 -0.56 km/s/Mpc
r_d: 147.14 +0.28 -0.29 Mpc
ωb: 0.02225 +0.00208 -0.00152
ωm: 0.1433 +0.0050 -0.0065
Ωm: 0.314 +0.013 -0.016
w0: -0.845 +0.069 -0.064 (prior ~U(-1.5, 0.0))
wa: -0.521 +0.444 -0.444 (prior ~U(-3.5, 2.0))
Chi squared: 1638.1 (2.21 sigma away from ΛCDM)
Log evidence: -838.6 (Δ logZ = -0.6 in favour of ΛCDM)
Degrees of freedom: 1722
"""
