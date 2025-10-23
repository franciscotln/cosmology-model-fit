from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2024DES.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data
import cmb.data_chen_compression as cmb

c = cmb.c  # km/s
Or_h2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    h, Om, w0 = params[0] / 100, params[1], params[3]
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))

    return (Or * one_plus_z**4 + Om * one_plus_z**3 + Ode * rho_de) ** 0.5


@njit
def theory_mu(params):
    dL = (1 + z_hel) * DM_z(z_cmb, params)
    return params[-1] + 25 + 5 * np.log10(dL)


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
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


def bao_theory(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    z_drag = cmb.z_drag(wb=Obh2, wm=Omh2)
    rd = cmb.rs_z(Ez, z_drag, params, H0, Obh2)

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
    H0, Om, Obh2 = params[0], params[1], params[2]

    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Obh2)
    chi2_cmb = np.dot(delta, np.dot(cmb.inv_cov_mat, delta))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_bao + chi_sn


bounds = np.array(
    [
        (60.0, 75.0),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # ωb
        (-1.5, 0.0),  # w0
        (-0.7, 0.7),  # ΔM
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
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.30),
                (emcee.moves.DEMove(), 0.56),
                (emcee.moves.DESnookerMove(), 0.14),
            ],
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

    one_sigma_contours = [15.9, 50, 84.1]

    pct = np.percentile(samples, one_sigma_contours, axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (w0_16, w0_50, w0_84),
        (dM_16, dM_50, dM_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_star_samples = cmb.z_star(wb=samples[:, 2], wm=Omh2_samples)
    z_drag_samples = cmb.z_drag(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_contours)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_contours)
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(z_drag_samples, one_sigma_contours)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r_d: {cmb.rs_z(Ez, z_dr_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"z_d: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evidence(samples, log_probs, log_probability, bounds):.1f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$H_0$", "$Ω_m$", "$ω_b$", "$w_0$", "$Δ_M$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
ΔM: -0.059 +0.008 -0.008
H0: 68.40 +0.28 -0.28 km/s/Mpc
Ωm: 0.3032 +0.0036 -0.0036
ωb: 0.02250 +0.00012 -0.00012
ωm: 0.14186 +0.00061 -0.00061
r*: 144.60 Mpc
z*: 1088.62 +0.14 -0.14
r_d: 147.13 Mpc
z_d: 1060.11 +0.27 -0.27
Chi squared: 1664.85
Log evidence: -849.5
Degrees of freedom: 1747

===============================

Flat wCDM w(z) = w0
ΔM: -0.073 +0.010 -0.010
H0: 67.41 +0.53 -0.53 km/s/Mpc
Ωm: 0.3099 +0.0048 -0.0048
ωb: 0.02259 +0.00013 -0.00013
ωm: 0.14081 +0.00079 -0.00079
w0: -0.952 +0.022 -0.022 (prior width 1.5: -1.5 to 0.0)
r*: 144.82 Mpc
z*: 1088.45 +0.16 -0.16
r_d: 147.33 Mpc
z_d: 1060.24 +0.28 -0.28
Chi squared: 1660.35
Log evidence: -850.6 (Δ logZ = -1.1 in favour of ΛCDM)
Degrees of freedom: 1746

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
ΔM: -0.075 +0.010 -0.010
H0: 66.90 +0.55 -0.55 km/s/Mpc
Ωm: 0.3148 +0.0054 -0.0053
ωb: 0.02259 +0.00012 -0.00012
ωm: 0.14089 +0.00070 -0.00069
w0: -0.889 +0.036 -0.035 (prior width 1.5: -1.5 to 0.0)
r*: 144.80 Mpc
z*: 1088.46 +0.15 -0.15
r_d: 147.32 Mpc
z_d: 1060.23 +0.27 -0.28
Chi squared: 1655.39
Log evidence: -847.6 (Δ logZ = 1.9 against ΛCDM)
Degrees of freedom: 1746

===============================

Flat w(z) = w0 + wa * z / (1 + z)
ΔM: -0.055 +0.011 -0.011
H0: 66.98 +0.55 -0.55 km/s/Mpc
Ωm: 0.3187 +0.0056 -0.0055
ωb: 0.02242 +0.00013 -0.00013
ωm: 0.14300 +0.00091 -0.00092
w0: -0.761 +0.057 -0.056 (prior width 1.5: -1.5 to 0.0)
wa: -0.802 +0.221 -0.237 (prior width 3.5: -2.5 to 1.0)
r*: 144.35 Mpc
z*: 1088.79 +0.17 -0.17
r_d: 146.90 Mpc
z_d: 1060.01 +0.28 -0.28
Chi squared: 1646.86
Log evidence: -845.4 (Δ logZ = 4.1 against ΛCDM)
Degrees of freedom: 1745
"""
