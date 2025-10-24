from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_desi_compression as cmb
import y2024BBN.prior_lcdm_chen as bbn
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2()

# arXiv:1807.06209v4 (Planck 2018)
theta_stx100 = 1.04110
theta_stx100_err = 0.00031

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, theta):
    h, Om, w0 = theta[0] / 100, theta[1], theta[3]
    z_plus_1 = 1 + z
    Or = Orh2 / h**2
    Ode = 1 - Om - Or
    cubed = z_plus_1**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(Or * z_plus_1**4 + Om * cubed + Ode * rho_de)


@njit
def theory_mu(theta):
    dL = (1 + z_hel) * DM_z(z_cmb, theta)
    return theta[-1] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, theta):
    return theta[0] * Ez(z, theta)


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
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


def bao_theory(z, qty, theta):
    H0, Om, Obh2 = theta[0], theta[1], theta[2]
    zd = cmb.z_drag(wb=Obh2, wm=Om * (H0 / 100) ** 2)
    rd = cmb.rs_z(Ez, zd, theta, H0, Obh2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


def theta_starx100_theory(params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    z_star = cmb.z_star(wb=Obh2, wm=Om * (H0 / 100) ** 2)
    rs_star = cmb.rs_z(Ez, z_star, params, H0, Obh2)
    DA_star = cmb.DA_z(Ez, z_star, params, H0)
    return 100 * rs_star / ((1 + z_star) * DA_star)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    delta_theta_100 = theta_stx100 - theta_starx100_theory(theta)
    chi2_theta_100 = (delta_theta_100 / theta_stx100_err) ** 2

    delta_bbn = bbn.Obh2 - theta[2]
    chi2_bbn = (delta_bbn / bbn.Obh2_sigma) ** 2

    delta_sn = mu_values - theory_mu(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao + chi2_theta_100 + chi2_bbn


bounds = np.array(
    [
        (50.0, 90.0),  # H0
        (0.1, 0.7),  # Ωm
        (0.020, 0.024),  # ωb = Ωb * h^2
        (-1.5, 0.0),  # w0
        (-0.4, 0.4),  # ΔM nuisance magnitude offset
    ],
    dtype=np.float64,
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(theta):
    return -0.5 * chi_squared(theta)


def log_probability(theta):
    lp = log_prior(theta)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def main():
    import emcee
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence

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

    with Pool(8) as pool:
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

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [w0_16, w0_50, w0_84],
        [dM_16, dM_50, dM_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    zd_samples = cmb.z_drag(wb=samples[:, 2], wm=Omh2_samples)
    z_st_samples = cmb.z_star(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    zd_16, zd_50, zd_84 = np.percentile(zd_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, [15.9, 50, 84.1])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z_d: {zd_50:.2f} +{(zd_84 - zd_50):.2f} -{(zd_50 - zd_16):.2f}")
    print(f"r_d: {cmb.rs_z(Ez, zd_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"100 θ*: {theta_starx100_theory(best_fit):.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(
        f"Log Evidence: {log_evidence(samples, log_probs, log_probability, bounds):.2f}"
    )
    print(f"Degrees of freedom: {2 + bao_data['value'].size + sn_size - len(best_fit)}")

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
Flat ΛCDM  w(z) = -1
H0: 68.39 +0.35 -0.35 km/s/Mpc
Ωm: 0.299 +0.004 -0.004
ωm: 0.1398 +0.0009 -0.0009
ωb: 0.02224 +0.00032 -0.00032
z_d: 1059.37 +0.75 -0.76
r_d: 147.92 Mpc
z*: 1088.77 +0.33 -0.32
r*: 145.28 Mpc
100 θ*: 1.04112
Chi squared: 1661.97
Log Evidence: -846.94
Degrees of freedom: 1746

===============================

Flat wCDM w(z) = w0
H0: 66.67 +0.59 -0.59 km/s/Mpc
Ωm: 0.308 +0.005 -0.005
ωm: 0.1367 +0.0013 -0.0013
ωb: 0.02236 +0.00032 -0.00032
w0: -0.909 +0.025 -0.025 (prior width 1.5: -1.5 to 0.0)
z_d: 1059.41 +0.76 -0.76
r_d: 148.64 Mpc
z*: 1088.43 +0.34 -0.33
r*: 146.02 Mpc
100 θ*: 1.04106
Chi squared: 1649.91
Log Evidence: -844.08 (Δ logZ = 2.86 against ΛCDM)
Degrees of freedom: 1745

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.54 +0.58 -0.58 km/s/Mpc
Ωm: 0.312 +0.005 -0.005
ωm: 0.1380 +0.0010 -0.0010
ωb: 0.02235 +0.00032 -0.00031
w0: -0.855 +0.037 -0.037 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at (z=0) = -1.5 * (1 + w0) = -0.2175 +/- 0.0555
z_d: 1059.48 +0.75 -0.75
r_d: 148.31 Mpc
z*: 1088.53 +0.33 -0.32
r*: 145.70 Mpc
100 θ*: 1.04111
Chi squared: 1647.13
Log Evidence: -842.31 (Δ logZ = 4.63 against ΛCDM)
Degrees of freedom: 1745

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 66.70 +0.58 -0.57 km/s/Mpc
Ωm: 0.315 +0.006 -0.006
ωm: 0.1403 +0.0017 -0.0019
ωb: 0.02231 +0.00032 -0.00032
w0: -0.793 +0.062 -0.061 (prior width 1.5: -1.5 to 0.0)
wa: -0.582 +0.273 -0.288 (prior width 3.5: -2.5 to 1.0)
z_d: 1059.55 +0.75 -0.75
r_d: 147.74 Mpc
z*: 1088.73 +0.35 -0.35
r*: 145.14 Mpc
100 θ*: 1.04098
Chi squared: 1645.85
Log Evidence: -843.38 (Δ logZ = 3.56 against ΛCDM)
Degrees of freedom: 1744

Flat w(z) = w0 + wa * ((1 + z)^2 - 1) / ((1 + z)^2 + 1) (reduces to w0waCDM at low z)
H0: 66.72 +0.58 -0.57 km/s/Mpc
Ωm: 0.315 +0.006 -0.006
ωm: 0.1404 +0.0017 -0.0019
ωb: 0.02231 +0.00032 -0.00032
w0: -0.802 +0.058 -0.055 (prior width 1.5: -1.5 to 0.0)
wa: -0.477 +0.219 -0.237 (prior width 3.5: -2.5 to 1.0)
z_d: 1059.55 +0.75 -0.75
r_d: 147.73 Mpc
z*: 1088.73 +0.36 -0.34
r*: 145.12 Mpc
100 θ*: 1.04099
Chi squared: 1645.88
Log Evidence: -843.76 (Δ logZ = 3.18 against ΛCDM)
Degrees of freedom: 1744
"""
