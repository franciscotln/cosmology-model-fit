from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_planck_act_compression as cmb
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2(2.044)
Omnu_h2 = cmb.Omnu_h2

# arXiv:2503.14452v2 (ACT + Planck 2018)
theta_stx100 = 1.04094
theta_stx100_err = 0.00026

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, Obc, Or, w0=-1, wa=0):
    Ol = 1 - Obc - Or
    inv_a = 1 + z
    cubic = inv_a**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Or * inv_a**4 + Obc * cubic + Ol * rho_de)


@njit
def theory_mu(theta):
    dL = (1 + z_hel) * DM_z(z_cmb, theta)
    return theta[0] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, theta):
    H0, Obh2, Och2, w0 = theta[1:]
    h = H0 / 100
    Obc = (Obh2 + Och2 + Omnu_h2) / h**2
    Or = Orh2 / h**2
    return H0 * Ez(z, Obc, Or, w0)


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


@njit
def bao_theory(z, qty, theta):
    Obh2, Och2 = theta[2], theta[3]
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnu_h2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    lA = cmb.cmb_distances(Ez, *theta[1:])[1]
    thetastar = 100 * np.pi / lA
    chi2_theta_100 = ((theta_stx100 - thetastar) / theta_stx100_err) ** 2

    delta_bbn = bbn.Obh2 - theta[2]
    chi2_bbn = (delta_bbn / bbn.Obh2_sigma) ** 2

    delta_sn = mu_values - theory_mu(theta)
    chi2_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi2_bao = solve_triang(cho_bao, delta_bao)

    return chi2_sn + chi2_bao + chi2_theta_100 + chi2_bbn


bounds = np.array(
    [
        (-0.4, 0.4),  # ΔM nuisance magnitude offset
        (50.0, 90.0),  # H0
        (0.010, 0.030),  # ωb = Ωb * h^2
        (0.05, 0.30),  # ωc = Ωc * h^2
        (-1.5, 0.0),  # w0
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
    from gelman_rubin import gelman_rubin
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
    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    one_sigma_contour = [15.9, 50, 84.1]

    [
        (dM_16, dM_50, dM_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, one_sigma_contour, axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    degs_of_freedom = 2 + len(bao_data["z"]) + sn_size - len(best_fit)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnu_h2
    Om_samples = Omh2_samples / (samples[:, 1] ** 2 / 100) ** 2
    rd_samples = cmb.r_drag(wb=samples[:, 2], wm=Omh2_samples)
    z_st_samples = cmb.z_star(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_contour)
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, one_sigma_contour)
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, one_sigma_contour)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, one_sigma_contour)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, *best_fit[1:]):.2f} Mpc")
    print(f"100 θ*: {100 * np.pi / cmb.cmb_distances(Ez, *best_fit[1:])[1]:.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degrees of freedom: {degs_of_freedom}")

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
        labels=["$Δ_M$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM  w(z) = -1
H0: 68.11 +0.45 -0.45 km/s/Mpc
ωb: 0.02199 +0.00052 -0.00053
ωc: 0.1168 +0.0008 -0.0008
ωm: 0.1394 +0.0011 -0.0011
Ωm: 0.000 +0.000 -0.000
w0: -1
wa: 0
r_d: 148.38 +0.69 -0.68 Mpc
z*: 1090.15 +0.73 -0.69
r*: 145.57 Mpc
100 θ*: 1.04096
Chi squared: 1661.23
Log Evidence: -847.74
Degrees of freedom: 1746

===============================

Flat wCDM w(z) = w0
H0: 66.60 +0.63 -0.63 km/s/Mpc
ωb: 0.02231 +0.00053 -0.00053
ωc: 0.1138 +0.0012 -0.0012
ωm: 0.1368 +0.0014 -0.0014
Ωm: 0.000 +0.000 -0.000
w0: -0.912 +0.026 -0.026 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 148.84 +0.70 -0.70 Mpc
z*: 1089.45 +0.73 -0.70
r*: 146.12 Mpc
100 θ*: 1.04092
Chi squared: 1650.21
Log Evidence: -845.44 (Δ logZ 2.3 against ΛCDM)
Degrees of freedom: 1745

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)
H0: 66.42 +0.62 -0.61 km/s/Mpc
ωb: 0.02227 +0.00053 -0.00053
ωc: 0.1151 +0.0009 -0.0009
ωm: 0.1380 +0.0011 -0.0011
Ωm: 0.000 +0.000 -0.000
w0: -0.840 +0.042 -0.042 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
r_d: 148.52 +0.68 -0.68 Mpc
z*: 1089.62 +0.71 -0.69
r*: 145.80 Mpc
100 θ*: 1.04094
Chi squared: 1646.95
Log Evidence: -843.32 (Δ logZ 4.42 against ΛCDM)
Degrees of freedom: 1745

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 66.59 +0.61 -0.61 km/s/Mpc
ωb: 0.02215 +0.00053 -0.00053
ωc: 0.1176 +0.0017 -0.0018
ωm: 0.1404 +0.0017 -0.0018
Ωm: 0.000 +0.000 -0.000
w0: -0.791 +0.062 -0.060 (prior width 1.5: -1.5 to 0.0)
wa: -0.616 +0.275 -0.290 (prior width 3.5: -2.5 to 1.0)
r_d: 148.01 +0.74 -0.74 Mpc
z*: 1089.99 +0.77 -0.73
r*: 145.24 Mpc
100 θ*: 1.04091
Chi squared: 1645.62
Log Evidence: -844.55 (Δ logZ 3.19 against ΛCDM)
Degrees of freedom: 1744
"""
