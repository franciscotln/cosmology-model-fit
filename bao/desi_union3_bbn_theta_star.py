from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_planck_act_compression as cmb
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2(2.044)
Omnuh2 = cmb.Omnu_h2
z_nr = cmb.z_nr

# arXiv:2503.14452v2 (ACT + Planck 2018)
theta_stx100 = 1.04094
theta_stx100_err = 0.00026

sn_legend, z_sn_vals, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Omnu_z(z):
    """
    Computes the appox. evolution of one massive
    neutrino species energy density with redshift
    """
    return (
        (1 + z) ** 4
        * (1 + ((1 + z_nr) / (1 + z)) ** 2) ** 0.5
        * (1 + (1 + z_nr) ** 2) ** -0.5
    )


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * Omnu_z(z)
    dark_energy_term = Ode * (4 * zp1**3 / (1 + 3 * zp1**3)) ** (4 * (1 + w0))

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def theory_mu(theta):
    dL = (1 + z_sn_vals) * DM_z(z_sn_vals, theta)
    return theta[0] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, theta):
    H0, Obh2, Och2, w0 = theta[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


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
    Obh2, Och2 = theta[2], theta[3]
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnuh2)

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
    thetastar100 = 100 * np.pi / lA
    chi2_theta_100 = ((theta_stx100 - thetastar100) / theta_stx100_err) ** 2

    delta_bbn = bbn.Obh2 - theta[2]
    chi2_bbn = (delta_bbn / bbn.Obh2_sigma) ** 2

    delta_sn = mu_values - theory_mu(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao + chi2_theta_100 + chi2_bbn


bounds = np.array(
    [
        (-1.0, 1.0),  # ΔM nuisance magnitude offset
        (50.0, 90.0),  # H0
        (0.010, 0.030),  # ωb = Ωb * h^2
        (0.05, 0.3),  # ωc = Ωc * h^2
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
    burn_in = 400
    nsteps = 4000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.70),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

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
    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    one_sigma_ci = [15.9, 50, 84.1]
    [
        (dM_16, dM_50, dM_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, one_sigma_ci, axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    degrees_of_freedom = 2 + len(bao_data["z"]) + len(z_sn_vals) - len(best_fit)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    zd_samples = cmb.z_drag(wb=samples[:, 2], wm=Omh2_samples)
    z_st_samples = cmb.z_star(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_ci)
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, one_sigma_ci)
    zd_16, zd_50, zd_84 = np.percentile(zd_samples, one_sigma_ci)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, one_sigma_ci)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z_d: {zd_50:.2f} +{(zd_84 - zd_50):.2f} -{(zd_50 - zd_16):.2f}")
    print(f"r_d: {cmb.rs_z(Ez, zd_50, *best_fit[1:]):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, *best_fit[1:]):.2f} Mpc")
    print(f"100 θ*: {100 * np.pi / cmb.cmb_distances(Ez, *best_fit[1:])[1]:.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degrees of freedom: {degrees_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
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
H0: 68.32 +0.47 -0.47 km/s/Mpc
ωb: 0.02209 +0.00054 -0.00053
ωc: 0.1164 +0.0008 -0.0008
ωm: 0.1392 +0.0011 -0.0011
Ωm: 0.298 +0.005 -0.004
w0: -1
wa: 0
z_d: 1059.02 +1.27 -1.28
r_d: 148.37 Mpc
z*: 1089.98 +0.74 -0.71
r*: 145.58 Mpc
100 θ*: 1.04095
Chi squared: 39.43
Log Evidence: -34.66
Degrees of freedom: 33

===============================

Flat wCDM w(z) = w0
H0: 66.78 +0.81 -0.80 km/s/Mpc
ωb: 0.02229 +0.00054 -0.00054
ωc: 0.1140 +0.0014 -0.0014
ωm: 0.1370 +0.0015 -0.0015
Ωm: 0.307 +0.006 -0.006
w0: -0.921 +0.034 -0.035 (prior width 1.5: -1.5 to 0.0)
wa: 0
z_d: 1059.32 +1.26 -1.28
r_d: 148.79 Mpc
z*: 1089.50 +0.75 -0.73
r*: 146.06 Mpc
100 θ*: 1.04092
Chi squared: 34.16
Log Evidence: -34.95 (Δ logZ = -0.29 in favour of ΛCDM)
Degrees of freedom: 32

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)
H0: 66.10 +0.89 -0.88 km/s/Mpc
ωb: 0.02229 +0.00054 -0.00054
ωc: 0.1149 +0.0010 -0.0010
ωm: 0.1379 +0.0012 -0.0012
Ωm: 0.316 +0.008 -0.008
w0: -0.815 +0.065 -0.065 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
z_d: 1059.38 +1.26 -1.28
r_d: 148.55 Mpc
z*: 1089.58 +0.74 -0.71
r*: 145.83 Mpc
100 θ*: 1.04093
Chi squared: 31.10
Log Evidence: -32.81 (Δ logZ = 1.85 against ΛCDM)
Degrees of freedom: 32

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 65.90 +0.88 -0.88 km/s/Mpc
ωb: 0.02214 +0.00054 -0.00054
ωc: 0.1180 +0.0017 -0.0019
ωm: 0.1407 +0.0018 -0.0019
Ωm: 0.324 +0.010 -0.010
w0: -0.716 +0.100 -0.097 (prior width 1.5: -1.5 to 0.0)
wa: -0.834 +0.371 -0.393 (prior width 4.0: -3.0 to 1.0)
z_d: 1059.23 +1.26 -1.28
r_d: 147.90 Mpc
z*: 1090.05 +0.79 -0.75
r*: 145.14 Mpc
100 θ*: 1.04090
Chi squared: 29.08
Log Evidence: -33.68 (Δ logZ = 0.98 against ΛCDM)
Degrees of freedom: 31
"""
