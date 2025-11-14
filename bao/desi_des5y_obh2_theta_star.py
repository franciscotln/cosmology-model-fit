from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_planck_act_compression as cmb
from y2025DESdovekie.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2(2.044)
Omnu_h2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

"""
Planck + ACT compressed priors for π/θ* and ωb, without the shift parameter R (arXiv:1808.05724v1)
"""
cho_cmb = cho_factor(cmb.covariance[1:, 1:], lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, Obc, Or, w0=-1, wa=0):
    Ol = 1 - Obc - Or
    zp1 = 1 + z
    cubic = zp1**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Or * zp1**4 + Obc * cubic + Ol * rho_de)


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


@njit
def theory_mu(theta):
    dL = (1 + z_hel) * DM_z(z_cmb, theta)
    return theta[0] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, *theta[1:])
    chi2_cmb = solve_triang(cho_cmb, delta_cmb[1:])

    delta_sn = mu_values - theory_mu(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi_sn + chi_bao + chi2_cmb


bounds = np.array(
    [
        (-0.4, 0.4),  # ΔM nuisance magnitude offset
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
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions
    from log_evidence import log_evidence
    from gelman_rubin import gelman_rubin

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

    [
        (dM_16, dM_50, dM_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnu_h2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    zd_samples = cmb.z_drag(wb=samples[:, 2], wm=Omh2_samples)
    z_st_samples = cmb.z_star(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])
    zd_16, zd_50, zd_84 = np.percentile(zd_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, [15.9, 50, 84.1])
    R, lA = cmb.cmb_distances(Ez, *best_fit[1:])[:2]

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z_d: {zd_50:.2f} +{(zd_84 - zd_50):.2f} -{(zd_50 - zd_16):.2f}")
    print(f"r_d: {cmb.rs_z(Ez, zd_50, *best_fit[1:]):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, *best_fit[1:]):.2f} Mpc")
    print(f"shift R: {R:.3f}")
    print(f"100 θ*: {100 * np.pi / lA:.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
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
        labels=["$Δ_M$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()


"""
*******************************
DESI DR2 + DES5Y + (π/θ*, ωb)CMB from Planck + ACT
*******************************
"""

"""
Flat ΛCDM  w(z) = -1

H0: 68.53 +0.28 -0.28 km/s/Mpc
Ωm: 0.298 +0.004 -0.004
ωb: 0.02249 +0.00011 -0.00011
ωc: 0.1168 +0.0008 -0.0008
ωm: 0.1399 +0.0008 -0.0008
w0: -1
wa: 0
z_d: 1059.98 +0.26 -0.26
r_d: 147.82 Mpc
z*: 1089.47 +0.15 -0.15
r*: 145.19 Mpc
shift R: 1.740
100 θ*: 1.04096
Chi squared: 1646.85
Log Evidence: -842.00
Degrees of freedom: 1725
"""


"""
Flat wCDM w(z) = w0

H0: 67.30 +0.56 -0.56 km/s/Mpc
Ωm: 0.304 +0.005 -0.005
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1145 +0.0012 -0.0012
ωm: 0.1377 +0.0012 -0.0012
w0: -0.937 +0.025 -0.025 (prior width 1.5: -1.5 to 0.0)
wa: 0
z_d: 1059.83 +0.27 -0.27
r_d: 148.43 Mpc
z*: 1089.26 +0.17 -0.18
r*: 145.78 Mpc
shift R: 1.733
100 θ*: 1.04093
Chi squared: 1640.50
Log Evidence: -842.05 (Δ logZ = -0.05 in favour of ΛCDM)
Degrees of freedom: 1724
"""


"""
Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)

H0: 67.18 +0.54 -0.54 km/s/Mpc
Ωm: 0.307 +0.005 -0.005
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1155 +0.0009 -0.0009
ωm: 0.1386 +0.0009 -0.0009
w0: -0.885 +0.040 -0.040 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
z_d: 1059.90 +0.26 -0.26
r_d: 148.18 Mpc
z*: 1089.35 +0.16 -0.16
r*: 145.54 Mpc
shift R: 1.736
100 θ*: 1.04094
Chi squared: 1638.87
Log Evidence: -840.75 (Δ logZ = 1.25 against ΛCDM)
"""


"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)

H0: 67.34 +0.54 -0.54 km/s/Mpc
Ωm: 0.310 +0.006 -0.006
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1173 +0.0017 -0.0019
ωm: 0.1404 +0.0017 -0.0019
w0: -0.850 +0.059 -0.058 (prior width 1.5: -1.5 to 0.0)
wa: -0.436 +0.266 -0.277 (prior width 3.5: -2.5 to 1.0)
z_d: 1060.02 +0.28 -0.28
r_d: 147.69 Mpc
z*: 1089.50 +0.20 -0.21
r*: 145.06 Mpc
shift R: 1.742
100 θ*: 1.04090
Chi squared: 1638.27
Log Evidence: -842.36 (Δ logZ = -0.36 in favour of ΛCDM)
Degrees of freedom: 1723
"""
