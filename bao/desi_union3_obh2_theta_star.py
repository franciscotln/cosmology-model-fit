from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_chen_compression as cmb
from y2023union3.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2()

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

"""
Planck compressed priors for π/θ* and ωb, without the shift parameter R (arXiv:1808.05724v1)
The geometry is constrained by θ* and the sound horizon by ωb.
"""
cho_cmb = cho_factor(cmb.covariance_wcdm[1:, 1:], lower=True)[0]

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200, dtype=np.float64)
dx = np.diff(z_grid)


@njit
def Ez(z, theta):
    h, Om, w0 = theta[0] / 100, theta[1], theta[3]
    z_plus_1 = 1 + z
    Or = Orh2 / h**2
    cubed = z_plus_1**3
    rho_de = (4 * cubed / (1 + 3 * cubed)) ** (4 * (1 + w0))
    return np.sqrt(Or * z_plus_1**4 + Om * cubed + (1 - Om) * rho_de)


@njit
def theory_mu(theta):
    dL = (1 + z_sn_vals) * DM_z(z_sn_vals, theta)
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
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
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


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    H0, Om, Obh2 = theta[0], theta[1], theta[2]

    delta_cmb = cmb.DISTANCE_PRIORS_WCDM - cmb.cmb_distances(Ez, theta, H0, Om, Obh2)
    chi_cmb = solve_triang(cho_cmb, delta_cmb[1:])

    delta_sn = mu_vals - theory_mu(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao + chi_cmb


bounds = np.array(
    [
        (50.0, 90.0),  # H0
        (0.1, 0.7),  # Ωm
        (0.020, 0.024),  # ωb = Ωb * h^2
        (-1.5, 0.0),  # w0
        (-0.8, 0.8),  # ΔM nuisance magnitude offset
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
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [w0_16, w0_50, w0_84],
        [dM_16, dM_50, dM_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    degrees_of_freedom = 2 + len(bao_data["value"]) + len(z_sn_vals) - len(best_fit)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    zd_samples = cmb.z_drag(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    zd_16, zd_50, zd_84 = np.percentile(zd_samples, [15.9, 50, 84.1])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z_d: {zd_50:.2f} +{(zd_84 - zd_50):.2f} -{(zd_50 - zd_16):.2f}")
    print(f"r_d: {cmb.rs_z(Ez, zd_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {degrees_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
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
ΔM: -0.121 +0.085 -0.086 mag
H0: 68.66 +0.30 -0.30 km/s/Mpc
Ωm: 0.298 +0.004 -0.004
ωm: 0.1404 +0.0008 -0.0009
ωb: 0.02235 +0.00014 -0.00014
w0: -1
wa: 0
z_d: 1059.67 +0.35 -0.35
r_d: 147.65 Mpc
Chi squared: 39.51
Log Evidence: -34.16
Degs of freedom: 33

===============================

Flat wCDM w(z) = w0
ΔM: -0.164 +0.090 -0.090 mag
H0: 67.00 +0.77 -0.75 km/s/Mpc
Ωm: 0.307 +0.006 -0.006
ωm: 0.1380 +0.0014 -0.0014
ωb: 0.02240 +0.00015 -0.00015
w0: -0.920 +0.033 -0.033 (prior width 1.5: -1.5 to 0.0)
wa: 0
z_d: 1059.59 +0.36 -0.37
r_d: 148.27 Mpc
Chi squared: 34.18
Log Evidence: -34.36 (Δ logZ = -0.20 in favour of ΛCDM)
Degs of freedom: 32

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)
ΔM: -0.176 +0.090 -0.090 mag
H0: 66.31 +0.85 -0.83 km/s/Mpc
Ωm: 0.316 +0.008 -0.008
ωm: 0.1389 +0.0010 -0.0010
ωb: 0.02240 +0.00015 -0.00015
w0: -0.813 +0.063 -0.063 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
z_d: 1059.65 +0.36 -0.36
r_d: 148.02 Mpc
Chi squared: 31.09
Log Evidence: -32.19 (Δ logZ = 1.97 against ΛCDM)
Degs of freedom: 32

===============================

Flat w(z) = w0 + wa * z / (1 + z)
ΔM: -0.170 +0.088 -0.088 mag
H0: 66.18 +0.83 -0.82 km/s/Mpc
Ωm: 0.324 +0.009 -0.009
ωm: 0.1418 +0.0017 -0.0018
ωb: 0.02236 +0.00014 -0.00014
w0: -0.716 +0.096 -0.093 (prior width 1.5: -1.5 to 0.0)
w_a: -0.828 +0.348 -0.367 (prior width 4.0: -3.0 to 1.0)
z_d: 1059.78 +0.36 -0.37
r_d: 147.29 Mpc
Chi squared: 29.41
Log Evidence: -33.12 (Δ logZ = 1.04 against ΛCDM)
Degs of freedom: 31
"""
