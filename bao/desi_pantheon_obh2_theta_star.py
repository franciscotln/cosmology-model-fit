from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_desi_compression as cmb
from y2022pantheonSHOES.data import get_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, mag_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

"""
Planck compressed priors for θ* and ωb, without ωm = Ωm * h^2 (arXiv:2503.14738v2)
This way we allow for the ratio ωb / ωm to vary freely independently from Planck.
"""
cmb_compressed_priors = cmb.DISTANCE_PRIORS[:2]  # θ*, ωb
cho_cmb = cho_factor(cmb.covariance[:2, :2], lower=True)[0]

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
def theory_apparent_mag(theta):
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


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

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
    cmb_observables = cmb.cmb_distances(Ez, theta, theta[0], theta[1], theta[2])[:2]
    delta_cmb = cmb_compressed_priors - cmb_observables
    chi2_cmb = solve_triang(cho_cmb, delta_cmb)

    delta_sn = mag_values - theory_apparent_mag(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao + chi2_cmb


bounds = np.array(
    [
        (50.0, 90.0),  # H0
        (0.1, 0.7),  # Ωm
        (0.020, 0.024),  # ωb = Ωb * h^2
        (-1.5, 0.0),  # w0
        (-20.0, -19.0),  # M
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

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [w0_16, w0_50, w0_84],
        [M_16, M_50, M_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    zd_samples = cmb.z_drag(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    zd_16, zd_50, zd_84 = np.percentile(zd_samples, [15.9, 50, 84.1])

    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z_d: {zd_50:.2f} +{(zd_84 - zd_50):.2f} -{(zd_50 - zd_16):.2f}")
    print(f"r_d: {cmb.rs_z(Ez, zd_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evidence(samples, log_probs, log_probability, bounds):.2f}")
    print(f"Degs of freedom: {2 + len(bao_data['z']) + len(z_cmb) - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_apparent_mag(best_fit),
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$H_0$", "$Ω_m$", "$ω_b$", "$w_0$", "$M$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM  w(z) = -1
M: -19.410 +0.009 -0.009 mag
H0: 68.49 +0.29 -0.29 km/s/Mpc
Ωm: 0.297 +0.004 -0.004
ωm: 0.1393 +0.0008 -0.0008
ωb: 0.02222 +0.00014 -0.00014
z_d: 1059.30 +0.34 -0.34
r_d: 148.06 Mpc
Chi squared: 1417.27
Log Evidence: -725.95
Degs of freedom: 1601

===============================

Flat wCDM w(z) = w0
M: -19.437 +0.015 -0.015 mag
H0: 67.26 +0.61 -0.60 km/s/Mpc
Ωm: 0.303 +0.005 -0.005
ωm: 0.1373 +0.0012 -0.0013
ωb: 0.02223 +0.00014 -0.00014
w0: -0.939 +0.027 -0.027 (prior width 1.5: -1.5 to 0.0)
z_d: 1059.18 +0.35 -0.35
r_d: 148.59 Mpc
Chi squared: 1412.34
Log Evidence: -726.58
Degs of freedom: 1600

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
M: -19.434 +0.013 -0.013 mag
H0: 67.23 +0.59 -0.58 km/s/Mpc
Ωm: 0.306 +0.005 -0.005
ωm: 0.1382 +0.0010 -0.0009
ωb: 0.02223 +0.00014 -0.00014
w0: -0.906 +0.038 -0.038 (prior width 1.5: -1.5 to 0.0)
z_d: 1059.23 +0.35 -0.35
r_d: 148.36 Mpc
Chi squared: 1411.51
Log Evidence: -725.81
Degrees of freedom: 1600

===============================

Flat w(z) = w0 + wa * z / (1 + z)
M: -19.429 +0.016 -0.016 mag
H0: 67.32 +0.59 -0.60 km/s/Mpc
Ωm: 0.307 +0.006 -0.006
ωm: 0.1392 +0.0018 -0.0020
ωb: 0.02223 +0.00014 -0.00014
w0: -0.884 +0.058 -0.056 (prior width 1.5: -1.5 to 0.0)
wa: -0.283 +0.250 -0.263 (prior width 3.5: -2.0 to 1.5)
z_d: 1059.30 +0.36 -0.37
r_d: 148.11 Mpc
Chi squared: 1411.79
Log Evidence: -728.04
Degs of freedom: 1599
"""
