from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2024DES.data import get_data
import cmb.data_chen_compression as cmb

c = cmb.c  # km/s
Or_h2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, app_mag_vals, cov_matrix_sn = get_data(False)

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_cmb = cho_factor(cmb.covariance, lower=True)[0]

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=1000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    h, Om, w0 = params[0] / 100, params[1], params[3]
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    rho_de = (4 * one_plus_z**3 / (1 + 3 * one_plus_z**3)) ** (4 * (1 + w0))
    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode * rho_de)


@njit
def DM_z(z, theta):
    dh_grid = (c / theta[0]) / Ez(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def theory_app_mag(params):
    dL = (1 + z_hel) * DM_z(z_cmb, params)
    return params[-1] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    H0, Om, Ob_h2 = params[0:3]

    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = solve_triang(cho_cmb, delta)

    delta_sn = app_mag_vals - theory_app_mag(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (55, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-1.5, 0.0),  # w0
        (-20.0, -19.0),  # M
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
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from .plotting import plot_predictions as plot_sn_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
    ]

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, pool=pool, moves=moves
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", nwalkers * nsteps * ndim / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (w0_16, w0_50, w0_84),
        (M_16, M_50, M_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    z_st = cmb.z_star(Obh2_50, Omh2_50)
    z_dr = cmb.z_drag(Obh2_50, Omh2_50)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"z*: {z_st:.2f}")
    print(f"z_drag: {z_dr:.2f}")
    print(f"r_s(z*) = {cmb.rs_z(Ez, z_st, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"r_s(z_drag) = {cmb.rs_z(Ez, z_dr, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")

    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=app_mag_vals - M_50,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_app_mag(best_fit) - M_50,
        label=f"$Ω_m$={Om_50:.3f}",
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
Flat ΛCDM w(z) = -1
H0: 66.86 +0.53 -0.53 km/s/Mpc
Ωm: 0.324 +0.008 -0.007
ωb: 0.02227 +0.00014 -0.00014
w0: -1
M: -19.418 +0.013 -0.013
z*: 1089.09
z_drag: 1059.82
r_s(z*) = 143.92 Mpc
r_s(z_drag) = 146.50 Mpc
Chi squared: 1643.66
Log evidence: -838.1
Degrees of freedom: 1734

===============================

Flat wCDM w(z) = w0
H0: 65.72 +0.75 -0.74 km/s/Mpc
Ωm: 0.333 +0.009 -0.009
ωb: 0.02237 +0.00014 -0.00015
w0: -0.942 +0.027 -0.027 (prior width 1.5: -1.5 to 0.0)
M: -19.435 +0.015 -0.015
z*: 1088.90
z_drag: 1059.94
r_s(z*) = 144.17 Mpc
r_s(z_drag) = 146.73 Mpc
Chi squared: 1639.33
Log evidence: -839.0
Degrees of freedom: 1733

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)
H0: 65.91 +0.67 -0.66 km/s/Mpc
Ωm: 0.331 +0.008 -0.008
ωb: 0.02237 +0.00014 -0.00014
w0: -0.895 +0.045 -0.045
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
M: -19.424 +0.014 -0.013
z*: 1088.90
z_drag: 1059.94
r_s(z*) = 144.19 Mpc
r_s(z_drag) = 146.75 Mpc
Chi squared: 1638.47
Log evidence: -838.1
Degrees of freedom: 1733

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.06 +1.04 -1.12 km/s/Mpc
Ωm: 0.320 +0.012 -0.011
ωb: 0.02235 +0.00014 -0.00014
w0: -0.770 +0.110 -0.115 (prior width 1.5: -1.5 to 0.0)
wa: -0.865 +0.572 -0.570 (prior width 6.5: -4.0 to 2.5)
M: -19.378 +0.034 -0.039
z*: 1088.94
z_drag: 1059.93
r_s(z*) = 144.12 Mpc
r_s(z_drag) = 146.69 Mpc
Chi squared: 1637.60
Log evidence: -838.2
Degrees of freedom: 1732
"""
