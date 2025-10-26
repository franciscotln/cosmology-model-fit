from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2024DES.data import get_data
import cmb.data_chen_compression as cmb

c = cmb.c  # km/s
Or_h2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()

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
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode * rho_de)


@njit
def DM_z(z, theta):
    dh_grid = (c / theta[0]) / Ez(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def theory_mu(params):
    dL = (1 + z_hel) * DM_z(z_cmb, params)
    return params[-1] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    H0, Om, Ob_h2 = params[0:3]

    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = solve_triang(cho_cmb, delta)

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (55, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-1.5, 0.0),  # w0
        (-0.7, 0.7),  # ΔM
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
        (dM_16, dM_50, dM_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    z_st = cmb.z_star(Obh2_50, Omh2_50)
    z_dr = cmb.z_drag(Obh2_50, Omh2_50)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"z*: {z_st:.2f}")
    print(f"z_drag: {z_dr:.2f}")
    print(f"r_s(z*) = {cmb.rs_z(Ez, z_st, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"r_s(z_drag) = {cmb.rs_z(Ez, z_dr, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")

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
        labels=["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "$Δ_M$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
H0: 66.85 +0.54 -0.53 km/s/Mpc
Ωm: 0.324 +0.008 -0.008
Ωb h^2: 0.02227 +0.00014 -0.00014
w0: -1
ΔM: -0.095 +0.013 -0.013
z*: 1089.09
z_drag: 1059.81
r_s(z*) = 143.93 Mpc
r_s(z_drag) = 146.52 Mpc
Chi squared: 1643.67
Log evidence: -838.5
Degrees of freedom: 1734

===============================

Flat wCDM w(z) = w0
H0: 65.72 +0.75 -0.75 km/s/Mpc
Ωm: 0.333 +0.009 -0.009
ωb: 0.02236 +0.00015 -0.00015
w0: -0.942 +0.027 -0.028
ΔM: -0.112 +0.015 -0.016
z*: 1088.91
z_drag: 1059.94
r_s(z*) = 144.17 Mpc
r_s(z_drag) = 146.73 Mpc
Chi squared: 1639.35 (Δ chi2 4.3 from ΛCDM)
Log evidence: -839.4
Degrees of freedom: 1733

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 65.90 +0.67 -0.66 km/s/Mpc
Ωm: 0.331 +0.008 -0.008
ωb: 0.02237 +0.00015 -0.00014
w0: -0.907 +0.040 -0.041
ΔM: -0.102 +0.013 -0.013
z*: 1088.90
z_drag: 1059.94
r_s(z*) = 144.18 Mpc
r_s(z_drag) = 146.74 Mpc
Chi squared: 1638.69 (Δ chi2 5.0 from ΛCDM)
Log evidence: -838.7
Degrees of freedom: 1733

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.11 +1.00 -1.09 km/s/Mpc
Ωm: 0.320 +0.011 -0.010
ωb: 0.02235 +0.00015 -0.00015
w0: -0.765 +0.108 -0.115
wa: -0.888 +0.575 -0.555
ΔM: -0.054 +0.034 -0.038
z*: 1088.94
z_drag: 1059.92
r_s(z*) = 144.13 Mpc
r_s(z_drag) = 146.69 Mpc
Chi squared: 1637.39 (Δ chi2 6.3 from ΛCDM)
Degrees of freedom: 1732

Flat w(z) = w0 + wa * ((1 + z)^2 - 1) / ((1 + z)^2 + 1) (reduces to w0waCDM at low z)
ρ_de = ρ_de_0 * (1 + z)^(3 * (1 + w0)) * {2 * (1 + z) / [1 + (1 + z)^2]}^(-3 * wa)
H0: 67.13 +1.04 -1.13 km/s/Mpc
Ωm: 0.320 +0.012 -0.011
ωb: 0.02235 +0.00014 -0.00015
w0: -0.779 +0.104 -0.110
wa: -0.724 +0.480 -0.480
ΔM: -0.053 +0.034 -0.039
z*: 1088.94
z_drag: 1059.93
r_s(z*) = 144.11 Mpc
r_s(z_drag) = 146.68 Mpc
Chi squared: 1637.48 (Δ chi2 6.2 from ΛCDM)
Log evidence: -839.6
Degrees of freedom: 1732
"""
