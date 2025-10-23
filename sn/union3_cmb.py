from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2023union3.data import get_data
import cmb.data_union3_compression as cmb

c = cmb.c  # km/s
O_r_h2 = cmb.Omega_r_h2()

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_cmb = cho_factor(cmb.covariance, lower=True)[0]

z_grid = np.linspace(0, np.max(z_sn_vals) + 0.1, num=1000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    h, Om, w0 = params[0] / 100, params[1], params[3]
    Or = O_r_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


@njit
def DM_z(z, params):
    dh_grid = (c / params[0]) / Ez(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def mu_theory(params):
    dL = (1 + z_sn_vals) * DM_z(z_sn_vals, params)
    return params[-1] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]

    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = solve_triang(cho_cmb, delta_cmb)

    delta_sn = mu_vals - mu_theory(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.45),  # Ωm
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
    from .plotting import plot_predictions
    from corner_plot import plot_corner_and_chains
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))

    with Pool(5) as pool:
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
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    one_sigma_percentiles = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_percentiles, axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (w0_16, w0_50, w0_84),
        (dM_16, dM_50, dM_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)
    degrees_of_freedom = len(mu_vals) + len(cmb.DISTANCE_PRIORS) - len(best_fit)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_percentiles)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_percentiles)
    z_d_16, z_d_50, z_d_84 = np.percentile(z_drag_samples, one_sigma_percentiles)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"r_d: {cmb.rs_z(Ez, z_d_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log Evidence: {log_evidence(samples, log_probs, log_probability):.1f}")
    print(f"Degrees of freedom: {degrees_of_freedom}")

    plot_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$H_0$", "$Ω_m$", "$ω_m$", "$w_0$", "$Δ_M$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*******************************

Flat ΛCDM w(z) = -1
H0: 67.11 +0.56 -0.56 km/s/Mpc
Ωm: 0.319 +0.008 -0.008
ωm: 0.14358 +0.00120 -0.00119
ωb: 0.02234 +0.00014 -0.00014
w0: -1
wa: 0
ΔM: -0.168 +0.088 -0.088
z*: 1091.99 +0.27 -0.27
z_drag: 1059.88 +0.29 -0.29
r*: 144.00 Mpc
r_d: 146.84 Mpc
Chi squared: 26.2
Log Evidence: -26.1
Degrees of freedom: 21

===============================

Flat wCDM w(z) = w0
H0: 65.20 +1.20 -1.22 km/s/Mpc
Ωm: 0.336 +0.014 -0.013
ωm: 0.14293 +0.00126 -0.00124
ωb: 0.02239 +0.00014 -0.00014
w0: -0.925 +0.043 -0.042 (prior width 1.5: -1.5 to 0.0)
wa: 0
ΔM: -0.220 +0.095 -0.095
z*: 1091.87 +0.28 -0.28
z_drag: 1059.94 +0.29 -0.29
r*: 144.14 Mpc
r_d: 146.96 Mpc
Chi squared: 23.2
Log Evidence: -27.4
Degrees of freedom: 20

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)^3)
H0: 65.29 +1.07 -1.06 km/s/Mpc
Ωm: 0.335 +0.012 -0.012
ωm: 0.14289 +0.00126 -0.00124
ωb: 0.02240 +0.00014 -0.00014
w0: -0.873 +0.067 -0.065 (prior width 1.5: -1.5 to 0.0)
wa: d w(z) / dz at z=0 = -1.5 * (1 + w0)
ΔM: -0.214 +0.092 -0.092
z*: 1091.86 +0.28 -0.28
z_drag: 1059.95 +0.29 -0.29
r*: 144.15 Mpc
r_d: 146.97 Mpc
Chi squared: 22.5
Log Evidence: -26.6
Degrees of freedom: 20

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 66.52 +1.33 -1.38 km/s/Mpc
Ωm: 0.323 +0.015 -0.013
ωm: 0.14308 +0.00124 -0.00126
ωb: 0.02238 +0.00014 -0.00014
w0: -0.686 +0.154 -0.160 (prior width 1.5: -1.5 to 0.0)
wa: -1.127 +0.739 -0.742 (prior width 8.5: -5.5 to 3.0)
ΔM: -0.157 +0.099 -0.098
z*: 1091.89 +0.28 -0.28
z_drag: 1059.93 +0.29 -0.29
r*: 144.12 Mpc
r_d: 146.94 Mpc
Chi squared: 22.2
Log Evidence: -27.6
Degrees of freedom: 19
"""
