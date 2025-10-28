from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2022pantheonSHOES.data import get_data
import cmb.data_chen_compression as cmb

c = cmb.c  # Speed of light in km/s
O_r_h2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, mb_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=1000)
dx = np.diff(z_grid)

one_plus_z_hel = 1 + z_hel


@njit
def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = O_r_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    fz_de = (2 * cubed**2 / (1 + cubed**2)) ** (1 + w0)

    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * fz_de)


@njit
def DM_z(theta):
    dh_grid = (c / theta[0]) / Ez(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z_cmb, z_grid, cum_dm)


@njit
def sn_apparent_mag(params):
    dL = one_plus_z_hel * DM_z(params)
    return params[-1] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]

    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = np.dot(delta, np.dot(cmb.inv_cov_mat, delta))

    delta_sn = mb_values - sn_apparent_mag(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.15, 0.40),  # Ωm
        (0.020, 0.025),  # Ωb * h^2
        (-2.0, 0.0),  # w0
        (-20, -19),  # M
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
    from corner_plot import plot_corner_and_chains
    from .plotting import plot_predictions as plot_sn_predictions
    from gelman_rubin import gelman_rubin

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 1500 + burn_in
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
        print("effective samples", nwalkers * ndim * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains = sampler.get_chain(discard=burn_in, flat=False)

    print("Gelman-Rubin statistic:", gelman_rubin(chains))

    one_sigma_conf_int = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_conf_int, axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (w0_16, w0_50, w0_84),
        (M_16, M_50, M_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    r_drag_samples = cmb.r_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_conf_int)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_conf_int)
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(z_drag_samples, one_sigma_conf_int)
    r_dr_16, r_dr_50, r_dr_84 = np.percentile(r_drag_samples, one_sigma_conf_int)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(
        f"z_drag: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}"
    )
    print(f"r_s(z*) = {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(
        f"r_s(z_drag) = {r_dr_50:.2f} +{(r_dr_84 - r_dr_50):.2f} -{(r_dr_50 - r_dr_16):.2f} Mpc"
    )
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mb_values - M_50,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=sn_apparent_mag(best_fit) - M_50,
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$H_0$", "$Ω_m$", "$ω_b$", "$w_0$", "M"],
        flat_samples=samples,
        samples=chains,
    )


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
H0: 67.21 +0.55 -0.53 km/s/Mpc
Ωm: 0.319 +0.008 -0.007
ωm: 0.14429 +0.00114 -0.00113
ωb: 0.02233 +0.00014 -0.00014
w0: -1
M: -19.444 +0.015 -0.015
z*: 1088.98 +0.20 -0.20
z_drag: 1059.89 +0.28 -0.28
r_s(z*) = 144.07 Mpc
r_s(z_drag) = 146.64 +0.26 -0.27 Mpc
Chi squared: 1403.48
Degrees of freedom: 1587

===============================

Flat wCDM w(z) = w0
H0: 66.67 +0.81 -0.81 km/s/Mpc
Ωm: 0.324 +0.009 -0.009
ωm: 0.14385 +0.00126 -0.00124
ωb: 0.02236 +0.00014 -0.00014
w0: -0.975 +0.029 -0.030
M: -19.456 +0.021 -0.021
z*: 1088.91 +0.22 -0.21
z_drag: 1059.94 +0.29 -0.29
r_s(z*) = 144.17 Mpc
r_s(z_drag) = 146.72 +0.28 -0.28 Mpc
Chi squared: 1402.76
Degrees of freedom: 1586

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.82 +0.68 -0.66 km/s/Mpc
Ωm: 0.322 +0.008 -0.008
ωm: 0.14383 +0.00125 -0.00123
ωb: 0.02236 +0.00014 -0.00014
w0: -0.946 +0.056 -0.056
wa: d w(z)/dz at z=0 = -3 * (1 + w0)
M: -19.449 +0.016 -0.016
z*: 1088.91 +0.22 -0.21
z_drag: 1059.94 +0.29 -0.29
r_s(z*) = 144.17 Mpc
r_s(z_drag) = 146.73 +0.29 -0.28 Mpc
Chi squared: 1402.63

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.24 +1.27 -1.40 km/s/Mpc
Ωm: 0.318 +0.014 -0.012
ωm: 0.14390 +0.00124 -0.00125
ωb: 0.02236 +0.00014 -0.00015
w0: -0.919 +0.105 -0.111
wa: -0.290 +0.543 -0.544
M: -19.434 +0.042 -0.049
z*: 1088.92 +0.22 -0.21
z_drag: 1059.93 +0.29 -0.29
r_s(z*) = 144.14 Mpc
r_s(z_drag) = 146.72 +0.29 -0.28 Mpc
Chi squared: 1402.68
Degrees of freedom: 1585
"""
