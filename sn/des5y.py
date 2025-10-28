from numba import njit
import numpy as np
from scipy.constants import c as c0
import scipy.stats as stats
from scipy.linalg import cho_factor, solve_triangular
from y2024DES.data import get_data, effective_sample_size

legend, z_cmb_vals, z_hel_vals, app_mag_vals, covmat = get_data(False)
cho = cho_factor(covmat, lower=True)[0]

c = c0 / 1000  # Speed of light (km/s)
H0 = 70  # Hubble constant (km/s/Mpc)

z_grid = np.linspace(0, np.max(z_cmb_vals) + 0.1, num=1200)
dx = np.diff(z_grid)

inv_a = 1 + z_grid


@njit
def Ez(params):
    Om, w0 = params[1], params[2]
    Ode = 1 - Om
    Rho_de = (2 * inv_a**6 / (1 + inv_a**6)) ** (1 + w0)
    return np.sqrt(Om * inv_a**3 + Ode * Rho_de)


@njit
def DM_z(z, params):
    dh_grid = (c / H0) / Ez(params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def theory_app_mag(params):
    dL = (1 + z_hel_vals) * DM_z(z_cmb_vals, params)
    return params[0] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta = app_mag_vals - theory_app_mag(params)
    return solve_triang(cho, delta)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([(-20, -19), (0, 0.8), (-2.0, 0.0)], dtype=np.float64)  # M, Ωm, w0

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from .plotting import plot_predictions, print_color, plot_residuals

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = burn_in + 2000
    initial_state = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.3),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_state, nsteps, progress=True)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Acceptance fraction", np.mean(sampler.acceptance_fraction))
        print_color(
            "effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau)
        )
    except Exception as e:
        print("Autocorrelation time could not be computed")

    [
        (M_16, M_50, M_84),
        (Om_16, Om_50, Om_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    theory_app_mag_vals = theory_app_mag(best_fit)
    residuals = app_mag_vals - theory_app_mag_vals

    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((app_mag_vals - np.mean(app_mag_vals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    M_label = f"{M_50:.3f} +{M_84-M_50:.3f} -{M_50-M_16:.3f} mag"
    Om_label = f"{Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}"
    w0_label = f"{w0_50:.2f} +{w0_84-w0_50:.2f} -{w0_50-w0_16:.2f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_cmb_vals[0]:.3f} - {z_cmb_vals[-1]:.3f}")
    print_color("Sample size", len(z_cmb_vals))
    print_color("M", M_label)
    print_color("Ωm", Om_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{stats.skew(residuals):.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit):.2f}")
    print_color("Effective deg of freedom", effective_sample_size - ndim)

    y_err = np.sqrt(covmat.diagonal())

    plot_corner_and_chains(
        labels=["$M$", "$Ω_m$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )
    plot_predictions(
        legend=legend,
        x=z_cmb_vals,
        y=app_mag_vals - M_50,
        y_err=y_err,
        y_model=theory_app_mag_vals - M_50,
        label=f"$Ω_m$={Om_label}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb_vals, residuals=residuals, y_err=y_err, bins=40)


if __name__ == "__main__":
    main()

"""
********************************
Dataset: DES-SN5YR
z range: 0.025 - 1.121
Sample size: 1829
********************************

Flat ΛCDM w(z) = -1
M: -19.302 +0.011 -0.011 mag
Ωm: 0.352 +0.017 -0.017
w0: -1
wa: 0
R-squared (%): 98.41
RMSD (mag): 0.263
Skewness of residuals: 3.407
Chi squared: 1640.07
Effective deg of freedom: 1733

==============================

Flat wCDM w(z) = w0
M: -19.292 +0.013 -0.013 mag
Ωm: 0.266 +0.071 -0.092
w0: -0.80 +0.14 -0.15
wa: 0
R-squared (%): 98.40
RMSD (mag): 0.264
Skewness of residuals: 3.415
Chi squared: 1638.52
Effective deg of freedom: 1732

==============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)^6)
M: -19.287 +0.014 -0.014 mag
Ωm: 0.308 +0.032 -0.033
w0: -0.77 +0.12 -0.13
wa: d w(z)/dz at z=0 = -3.0 * (1 + w0)
R-squared (%): 98.40
RMSD (mag): 0.264
Skewness of residuals: 3.421
Chi squared: 1637.28
Effective deg of freedom: 1732

==============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
M: -19.262 +0.018 -0.018 mag
Ωm: 0.497 +0.033 -0.043
w0: -0.35 +0.39 -0.31 (prior width 4.5: -2.5 to 2.0)
wa: -9.08 +3.90 -4.81 (prior width 40: -30 to 10)
R-squared (%): 98.40
RMSD (mag): 0.264
Skewness of residuals: 3.454
Chi squared: 1631.98
Effective deg of freedom: 1731

Flat w(z) = w0 + wa * ((1 + z)^2 - 1) / ((1 + z)^2 + 1) (reduces to w0waCDM at low z)
ρ_de = ρ_de_0 * (1 + z)^(3 * (1 + w0)) * {2 * (1 + z) / [1 + (1 + z)^2]}^(-3 * wa)
M: -19.263 +0.018 -0.018 mag
Ωm: 0.498 +0.033 -0.040
w0: -0.39 +0.36 -0.29 (prior width 4.5: -2.5 to 2.0)
wa: -8.12 +3.39 -4.31 (prior width 40: -30 to 10)
R-squared (%): 98.40
RMSD (mag): 0.264
Skewness of residuals: 3.453
Chi squared: 1631.95
Effective deg of freedom: 1731
"""
