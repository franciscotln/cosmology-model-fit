from numba import njit
import numpy as np
from scipy.constants import c as c0
import scipy.stats as stats
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025DESdovekie.data import get_data, effective_sample_size

legend, z_cmb, z_hel, mu_vals, covmat = get_data()

cho = cho_factor(covmat, lower=True)[0]

c = c0 / 1000  # Speed of light (km/s)
H0 = 70.0  # Hubble constant (km/s/Mpc)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=3000)
dx = np.diff(z_grid)

inv_a = 1.0 + z_grid


@njit
def Ez(params):
    Om, w0 = params[1], params[2]
    Ode = 1.0 - Om
    rho_de = (2 * inv_a**3 / ((1.0 + w0) + (1.0 - w0) * inv_a**3)) ** 2
    return np.sqrt(Om * inv_a**3 + Ode * rho_de)


@njit
def DM_z(z, params):
    dh_grid = (c / H0) / Ez(params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def theory_mu(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta = mu_vals - theory_mu(params)
    return solve_triang(cho, delta)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-0.2, 0.2),  # ΔM
        (0.0, 0.8),  # Ωm
        (-1.0, 0.0),  # w0
    ]
)

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
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from sn.plotting import plot_predictions, print_color, plot_residuals

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = burn_in + 2500
    initial_state = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.25),
        (emcee.moves.DEMove(), 0.75),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_state, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)
    print(f"Gelman-Rubin: {gelman_rubin(chains_samples)}")
    print_color("Log evidence", f"{log_evd:.1f}")

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

    theory_mu_vals = theory_mu(best_fit)
    residuals = mu_vals - theory_mu_vals

    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mu_vals - np.mean(mu_vals)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    M_label = f"{M_50:.3f} +{M_84-M_50:.3f} -{M_50-M_16:.3f} mag"
    Om_label = f"{Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}"
    w0_label = f"{w0_50:.2f} +{w0_84-w0_50:.2f} -{w0_50-w0_16:.2f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_cmb[0]:.3f} - {z_cmb[-1]:.3f}")
    print_color("Sample size", len(z_cmb))
    print_color("ΔM", M_label)
    print_color("Ωm", Om_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{stats.skew(residuals):.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit):.2f}")
    print_color("Effective deg of freedom", effective_sample_size - ndim)

    y_err = np.sqrt(covmat.diagonal())

    labels = ["$ΔM$", "$Ω_m$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=mu_vals,
        y_err=y_err,
        y_model=theory_mu_vals,
        label=f"$Ω_m$={Om_label}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=y_err, bins=40)


if __name__ == "__main__":
    main()

"""
********************************
Dataset: DES-SN5YR Dovekie - effective: 1714 SNe
z range: 0.025 - 1.144
Sample size: 1820
********************************

Flat ΛCDM w(z) = -1
Ωm: 0.331 +0.015 -0.015
w0: -1
wa: 0
R-squared (%): 98.38
RMSD (mag): 0.268
Skewness of residuals: 3.206
Chi squared: 1631.42
Log evidence: -822.3
Effective deg of freedom: 1712

==============================

Flat wCDM w(z) = w0
Ωm: 0.260 +0.065 -0.085
w0: -0.83 +0.14 -0.15
wa: 0
R-squared (%): 98.37
RMSD (mag): 0.268
Skewness of residuals: 3.213
Chi squared: 1630.18
Effective deg of freedom: 1711

==============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
Ωm: 0.287 +0.030 -0.035
w0: -0.81 +0.11 -0.11
wa: d w(z)/dz at z=0 = -(3/2) * (1 - w0^2)
R-squared (%): 98.37
RMSD (mag): 0.268
Skewness of residuals: 3.217
Chi squared: 1629.55
Log evidence: -822.4
Effective deg of freedom: 1711

==============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
"""
