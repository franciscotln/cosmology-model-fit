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

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = np.diff(z_grid)
zp1 = 1.0 + z_grid


@njit
def Ode_z(w0):
    # Thawing quintessence with w(z) ranging from -1 to 1
    return (2 * zp1**3 / ((1.0 + w0) + (1.0 - w0) * zp1**3)) ** 2


@njit
def Hz(params):
    H0, Om = params[1], params[2]
    return H0 * np.sqrt(Om * zp1**3 + 1.0 - Om)


@njit
def DM_z(z, params):
    dH_grid = c / Hz(params)
    dh = (dH_grid[:-1] + dH_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dH_grid)


pivot_mask = z_cmb <= 0.10563


@njit
def mu_corr(params, DM_obs):
    v_km_s = 100 * params[3] * np.where(z_cmb <= 0.11, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_obs)


@njit
def theory_mu(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    DM = DM_z(z_cmb, params)
    diff = mu_vals - mu_corr(params, DM) - theory_mu(params[0], DM)
    return solve_triang(cho, diff)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-1.0, 1.0),  # ΔM
        (56.0, 85.0),  # H0
        (0.0, 0.8),  # Ωm
        (-6.0, 3.0),  # v x 100 km/s
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    # H0 prior from TRGB Freedman et al
    return normalization - 0.5 * (params[1] - 70.39) ** 2 / 1.80**2


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
    np.random.seed(42)
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
    MAP_params = samples[np.argmax(log_probs)]

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
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (vflow_16, vflow_50, vflow_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    DM_best = DM_z(z_cmb, best_fit)
    mu_pred = theory_mu(offset=M_50, DM=DM_best)
    corrected_mu = mu_vals - mu_corr(best_fit, DM_best)
    residuals = corrected_mu - mu_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mu_vals - np.mean(mu_vals)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot)

    rmsd = np.sqrt(np.mean(residuals**2))

    H_label = f"{H0_50:.2f} +{H0_84-H0_50:.2f} -{H0_50-H0_16:.2f} km/s/Mpc"
    M_label = f"{M_50:.3f} +{M_84-M_50:.3f} -{M_50-M_16:.3f} mag"
    Om_label = f"{Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}"
    v_label = f"{vflow_50:.2f} +{vflow_84-vflow_50:.2f} -{vflow_50-vflow_16:.2f} km/s"

    print_color("Dataset", legend)
    print_color("z range", f"{z_cmb[0]:.3f} - {z_cmb[-1]:.3f}")
    print_color("Sample size", len(z_cmb))
    print_color("ΔM", M_label)
    print_color("H0", H_label)
    print_color("Ωm", Om_label)
    print_color("v_flow", v_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{stats.skew(residuals):.3f}")
    print_color("Log evidence", f"{log_evd:.1f}")
    print_color("Chi2 (MAP)", f"{chi_squared(MAP_params):.2f}")
    print_color("Effective deg of freedom", effective_sample_size - ndim)

    y_err = np.sqrt(covmat.diagonal())

    labels = ["$ΔM$", "$H_0$", "$Ω_m$", "$v_{flow}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=corrected_mu,
        y_err=y_err,
        y_model=mu_pred,
        label=f"$Ω_m$={Om_label}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=y_err, bins=60)


if __name__ == "__main__":
    main()

"""
********************************
Dataset: DES-SN5YR Dovekie - effective: 1714 SNe
z range: 0.025 - 1.144
Sample size: 1820
********************************
"""

"""
Flat ΛCDM
ΔM: 0.020 +0.056 -0.058 mag
H0: 70.39 +1.80 -1.81 km/s/Mpc
Ωm: 0.331 +0.015 -0.015
R-squared (%): 98.38
RMSD (mag): 0.268
Skewness of residuals: 3.2
Log evidence: -825.7
Chi2 (MAP): 1631.42
Effective deg of freedom: 1711
"""

"""
Flat ΛCDM w(z) = -1
Isotropic velocity SNe observed redshifts (turning point z <= 0.10563 inflow z > 0.10563 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

ΔM: 0.005 +0.056 -0.057 mag
v: -1.41 +0.66 -0.65 km/s (prior ~ U(-6, 3)) x 100 km/s
v / (z_cut=0.10563): -1335 ± 625 km/s
H0: 70.40 +1.78 -1.79 km/s/Mpc
Ωm: 0.308 +0.018 -0.018
R-squared (%): 98.37
RMSD (mag): 0.269
Skewness of residuals: 3.2
Log evidence: -825.2
Chi2 (MAP): 1626.91 (2.12 sigma significance)
Effective deg of freedom: 1710
"""

"""
RERUN
Flat wCDM w(z) = w0
Ωm: 0.260 +0.065 -0.085
w0: -0.83 +0.14 -0.15
R-squared (%): 98.37
RMSD (mag): 0.268
Skewness of residuals: 3.2
Chi squared: 1630.18
Effective deg of freedom: 1710
"""

"""
RERUN
Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
Ωm: 0.287 +0.030 -0.035
w0: -0.81 +0.11 -0.11
wa: d w(z)/dz at z=0 = -(3/2) * (1 - w0^2)
R-squared (%): 98.37
RMSD (mag): 0.268
Skewness of residuals: 3.2
Chi squared: 1629.55
Log evidence: -825.9
Effective deg of freedom: 1710
"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
"""
