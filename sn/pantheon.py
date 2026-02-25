from numba import njit
import numpy as np
import scipy.stats as stats
from scipy.linalg import cho_factor, solve_triangular
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2022pantheonSHOES.data import get_data

legend, z_cmb, z_hel, apparent_mag_vals, cov_matrix = get_data()

c = c0 / 1000  # Speed of light (km/s)
H0 = 70.0  # Hubble constant (km/s/Mpc)

cho = cho_factor(cov_matrix, lower=True)[0]

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=3000)
dz = np.diff(z_grid)

cubed = (1.0 + z_grid) ** 3


@njit
def Ode_z(w0):
    # Thawing quintessence
    return (2 * cubed / ((1.0 + w0) + (1.0 - w0) * cubed)) ** 2


@njit
def Ez(params):
    Om = params[1]
    return np.sqrt(Om * cubed + (1.0 - Om))


@njit
def DM_z(params):
    dh_grid = (c / H0) / Ez(params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z_cmb, z_grid, cum_dm, dh_grid)


@njit
def outflow_correction(params):
    # up to second order in z
    Om, v_100 = params[1], params[2]
    q0 = 1.5 * Om - 1.0
    q_term = (1.0 - q0) * z_cmb
    q_corr = (1.0 + q_term) / (1.0 + 0.5 * q_term)
    v_ratio = 100 * v_100 / (c * z_cmb)
    return v_ratio * (5 / np.log(10)) * q_corr


@njit
def apparent_mag(params):
    Mz = params[0] + outflow_correction(params)
    return Mz + 25.0 + 5 * np.log10((1.0 + z_hel) * DM_z(params))


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta = apparent_mag_vals - apparent_mag(params)
    return solve_triang(cho, delta)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-20.0, -19.0),  # M
        (0.0, 0.7),  # Ωm
        (-1.70, 3.20),  # v_flow [x 100 km/s]
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
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions, print_color, plot_residuals

    burn_in = 500
    n_dim = len(bounds)
    n_walkers = 150
    n_steps = burn_in + 2500
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.30),
        (emcee.moves.DEMove(), 0.70),
    ]

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability, pool=pool, moves=moves
        )
        sampler.run_mcmc(
            initial_pos, n_steps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    print_color("Gelman-Rubin", gelman_rubin(chains_samples))
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Acceptance fraction", np.mean(sampler.acceptance_fraction))
        print_color(
            "effective samples", n_walkers * (n_steps - burn_in) * n_dim / np.max(tau)
        )
    except:
        print_color("Autocorrelation time", "Not available")

    [
        (M0_16, M0_50, M0_84),
        (Om_16, Om_50, Om_84),
        (vf_16, vf_50, vf_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    predicted_apparent_mag = apparent_mag(best_fit)
    residuals = apparent_mag_vals - predicted_apparent_mag

    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(apparent_mag_vals)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((apparent_mag_vals - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    M_label = f"{M0_50:.3f} +{M0_84-M0_50:.3f}/-{M0_50-M0_16:.3f}"
    omega_label = f"{Om_50:.3f} +{Om_84-Om_50:.3f}/-{Om_50-Om_16:.3f}"
    vf_label = f"{vf_50:.3f} +{vf_84-vf_50:.3f}/-{vf_50-vf_16:.3f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_cmb[0]:.4f} - {z_cmb[-1]:.4f}")
    print_color("M", M_label)
    print_color("Ωm", omega_label)
    print_color("v_flow", vf_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Degs of freedom", len(z_cmb) - len(best_fit))
    print_color("Chi squared", f"{chi_squared(best_fit):.2f}")
    print_color("Log Evidence", f"{log_evd:.1f}")

    labels = ["$M_0$", "$Ω_m$", "$v_{flow}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=apparent_mag_vals - M0_50,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=predicted_apparent_mag - M0_50,
        label=f"$Ω_m$={Om_50:.3f}, $M$={M0_50:.3f}",
        x_scale="log",
    )
    plot_residuals(
        z_values=z_cmb,
        residuals=residuals,
        y_err=np.sqrt(np.diag(cov_matrix)),
        bins=40,
    )


if __name__ == "__main__":
    main()


"""
*****************************
Dataset: Pantheon+ (2022)
z range: 0.0102 - 2.2614
Sample size: 1590
*****************************
"""

"""
ΛCDM
M: -19.351 +0.007/-0.007
Ωm: 0.332 +0.018/-0.018 (agreement with BAO within 1.94 sigma)
R-squared (%): 99.74
RMSD (mag): 0.153
Skewness of residuals: 0.090
kurtosis of residuals: 1.582
Degs of freedom: 1588
Chi squared: 1402.92
Log Evidence: -708.9

=============================

Flat ΛCDM w(z) = -1
Void outflow correction to absolute mag of SNe M(z) = M_inf + v_flow_corr
v_flow_corr = 100 * v_flow * (5 / ln(10)) / (c * z_cmb) * q_term with v_flow in units 100 km/s

M_inf: -19.367 +0.012/-0.012 mag
v_flow: 76.2 +- 48.9 km/s (prior ~ U(-1.7, 3.2) x 100 km/s)
Ωm: 0.315 +0.021/-0.020
R-squared (%): 99.74
RMSD (mag): 0.153
Skewness of residuals: 0.055
kurtosis of residuals: 1.580
Degs of freedom: 1587
Chi squared: 1400.44
Log Evidence: -709.0
"""

"""
wCDM
M: -19.347 +0.009/-0.009
Ωm: 0.292 +0.064/-0.078
w0: -0.901 +0.142/-0.159 (prior width 1: -1.5 to -0.5)
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.079
kurtosis of residuals: 1.589
Degs of freedom: 1587
Chi squared: 1402.47
Log Evidence: -709.5
"""

"""
Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
M: -19.344 +0.008/-0.008
Ωm: 0.299 +0.028/-0.034
w0: -0.873 +0.103/-0.084 (prior width 2/3: -1.0 to -1/3)
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.074
kurtosis of residuals: 1.592
Degs of freedom: 1587
Chi squared: 1402.68
Log Evidence: -709.3
"""

"""
Flat w0waCDM
M0: 19.348 +0.010/-0.010 mag
Ωm: 0.337 +0.082/-0.148
w0: -0.919 +0.146/-0.162 (0.53 sigma)
wa: -0.3614 +1.0279/-1.8010 (0.26 sigma)
R-squared: 99.74 %
RMSD (mag): 0.154
Skewness of residuals: 0.076
kurtosis of residuals: 1.601
Degs of freedom: 1586
"""
