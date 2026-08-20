from numba import njit
import numpy as np
import scipy.stats as stats
from scipy.linalg import cho_factor, solve_triangular
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2022pantheonSHOES.data import get_data

legend, z_cmb, z_hel, mb_vals, cov_matrix = get_data()

c = c0 / 1000  # Speed of light (km/s)

cho = cho_factor(cov_matrix, lower=True)[0]

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = np.diff(z_grid)

cubed = (1.0 + z_grid) ** 3


@njit
def Ode_z(w0):
    # Thawing quintessence
    return (2 * cubed / ((1.0 + w0) + (1.0 - w0) * cubed)) ** 2


@njit
def H_z(params):
    H0, Om = params[1], params[2]
    return H0 * np.sqrt(Om * cubed + (1.0 - Om))


@njit
def DM_z(params, z):
    dh_grid = c / H_z(params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def mu_corr(params, DM_ref):
    # Heaviside step at z = 0.15
    v_km_s = 100 * params[3] * np.where(z_cmb <= 0.15, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    return 5.0 * np.log10(DM_z(params, z_cosmo) / DM_ref)


@njit
def mu_theory(DM):
    return 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    DM = DM_z(params, z_cmb)
    delta = mb_vals - params[0] - mu_corr(params, DM) - mu_theory(DM)
    return solve_triang(cho, delta)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-20.0, -19.0),  # M
        (50.0, 90.0),  # H0
        (0.0, 0.7),  # Ωm
        (-3.0, 3.0),  # v x 100 km/s
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

    with Pool(6) as pool:
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
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (v_16, v_50, v_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    DM = DM_z(best_fit, z_cmb)
    mB_pred = mu_theory(DM) + M0_50
    corrected_mags = mb_vals - mu_corr(best_fit, DM)
    residuals = corrected_mags - mB_pred

    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    mean_app_mag = np.mean(corrected_mags)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((corrected_mags - mean_app_mag) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    rmsd = np.sqrt(np.mean(residuals**2))

    M_label = f"{M0_50:.3f} +{M0_84-M0_50:.3f}/-{M0_50-M0_16:.3f}"
    H0_label = f"{H0_50:.2f} +{H0_84-H0_50:.2f}/-{H0_50-H0_16:.2f} km/s/Mpc"
    omega_label = f"{Om_50:.3f} +{Om_84-Om_50:.3f}/-{Om_50-Om_16:.3f}"
    v_label = f"{v_50:.3f} +{v_84-v_50:.3f}/-{v_50-v_16:.3f} x 100 km/s"

    print_color("Dataset", legend)
    print_color("z range", f"{z_cmb[0]:.4f} - {z_cmb[-1]:.4f}")
    print_color("M", M_label)
    print_color("H0", H0_label)
    print_color("Ωm", omega_label)
    print_color("v", v_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Degs of freedom", len(z_cmb) - len(best_fit))
    print_color("Chi squared", f"{chi_squared(best_fit):.2f}")
    print_color("Log Evidence", f"{log_evd:.1f}")

    labels = ["$M_0$", "$H_0$", "$Ω_m$", "$v_{100}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=corrected_mags - M0_50,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=mB_pred - M0_50,
        label=f"$Ω_m$={Om_50:.3f}",
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
M: -19.339 +0.055/-0.057 mag
H0: 70.38 +- 1.80 km/s/Mpc
Ωm: 0.332 +0.018/-0.018
R-squared (%): 99.74
RMSD (mag): 0.153
Skewness of residuals: 0.090
kurtosis of residuals: 1.582
Degs of freedom: 1587
Chi squared: 1402.92
Log Evidence: -711.0

=============================

Flat ΛCDM
Isotropic velocity SNe observed redshifts (turning point z <= 0.15 inflow z > 0.15 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

M: -19.351 +0.055/-0.057
H0: 70.40 +1.79/-1.80 km/s/Mpc
Ωm: 0.315 +0.021/-0.020
v: -0.68 +0.41/-0.41 x 100 km/s (prior ~ U[-3, 3])
R-squared (%): 99.75
RMSD (mag): 0.154
Skewness of residuals: 0.055
kurtosis of residuals: 1.586
Degs of freedom: 1586
Chi squared: 1400.15 (1.55 sigma significance)
Log Evidence: -711.4
"""

"""
wCDM
M: -19.335 +0.055/-0.057
H0: 70.39 +1.80/-1.79 km/s/Mpc
Ωm: 0.292 +0.063/-0.076
w0: -0.901 +0.140/-0.159 (prior ~ U[-1.5, -0.5])
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.079
kurtosis of residuals: 1.590
Degs of freedom: 1586
Chi squared: 1402.47
Log Evidence: -711.7
"""

"""
Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
M: -19.332 +0.055/-0.057 mag
H0: 70.40 +1.79/-1.81 km/s/Mpc
Ωm: 0.299 +0.028/-0.034
w0: -0.873 +0.102/-0.083 (prior ~ U[-1.0, -1/3]) truncated posterior
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.074
kurtosis of residuals: 1.592
Degs of freedom: 1586
Chi squared: 1402.70
Log Evidence: -711.5 (not accurate due to truncation of w0 posterior)
"""

"""
Flat w0waCDM
TODO: re-run after adding H0 prior
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
