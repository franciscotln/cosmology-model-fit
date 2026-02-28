from numba import njit
import numpy as np
import scipy.stats as stats
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2022pantheonSHOES.data_shoes import get_data

legend, z_cmb, z_hel, mB_vals, ceph_dists, cov_matrix = get_data()

ceph_mask = ceph_dists != -9

cho = cho_factor(cov_matrix, lower=True)[0]

c = c0 / 1000  # Speed of light (km/s)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=3000)
dz = np.diff(z_grid)

zp1 = 1.0 + z_grid
zp1_hel = 1.0 + z_hel


@njit
def Ode_z(params):
    w0 = params[3]
    return zp1 ** (3 * (1.0 + w0))  # wCDM
    # return (2 * zp1**3 / ((1.0 + w0) + (1.0 - w0) * zp1**3)) ** 2  #  quintessence


@njit
def Hz(params):
    H0, O_m = params[1], params[2]
    return H0 * np.sqrt(O_m * zp1**3 + (1.0 - O_m))


@njit
def DM_z(z, theta):
    dh_grid = c / Hz(theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


correction_mask = (z_cmb <= 0.1) & ~ceph_mask


@njit
def mu_corr(params):
    z_pec = 100 * params[3] / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)

    return np.where(
        correction_mask,
        5.0 * np.log10(DM_z(z_cosmo, params) / DM_z(z_cmb, params)),
        0.0,
    )


@njit
def mu_theory(params):
    return 25.0 + 5.0 * np.log10(zp1_hel * DM_z(z_cmb, params))


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    mu_pred = np.where(ceph_mask, ceph_dists, mu_theory(params))
    mB_theory = mu_pred + params[0] + mu_corr(params)
    delta = mB_vals - mB_theory
    return solve_triang(cho, delta)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-20.0, -18.5),  # M
        (60.0, 85.0),  # H0
        (0.1, 0.6),  # Ωm
        (-3.5, 2.0),  # v / 100 km/s
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
    from log_evidence import log_evidence
    from sn.plotting import plot_predictions, print_color, plot_residuals

    n_dim = len(bounds)
    n_walkers = 150
    burn_in = 500
    n_steps = burn_in + 2500
    np.random.seed(42)
    state0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.2),
        (emcee.moves.DEMove(), 0.8),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool, moves)
        sampler.run_mcmc(
            state0, n_steps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)
    print_color("Log Evidence", f"{log_evd:.2f}")

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Acceptance fraction", np.mean(sampler.acceptance_fraction))
        print_color(
            "effective samples", n_dim * n_walkers * (n_steps - burn_in) / np.max(tau)
        )
    except:
        print_color("Autocorrelation time", "Not available")

    [
        (M_16, M_50, M_84),
        (H0_16, H0_50, H0_84),
        (omega_16, omega_50, omega_84),
        (v_16, v_50, v_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    mu_pred = mu_theory(best_fit)
    mB_corrected = mB_vals - mu_corr(best_fit)
    residuals = mB_corrected - M_50 - np.where(ceph_mask, ceph_dists, mu_pred)

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mB_corrected - np.mean(mB_corrected)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    rmsd = np.sqrt(np.mean(residuals**2))

    M_label = f"{M_50:.3f} +{M_84-M_50:.3f}/-{M_50-M_16:.3f} mag"
    H0_label = f"{H0_50:.2f} +{H0_84-H0_50:.2f}/-{H0_50-H0_16:.2f} km/s/Mpc"
    omega_label = f"{omega_50:.3f} +{omega_84-omega_50:.3f}/-{omega_50-omega_16:.3f}"
    v_flow_label = f"{v_50:.3f} +{v_84-v_50:.3f}/-{v_50-v_16:.3f} x 100 km/s"

    print_color("Dataset", legend)
    print_color("z range", f"{z_cmb[0]:.4f} - {z_cmb[-1]:.4f}")
    print_color("Sample size", len(z_cmb))
    print_color("M", M_label)
    print_color("H0", H0_label)
    print_color("Ωm", omega_label)
    print_color("v", v_flow_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{stats.skew(residuals):.3f}")
    print_color("kurtosis of residuals", f"{stats.kurtosis(residuals):.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit):.2f}")

    mu_std = np.sqrt(cov_matrix.diagonal())

    labels = ["M", "$H_0$", "$Ω_m$", "$v_{100}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=mB_corrected - M_50,
        y_err=mu_std,
        y_model=mu_pred,
        label=f"H0={H0_50:.2f} km/s/Mpc",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=mu_std, bins=65)


if __name__ == "__main__":
    main()

"""
*****************************
Dataset: Pantheon+ and SH0ES
z range: 0.0012 - 2.2614
Sample size: 1657
*****************************

ΛCDM

M: -19.24 +0.03/-0.03 mag
H0 (km/s/Mpc): 73.5 +- 1.0
Ωm: 0.332 +0.018/-0.018
R-squared: 99.78 %
RMSD (mag): 0.153
Skewness of residuals: 0.085
kurtosis of residuals: 1.555
Chi squared: 1452.02

=============================

ΛCDM
Isotropic velocity SNe observed redshifts (limit to z <= 0.1)
z_cosmo = -1 + (1 + z) / (1 + v/c)

M: -19.242 +0.029/-0.029 mag
H0: 74.01 +1.07/-1.06 km/s/Mpc
Ωm: 0.317 +0.020/-0.020
v: -0.68 +0.45/-0.45 x 100 km/s
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.052
kurtosis of residuals: 1.558
Chi squared: 1449.61 (1.55 sigma significance)
Log Evidence: -736.14

=============================

wCDM

M: -19.24 +0.03/-0.03 mag
H0 (km/s/Mpc): 73.5 +1.0/-1.0 km/s/Mpc
Ωm: 0.301 +0.062/-0.075
w0: -0.92 +0.14/-0.16
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.076
kurtosis of residuals: 1.561
Chi squared: 1451.70 (0.57 sigma significance)

=============================

Flat wzCDM w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)

M: -19.243 +0.030/-0.029
H0 (km/s/Mpc): 73.34 +1.04/-1.01
Ωm: 0.300 +0.028/-0.033
w0: -0.877 +0.101/-0.080
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
R-squared (%): 99.78
RMSD (mag): 0.153
Skewness of residuals: 0.070
kurtosis of residuals: 1.564
Chi squared: 1451.86 (0.40 sigma significance)
"""
