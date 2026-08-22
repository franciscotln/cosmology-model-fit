from numba import njit
import numpy as np
import scipy.stats as stats
from scipy.linalg import cho_factor
from scipy.constants import c as c0
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2022pantheonSHOES.data import get_data_with_position

legend, z_cmb, z_hel, mb_vals, ra, dec, cov_matrix = get_data_with_position()
cho = cho_factor(cov_matrix, lower=True)[0]

c = c0 / 1000  # Speed of light (km/s)

ra_rad = np.deg2rad(ra)
dec_rad = np.deg2rad(dec)
nx = np.cos(dec_rad) * np.cos(ra_rad)
ny = np.cos(dec_rad) * np.sin(ra_rad)
nz = np.sin(dec_rad)

# Target direction coordinates
ra_fixed_deg = 217  # deg
dec_fixed_deg = -29  # deg
# Convert direction to unit vector (dx, dy, dz)
ra_f_rad = np.deg2rad(ra_fixed_deg)
dec_f_rad = np.deg2rad(dec_fixed_deg)
d1 = np.cos(dec_f_rad) * np.cos(ra_f_rad)
d2 = np.cos(dec_f_rad) * np.sin(ra_f_rad)
d3 = np.sin(dec_f_rad)
cos_angle = nx * d1 + ny * d2 + nz * d3

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = np.diff(z_grid)

cubed = (1.0 + z_grid) ** 3


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
def mu_theory(DM):
    return 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def mu_corr(params, DM_obs):
    DZ = 0.02  # sharpness of the transition
    Z_C = 0.10  # redshift where the velocity drops by 50%
    attenuation = 0.5 * (1.0 - np.tanh((z_cmb - Z_C) / DZ))
    v_km_s = 100 * params[3] * cos_angle * attenuation
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    return 5.0 * np.log10(DM_z(params, z_cosmo) / DM_obs)


@njit
def chi_squared(params):
    DM = DM_z(params, z_cmb)
    delta = mb_vals - params[0] - mu_corr(params, DM) - mu_theory(DM)
    return solve_triangular(cho, delta)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array(
    [
        (-20, -19),  # M
        (50, 90),  # H0
        (0, 0.7),  # Ωm
        (-1, 4), # v (x 100 km/s) dipole towards the Shapley supercluster (v > 0)
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    # H0 prior from TRGB Freedman et al
    return normalization - 0.5 * (params[1] - 70.39) ** 2 / 1.80**2


@njit
def log_probability_jit(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp - 0.5 * chi_squared(params)


def log_probability(params):
    return log_probability_jit(params)


def main():
    import emcee
    from multiprocessing import Pool
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions, print_color, plot_residuals

    n_dim = len(bounds)
    n_walkers = 150
    burn_in = 500
    n_steps = burn_in + 2000
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
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
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

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

    M_label = f"{M0_50:.3f} +{M0_84-M0_50:.3f}/-{M0_50-M0_16:.3f}"
    H0_label = f"{H0_50:.2f} +{H0_84-H0_50:.2f}/-{H0_50-H0_16:.2f} km/s/Mpc"
    omega_label = f"{Om_50:.3f} +{Om_84-Om_50:.3f}/-{Om_50-Om_16:.3f}"
    v_label = f"{v_50:.3f} +{v_84-v_50:.3f}/-{v_50-v_16:.3f} x 100 km/s"

    print_color("Dataset", legend)
    print_color("z range", f"{z_cmb[0]:.4f} - {z_cmb[-1]:.4f}")
    print_color("M", M_label)
    print_color("H0", H0_label)
    print_color("Ωm", omega_label)
    print_color("v (dipole)", v_label)
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("DOF", len(z_cmb) - len(best_fit))
    print_color("Chi squared", f"{chi_squared(best_fit):.2f}")
    print_color("Log Evidence", f"{log_evd:.1f}")

    labels = ["$M_0$", "$H_0$", "$Ω_m$", "$v/100$"]
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


# ----------- Flat ΛCDM -----------
# M: -19.341 +0.055/-0.057
# H0: 70.4 +1.8/-1.8 km/s/Mpc
# Ωm: 0.332 +0.018/-0.018
# v (dipole): 129 +38/-38 km/s
# Skewness of residuals: 0.085
# DOF: 1586
# Chi squared: 1391.22
# Log Evidence: -706.8
# ---------------------------------


# ----------- Flat ΛCDM (v=0) -----
# M: -19.339 +0.055/-0.057 mag
# H0: 70.38 +- 1.80 km/s/Mpc
# Ωm: 0.332 +0.018/-0.018
# Skewness of residuals: 0.090
# DOF: 1587
# Chi squared: 1402.92
# Log Evidence: -711.0
# ---------------------------------