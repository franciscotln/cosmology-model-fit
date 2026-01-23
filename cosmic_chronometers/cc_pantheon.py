from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2022pantheonSHOES.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_cc_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
legend, z_vals, z_hel_vals, app_mag_values, cov_matrix_sn = get_sn_data()

cov_sn_cho = cho_factor(cov_matrix_sn, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

z_grid = np.linspace(0, np.max(z_vals) + 0.1, num=1000)
dx = np.diff(z_grid)

c = c0 / 1000  # Speed of light in km/s


@njit
def Ez(z, Om, w0):
    cubed = (1 + z) ** 3
    rho_de = ((4 * cubed) / (1 + 3 * cubed)) ** (4 * (1 + w0))
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


@njit
def H_z(z, params):
    H0, Om, w0 = params[1], params[3], params[4]
    return H0 * Ez(z, Om, w0)


@njit
def DM_z(z, params):
    dh_grid = c / H_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def app_mag_theory(params):
    dL = (1 + z_hel_vals) * DM_z(z_vals, params)
    return params[2] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_sn = app_mag_values - app_mag_theory(params)
    chi_sn = solve_triang(cov_sn_cho, delta_sn)

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = solve_triang(cho_cc, delta_cc) * params[0] ** -2

    return chi_sn + chi_cc


bounds = np.array(
    [
        (0.1, 1.5),  # f_cc
        (55, 80),  # H0
        (-20, -19),  # M
        (0.15, 0.70),  # Ωm
        (-2.0, 0.0),  # w0
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
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc + 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee
    from corner_plot import plot_corner_and_chains
    from multiprocessing import Pool

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 1500 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.3),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
    ]

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [h0_16, h0_50, h0_84],
        [M_16, M_50, M_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    deg_of_freedom = len(z_vals) + len(z_cc_vals) - len(best_fit)

    Omh2_samples = samples[:, 3] * (samples[:, 1] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) * f_cc_50,
        label=f"{cc_legend}: $H_0$={h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=app_mag_values - M_50,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=app_mag_theory(best_fit) - M_50,
        label=f"Best fit: $H_0$={h0_50:.1f} km/s/Mpc, $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$f_{CC}$", "$H_0$", "M", "$Ω_m$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
f_cc: 0.70 +0.10 -0.08
H0: 67.1 +2.5 -2.5 km/s/Mpc
M: -19.445 +0.077 -0.079 mag
Ωm: 0.331 +0.017 -0.017
ωm: 0.1490
w0: -1
wa: 0
Chi squared: 1433.17
Degrees of freedom: 1619

===============================

Flat wCDM: w(z) = w0
f_cc: 0.71 +0.10 -0.08
H0: 67.4 +2.7 -2.6 km/s/Mpc
M: -19.432 +0.085 -0.087 mag
Ωm: 0.317 +0.040 -0.043
ωm: 0.1432 +0.0159 -0.0173
w0: -0.958 +0.102 -0.113
wa: 0
Chi squared: 1432.42
Degrees of freedom: 1618

===============================

Flat alternative: w(z) = -1 + (1 + w0) / (1 + z)**3
f_cc: 0.70 +0.10 -0.08
H0: 67.3 +2.6 -2.6
M: -19.435 +0.083 -0.086
Ωm: 0.322 +0.029 -0.029
ωm: 0.1455 +0.0126 -0.0124
w0: -0.958 +0.101 -0.113
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
Chi squared: 1432.53
Degrees of freedom: 1618
"""
