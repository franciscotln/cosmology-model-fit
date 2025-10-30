from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2024DES.data import get_data, effective_sample_size
from y2005cc.data import get_data as get_cc_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, observed_mu_vals, cov_matrix_sn = get_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

grid = np.linspace(0, np.max(z_cmb) + 0.1, num=1000)
dx = np.diff(grid)

c = 299792.458  # Speed of light in km/s


@njit
def Ez(z, params):
    Om, w0 = params[3], params[4]
    cubed = (1 + z) ** 3
    rho_de = np.exp((1 + w0) * (1 - 1 / cubed))
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


@njit
def DM_z(z, params):
    dh_grid = c / H_z(grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, grid, cum_dm)


@njit
def theory_mu(params):
    return params[1] + 25 + 5 * np.log10((1 + z_hel) * DM_z(z_cmb, params))


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


bounds = np.array(
    [
        (0.4, 2.5),  # f_cc
        (-0.6, 0.6),  # ΔM
        (55, 80),  # H0
        (0.1, 0.6),  # Ωm
        (-2.0, 0.0),  # w0
    ],
    dtype=np.float64,
)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_sn = observed_mu_vals - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = params[0] ** 2 * solve_triang(cho_cc, delta_cc)

    return chi_sn + chi_cc


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee
    from multiprocessing import Pool
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_cc_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.3),
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

    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [dM_16, dM_50, dM_84],
        [h0_16, h0_50, h0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([f_cc_50, dM_50, h0_50, Om_50, w0_50], dtype=np.float64)
    deg_of_freedom = effective_sample_size + z_cc_vals.size - len(best_fit)

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_50,
        label=f"{cc_legend}: $H_0$={h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=observed_mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}, $H_0$={h0_50:.1f} km/s/Mpc",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$f_{CCH}$", "$Δ_M$", "$H_0$", "$Ω_m$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
f_cc: 1.47 +0.19 -0.18
ΔM: -0.112 +0.072 -0.075 mag
H0: 65.9 +2.3 -2.3 km/s/Mpc
Ωm: 0.349 +0.016 -0.016
w0: -1
wa: 0
Chi squared: 1672.35
Degrees of freedom: 1764

==============================

Flat wCDM: w(z) = w0
f_cc: 1.46 +0.18 -0.18
ΔM: -0.075 +0.082 -0.083 mag
H0: 66.8 +2.6 -2.5 km/s/Mpc
Ωm: 0.308 +0.042 -0.047
w0: -0.884 +0.100 -0.109
wa: 0
Chi squared: 1670.64
Degrees of freedom: 1763

==============================

Flat alternative: w(z) = -1 + (1 + w0) / (1 + z)^3
f_cc: 1.46 +0.18 -0.18
ΔM: -0.070 +0.079 -0.080 mag
H0: 66.8 +2.5 -2.4 km/s/Mpc
Ωm: 0.318 +0.028 -0.027
w0: -0.838 +0.101 -0.113
wa: d w(z)/dz at z=0 = -3 * (1 + w0)
Chi squared: 1669.73
Log evidence: -962.0
Degrees of freedom: 1763

==============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
f_cc: 1.45 +0.18 -0.18
ΔM: -0.177 +0.087 -0.087 mag
H0: 63.0 +2.7 -2.6 km/s/Mpc
Ωm: 0.452 +0.030 -0.041
w0: -0.544 +0.215 -0.208
wa: -5.585 +2.465 -2.489
Chi squared: 1664.59
Degrees of freedom: 1762
"""
