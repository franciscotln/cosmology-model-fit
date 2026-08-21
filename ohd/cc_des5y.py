from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from scipy.constants import c as c0
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2025DESdovekie.data import get_data, effective_sample_size
from y2005cc.data import get_data as get_cc_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, observed_mu_vals, cov_matrix_sn = get_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_cc = np.linalg.inv(cov_matrix_cc)
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dx = np.diff(grid)

c = c0 / 1000  # Speed of light in km/s


@njit
def Ode_z(z, w0):
    cubed = (1.0 + z) ** 3
    # return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2
    return cubed ** (1.0 + w0)  # wCDM


@njit
def Ez(z, params):
    Om, w0 = params[3], params[4]
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0))


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


@njit
def DM_z(z, params):
    dh_grid = c / H_z(grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_hermite(z, grid, cum_dm, dh_grid)


mu_z_term = 25.0 + 5 * np.log10(1.0 + z_hel)


@njit
def theory_mu(params):
    return params[1] + mu_z_term + 5 * np.log10(DM_z(z_cmb, params))


bounds = np.array(
    [
        (0.2, 3),  # f_cc
        (-0.5, 0.5),  # ΔM
        (50, 85),  # H0
        (0.05, 0.6),  # Ωm
        (-1, -1 / 3),  # w0
    ],
    dtype=np.float64,
)


@njit
def chi_squared(params):
    delta_sn = observed_mu_vals - theory_mu(params)
    chi_sn = solve_triangular(cho_sn, delta_sn)

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = params[0] ** 2 * delta_cc.dot(np.dot(inv_cov_cc, delta_cc))

    return chi_sn + chi_cc


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


@njit
def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


@njit
def log_probability_jit(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def log_probability(params):
    return log_probability_jit(params)


def main():
    import emcee
    from multiprocessing import Pool
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_cc_predictions

    ndim = len(bounds)
    nwalkers = 100
    burn_in = 350
    nsteps = 3500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.25),
        (emcee.moves.DEMove(), 0.75),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

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
        (fcc_16, fcc_50, fcc_84),
        (dM_16, dM_50, dM_84),
        (h0_16, h0_50, h0_84),
        (Om_16, Om_50, Om_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    deg_of_freedom = effective_sample_size + z_cc_vals.size - len(best_fit)

    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    labels = ["$f_{CCH}$", "$Δ_M$", "$H_0$", "$Ω_m$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / fcc_50,
        label=f"{cc_legend}: $H_0$={h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=observed_mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}, $H_0$={h0_50:.1f} km/s/Mpc",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# f_cc: 1.49 +0.18 -0.17
# ΔM: -0.090 +0.071 -0.074 mag
# H0: 66.9 +2.3 -2.3 km/s/Mpc
# Ωm: 0.330 +0.014 -0.014
# Chi squared: 1666.61
# Log evidence: -968.2
# Degrees of freedom: 1746
# ---------------------------------


# ----------- Flat wCDM -----------
# f_cc: 1.47 +0.18 -0.17
# ΔM: -0.068 +0.081 -0.081 mag
# H0: 67.4 +2.6 -2.5 km/s/Mpc
# Ωm: 0.307 +0.038 -0.042
# w0: -0.931 +0.100 -0.107 (prior width 1: -1.5 to -0.3)
# Chi squared: 1665.49
# Log evidence: -969.5
# Degrees of freedom: 1745
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
#
# f_cc: 1.47 +0.18 -0.17
# ΔM: -0.055 +0.076 -0.076 mag
# H0: 67.7 +2.4 -2.4 km/s/Mpc
# Ωm: 0.306 +0.021 -0.023
# w0: -0.875 +0.087 -0.078 (prior width 2/3: -1 to -1/3 - Truncated)
# wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
# Chi squared: 1664.70
# Log evidence: -968.6 (inacurate: truncated posterior, needs to be done with Nautilus)
# Degrees of freedom: 1745
# ---------------------------------


# ---------- Flat w0waCDM ---------
# TODO
# ---------------------------------
