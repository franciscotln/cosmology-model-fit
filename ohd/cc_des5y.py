from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from scipy.constants import c as c0
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2025DESdovekie.data import get_data, effective_sample_size
from y2005cc.data import get_data as get_cc_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = grid[1] - grid[0]

c = c0 / 1000  # Speed of light in km/s


@njit
def Ode_z(z, w0):
    cubed = (1.0 + z) ** 3
    # return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2
    return cubed ** (1.0 + w0)  # wCDM


@njit
def H_z(z, params):
    H0, Om = params[3], params[4]
    return H0 * np.sqrt(Om * (1. + z) ** 3 + (1. - Om))


@njit
def DM_z(z, params):
    dh_grid = c / H_z(grid, params)
    dh = 0.5 * (dh_grid[:-1] + dh_grid[1:])
    cum_dm = np.zeros(grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, grid, cum_dm, dh_grid)


@njit
def get_z_cosmo(params):
    # z_turn = 0.10563
    v_km_s = 100 * params[5] * np.where(z_cmb <= 0.11, 1, -1)
    z_pec = v_km_s / c
    return -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)


def mu_corr(params):
    # For plotting purposes only
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_z(z_cmb, params))


@njit
def theory_mu(params, DM):
    return params[2] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


bounds = np.array(
    [
        (0.1, 6.0),  # f0_cc
        (-9.0, 9.0),  # fa_cc
        (-0.5, 0.5),  # ΔM
        (50.0, 85.0),  # H0
        (0.05, 0.6),  # Ωm
        (-4.5, 4.5),  # v / 100 km/s
    ]
)


@njit
def chi_squared(params, f_cc_arr):
    z_cosmo = get_z_cosmo(params)
    DM_cosmo = DM_z(z_cosmo, params)
    delta_sn = mu_vals - theory_mu(params, DM_cosmo)
    chi_sn = solve_triangular(cho_sn, delta_sn)

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = solve_triangular(cho_cc, f_cc_arr * delta_cc)

    return chi_sn + chi_cc


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


@njit
def log_likelihood(params):
    f0_cc, fa_cc = params[0], params[1]
    f_cc_arr = f0_cc + fa_cc * z_cc_vals / (1. + z_cc_vals)

    if np.any(f_cc_arr < 1e-4):
        return -np.inf

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2.0 * np.log(f_cc_arr).sum()
    return -0.5 * chi_squared(params, f_cc_arr) - 0.5 * normalization_cc


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
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
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
        (f0_16, f0_50, f0_84),
        (fa_16, fa_50, fa_84),
        (dM_16, dM_50, dM_84),
        (h0_16, h0_50, h0_84),
        (Om_16, Om_50, Om_84),
        (v_16, v_50, v_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = samples[np.argmax(log_probs)]
    DOF = effective_sample_size + N_cc - len(best_fit)

    f_cc_arr = best_fit[0] + best_fit[1] * z_cc_vals / (1. + z_cc_vals)

    print(f"f0_cc: {f0_50:.2f} +{(f0_84 - f0_50):.2f} -{(f0_50 - f0_16):.2f}")
    print(f"fa_cc: {fa_50:.1f} +{(fa_84 - fa_50):.1f} -{(fa_50 - fa_16):.1f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"v/100 km/s: {v_50:.2f} +{(v_84 - v_50):.2f} -{(v_50 - v_16):.2f}")
    print(f"Chi squared (MAP): {chi_squared(best_fit, f_cc_arr):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {DOF}")

    labels = ["$f_{0CCH}$", "$f_{aCCH}$", "$Δ_M$", "$H_0$", "$Ω_m$", "$v_{100}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_arr,
        label=f"{cc_legend}: $H_0$={h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit, DM_z(z_cmb, best_fit)),
        label=f"$Ω_m$={Om_50:.3f}, $H_0$={h0_50:.1f} km/s/Mpc",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# H0: 67.6 +1.8 -1.9 km/s/Mpc
# Ωm: 0.327 +- 0.014
#
# f0_cc: 2.92 +0.57 -0.55
# fa_cc: -3.2 +- 1.1
# ΔM: -0.070 +0.057 -0.060 mag
# Chi squared (MAP): 1671.49
# Log evidence: -979.1
# Degrees of freedom: 1748
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.10563 inflow z > 0.10563 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# H0: 68.1 +- 1.8 km/s/Mpc
# Ωm: 0.307 +- 0.016
# v/100 km/s: -1.43 +- 0.65 (prior U[-4.5, 4.5])
#
# f0_cc: 2.98 +0.57 -0.55
# fa_cc: -3.3 +1.2 -1.1
# ΔM: -0.067 +0.056 -0.058 mag
# Chi squared (MAP): 1665.86 (2.37 sigma significance)
# Log evidence: -978.4
# Degrees of freedom: 1747
# ---------------------------------


# ----------- Flat wCDM -----------
# H0: 67.4 +1.8 -1.8 km/s/Mpc
# Ωm: 0.294 +0.040 -0.048
# w0: -0.91 +- 0.11 (prior U[-2, 0])
#
# f0_cc: 2.95 +0.57 -0.55
# fa_cc: -3.3 +- 1.1
# ΔM: -0.071 +0.058 -0.059 mag
# Chi squared (MAP): 1671.09 (0.63 sigma significance)
# Log evidence: -980.7
# Degrees of freedom: 1747
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
#
# H0: 67.3 +1.9 -1.8 km/s/Mpc
# Ωm: 0.284 +0.031 -0.044
# w0: -0.88 +0.09 -0.08 (prior U[-1, -1/3])
#
# f0_cc: 2.97 +0.56 -0.55
# fa_cc: -3.4 +- 1.1
# ΔM: -0.071 +- 0.058 mag
# Chi squared (MAP): 1671.38 (0.33 sigma significance)
# Log evidence: -979.6 (inacurate: truncated posterior, needs to be done with Nautilus)
# Degrees of freedom: 1747
# ---------------------------------


# ---------- Flat w0waCDM ---------
# TODO
# ---------------------------------
