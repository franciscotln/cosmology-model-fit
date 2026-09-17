from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from scipy.constants import c as c0
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2025DESdovekie.data import get_data, effective_sample_size
from y2005cc.data import get_data as get_cc_data

cc_legend, z_cc_vals, H_cc_vals, diag_stat_cc, cov_mat_sys_cc = get_cc_data(split_sys=True)
sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
N_cc = len(z_cc_vals)

grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = grid[1] - grid[0]

c = c0 / 1000  # Speed of light in km/s


@njit
def Ode_z(z, w0, wa=0.0):
    cubed = (1.0 + z) ** 3
    return cubed ** (1.0 + w0)  # wCDM
    # return cubed ** (1.0 + w0 + wa) * np.exp(-3 * wa * z / (1.0 + z))  # w0waCDM
    # return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2  # thawing quintessence


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
    offset = 1e-3 * params[5] * np.where(z_cmb <= 0.10563, 1, -1)
    return z_cmb + offset


def mu_corr(params):
    # For plotting purposes only
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_z(z_cmb, params))


@njit
def theory_mu(params, DM):
    return params[2] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


bounds = np.array(
    [
        (np.log(0.25), np.log(1.40)),  # ln(fp_cc)
        (-4.0, 4.0),  # n_cc
        (-0.5, 0.5),  # ΔM
        (50.0, 85.0),  # H0
        (0.05, 0.6),  # Ωm
        (-1.5, 1.5),  # 1000 Δz
    ]
)


@njit
def chi_squared(params, L_cc):
    z_cosmo = get_z_cosmo(params)
    DM_cosmo = DM_z(z_cosmo, params)
    delta_sn = mu_vals - theory_mu(params, DM_cosmo)
    y_sn = solve_triangular(cho_sn, delta_sn)
    chi_sn = np.dot(y_sn, y_sn)

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    y_cc = solve_triangular(L_cc, delta_cc)
    chi_cc = np.dot(y_cc, y_cc)

    return chi_sn + chi_cc


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


z_pivot = 0.696


@njit
def log_likelihood(params):
    fp_cc, n_cc = np.exp(params[0]), params[1]
    fz_cc = fp_cc * ((1.0 + z_cc_vals) / (1.0 + z_pivot))**n_cc

    cov_mat_cc = cov_mat_sys_cc + np.diag(diag_stat_cc**2 * fz_cc**2) 
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2.0 * np.sum(np.log(np.diag(L_cc)))

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc
    return -0.5 * chi_squared(params, L_cc) - 0.5 * normalization_cc


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
    from ohd.plot_predictions import plot_cc_predictions

    ndim = len(bounds)
    nwalkers = 100
    burn_in = 350
    nsteps = 3500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [(emcee.moves.KDEMove(), 0.20), (emcee.moves.DEMove(), 0.80)]

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
        (ln_fp_16, ln_fp_50, ln_fp_84),
        (n_16, n_50, n_84),
        (dM_16, dM_50, dM_84),
        (h0_16, h0_50, h0_84),
        (Om_16, Om_50, Om_84),
        (dz1000_16, dz1000_50, dz1000_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = samples[np.argmax(log_probs)]
    DOF = effective_sample_size + N_cc - len(best_fit)

    fz_cc = np.exp(best_fit[0]) * ((1.0 + z_cc_vals) / (1.0 + z_pivot))**best_fit[1]
    cov_mat_cc = cov_mat_sys_cc + np.diag(diag_stat_cc**2 * fz_cc**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)

    print(f"ln(fp_cc): {ln_fp_50:.2f} +{(ln_fp_84 - ln_fp_50):.2f} -{(ln_fp_50 - ln_fp_16):.2f}")
    print(f"n_cc: {n_50:.2f} +{(n_84 - n_50):.2f} -{(n_50 - n_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"1000 Δz: {dz1000_50:.2f} +{(dz1000_84 - dz1000_50):.2f} -{(dz1000_50 - dz1000_16):.2f}")
    print(f"Chi squared (MAP): {chi_squared(best_fit, L_cc):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"DOF: {DOF}")

    labels = ["$ln(f_{pCCH})$", "$n_{CCH}$", "$Δ_M$", "$H_0$", "$Ω_m$", "1000 Δz"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(diag_stat_cc**2 + np.diag(cov_mat_sys_cc)),
        label=f"{cc_legend}: $H_0$={best_fit[3]:.1f} km/s/Mpc",
        err_scaling=1 / fz_cc,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit, DM_z(z_cmb, best_fit)),
        label=f"$Ω_m$={best_fit[4]:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# H0: 66.6 +2.9 -2.8 km/s/Mpc
# Ωm: 0.330 +0.015 -0.014

# ln(fp_cc): -0.51 +0.17 -0.17
# n_cc: 1.33 +0.53 -0.50
# ΔM: -0.101 +0.089 -0.093 mag
# Chi squared (MAP): 1669.81
# Log evidence: -981.0
# DOF: 1748
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.10563 positive z > 0.10563 negative
# z_cosmo = z_cmb +- offset

# H0: 67.7 +3.0 -3.0 km/s/Mpc
# Ωm: 0.311 +0.017 -0.016
# 1000 Δz: 0.47 +0.23 -0.23 (prior ~ U[-1.5, 1.5])

# ln(fp_cc): -0.50 +0.18 -0.17
# n_cc: 1.34 +0.53 -0.51
# ΔM: -0.080 +0.091 -0.095 mag
# Chi squared (MAP): 1665.64
# Log evidence: -980.5
# DOF: 1747
# ---------------------------------


# ----------- Flat wCDM -----------
# H0: 66.9 +3.0 -2.9 km/s/Mpc
# Ωm: 0.306 +0.040 -0.046
# w0: -0.93 +0.11 -0.11 (prior ~ U[-1.5, 0.0])

# ln(fp_cc): -0.50 +0.18 -0.17
# n_cc: 1.36 +0.54 -0.52
# ΔM: -0.087 +0.094 -0.096 mag
# Chi squared (MAP): 1667.32
# Log evidence: -982.4
# DOF: 1747
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa <= -1 / 3 enforced in the likelihood
#
# H0: 65.5 +2.9 -2.9 km/s/Mpc
# Ωm: 0.390 +0.027 -0.042
# w0: -0.86 +0.10 -0.11 (prior ~ U[-3.0, 1.0])
# wa: < -2.123 (prior ~ U[-3.0, 2.0], posterior truncated)

# ln(fp_cc): -0.51 +0.17 -0.17
# n_cc: 1.36 +0.52 -0.50
# ΔM: -0.121 +0.093 -0.098 mag
# Chi squared (MAP): 1661.04
# Log evidence: -1083.1 (inaccurate due to truncated posterior)
# DOF: 1746
# ---------------------------------
