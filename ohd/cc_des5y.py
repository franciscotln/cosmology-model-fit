from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from scipy.constants import c as c0
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2025DESdovekie.data import get_data, effective_sample_size
from y2005cc.data_no_loubser import get_data as get_cc_data

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


params_names = ["ln_fp", "n_cc", "dM", "H0", "om", "dz_1000"]
labels = ["ln(f_{p_cc})", "n_{cc}", "Δ_M", "H_0", "Ω_m", "1000 Δz"]
bounds = np.array(
    [
        (-1.4, 0.4),  # ln(fp_cc)
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


@njit
def get_fz(params):
    z_pivot = 0.728 # corr(ln(fp), n) = -8.53e-04
    f_piv, n = np.exp(params[0]), params[1]
    return f_piv * ((1.0 + z_cc_vals) / (1.0 + z_pivot))**n


@njit
def log_likelihood(params):
    cov_mat_cc = cov_mat_sys_cc + np.diag(diag_stat_cc**2 * get_fz(params)**2) 
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2 * np.sum(np.log(np.diag(L_cc)))

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc
    return -0.5 * (chi_squared(params, L_cc) + normalization_cc)


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
    from getdist import MCSamples, plots
    import matplotlib.pyplot as plt
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

    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    flat_log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=False)

    # reshape for getdist
    chain_list = np.moveaxis(samples, 1, 0)
    loglike_list = np.moveaxis(log_probs, 1, 0)
    gd_samples = MCSamples(
        samples=chain_list,
        loglikes=-loglike_list,
        names=params_names,
        labels=labels,
        label='DES5Y + CC'
    )

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = flat_samples[np.argmax(flat_log_probs)]
    DOF = effective_sample_size + N_cc - len(best_fit)
    print(f"log likelihood (MAP): {log_likelihood(best_fit):.2f}")
    print(f"DOF: {DOF}")

    plots.getSubplotPlotter().triangle_plot(
        roots=gd_samples,
        filled=True,
        title_limit=1,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(diag_stat_cc**2 + np.diag(cov_mat_sys_cc)),
        label=f"{cc_legend}: $H_0$={best_fit[3]:.1f} km/s/Mpc",
        err_scaling=1 / get_fz(best_fit),
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
# H0 = 67.3 ± 2.9 km/s/Mpc
# Ωm = 0.329 ± 0.014
#
# ln(fp_cc) = -0.52 ± 0.18
# n_cc = 1.52 +0.52 -0.57
# ΔM = -0.081 ± 0.091 mag
#
# log likelihood (MAP): -955.89
# DOF: 1745
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.10563 positive z > 0.10563 negative
# z_cosmo = z_cmb +- offset
#
# H0 = 68.5 ± 3.0 km/s/Mpc
# Ωm = 0.309 ± 0.017
# 1000 Δz = 0.49 ± 0.23 (prior ~ U[-1.5, 1.5])
#
# ln(fp_cc) = -0.51 ± 0.18
# n_cc = 1.53 +0.51 -0.57
# ΔM = -0.057 ± 0.091 mag
#
# log likelihood (MAP): -953.63
# DOF: 1744
# ---------------------------------


# ----------- Flat wCDM -----------
# H0 = 67.8 ± 3.0 km/s/Mpc
# Ωm = 0.288 +0.053 -0.039
# w0 = -0.90 +0.12 -0.10 (prior U[-1.5, 0])
#
# ln(fp_cc) = -0.51 ± 0.18
# n_cc = 1.55 +0.52 -0.58
# ΔM = -0.057 ± 0.096 mag
#
# log likelihood (MAP): -955.51
# DOF: 1744
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < -1 / 3 enforced in the likelihood
#
# H0 = 64.2 ± 3.1 km/s/Mpc
# Ωm = 0.444 +0.045 -0.027
# w0 = -0.56 +0.20 -0.26 (prior ~ U[-2, 0])
# wa = -6 +3 -2 (prior ~ U[-15, 2])
#
# ln(fp_cc) = -0.50 ± 0.18
# n_cc = 1.62 ± 0.55
# Δ_M = -0.15 ± 0.10 mag
#
# log likelihood (MAP): -952.71
# DOF: 1743
# ---------------------------------
