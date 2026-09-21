from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2022pantheonSHOES.data import get_data as get_sn_data
from y2005cc.data_no_loubser import get_data as get_cc_data


cc_legend, z_cc, H_cc, diag_stat_cc, cov_mat_sys_cc = get_cc_data(split_sys=True)
legend, z_cmb, z_hel, mB_vals, cov_matrix_sn = get_sn_data()

L_sn = cho_factor(cov_matrix_sn, lower=True)[0]

N_cc = len(z_cc)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]

c = c0 / 1000  # Speed of light in km/s


@njit
def H_z(z, params):
    H0, Om = params[2], params[3]
    w0 = params[5]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * (1.0 + z) ** (3 * (1 + w0)))


@njit
def DM_z(z, params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def mB_theory(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[4] + 25 + 5 * np.log10(dL)



@njit
def chi_squared(params, L_cc):
    delta_sn = mB_vals - mB_theory(params)
    y_sn = solve_triangular(L_sn, delta_sn)

    delta_cc = H_cc - H_z(z_cc, params)
    y_cc = solve_triangular(L_cc, delta_cc)

    return np.dot(y_sn, y_sn) + np.dot(y_cc, y_cc)


names=["ln_fp_cc", "n_cc", "H0", "om", "M", "w0"]
labels=["ln(fp_{cc})", "n_{cc}", "H_0", "Ω_m", "M", "w_0"]
bounds = np.array([
    (-2, 1),
    (-3.0, 7.0),
    (55.0, 80.0),
    (0.15, 0.70),
    (-20.0, -19.0),
    (-2.0, 0.0)
])

prior_normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return prior_normalization


@njit
def get_fz(params):
    z_pivot = 1.035
    f_piv, n = np.exp(params[0]), params[1]
    return f_piv * ((1.0 + z_cc) / (1.0 + z_pivot))**n


@njit
def log_likelihood(params):
    cov_mat_cc = cov_mat_sys_cc + np.diag(diag_stat_cc**2 * get_fz(params)**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2.0 * np.sum(np.log(np.diag(L_cc)))

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
    burn_in = 500
    nsteps = 2500 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [(emcee.moves.KDEMove(), 0.2), (emcee.moves.DEMove(), 0.8)]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"})

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    flat_log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=False)
    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    samples = sampler.get_chain(discard=burn_in, flat=False)

    chain_list = np.moveaxis(samples, 1, 0)
    loglike_list = np.moveaxis(log_probs, 1, 0)

    gd_samples = MCSamples(
        samples=chain_list,
        loglikes=-loglike_list,
        names=names,
        labels=labels,
        label='CCH + Pantheon+'
    )
    gd_samples.addDerived(
        paramVec=gd_samples["om"] * (gd_samples["H0"] / 100) ** 2,
        name="omh2",
        label="Ω_m h^2",
    )
    gd_samples.updateBaseStatistics()

    print("corr(ln(fp), n)", gd_samples.corr(["ln_fp_cc", "n_cc"])[0, 1])

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = flat_samples[np.argmax(flat_log_probs)]
    DOF = len(z_cmb) + len(z_cc) - len(best_fit)

    fz_cc = get_fz(best_fit)
    print(f"log likelihood (MAP): {log_likelihood(best_fit):.2f}")
    print(f"DOF: {DOF}")

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=names,
        title_limit=1,
        contour_colors=["C0"],
        filled=True,
    )
    plt.show()
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc,
        H=H_cc,
        H_err=np.sqrt(np.diag(cov_mat_sys_cc) + diag_stat_cc**2),
        label=f"{cc_legend}: $H_0$={best_fit[2]:.1f} km/s/Mpc",
        err_scaling=1 / fz_cc,
    )
    plot_sn_predictions(
        legend=legend,
        x=z_cmb,
        y=mB_vals - best_fit[4],
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mB_theory(best_fit) - best_fit[4],
        label=f"$Ω_m$={best_fit[3]:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# Flat ΛCDM: w(z) = -1
# H0 = 66.9 ± 2.8 km/s/Mpc
# Ωm = 0.330 ± 0.016
# Ωm h^2 = 0.148 ± 0.012
#
# M = -19.452 ± 0.088 mag
# ln(fp_cc) = -0.50 ± 0.27
# n_cc = 3.05 +0.86 -1.20
#
# log likelihood (MAP): -838.79
# DOF: 1621
# ---------------------------------

# Flat wCDM: w(z) = w
# H0 = 67.0 ± 2.8 km/s/Mpc
# Ωm = 0.310 +0.048 -0.037
# Ωm h^2 = 0.139 +0.021 -0.018
# w0 = -0.95 +0.11 -0.10 (prior ~ U[-2, 0])
#
# M = -19.446 ± 0.091 mag
# ln(fp_cc) = -0.46 ± 0.27
# n_cc = 3.07 +0.87 -1.2
#
# log likelihood (MAP): -838.68
# DOF: 1620
# ---------------------------------
