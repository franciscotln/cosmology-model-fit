from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2022pantheonSHOES.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data


cc_legend, z_cc_vals, H_cc_vals, H_err, cov_mat_sys_cc = get_cc_data(split_sys=True)
legend, z_cmb, z_hel, mB_vals, cov_matrix_sn = get_sn_data()

L_sn = cho_factor(cov_matrix_sn, lower=True)[0]

N_cc = len(z_cc_vals)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]

c = c0 / 1000  # Speed of light in km/s


@njit
def H_z(z, params):
    H0, Om = params[2], params[3]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


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

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    y_cc = solve_triangular(L_cc, delta_cc)

    return np.dot(y_sn, y_sn) + np.dot(y_cc, y_cc)


bounds = np.array(
    [
        (np.log(0.3), np.log(1.2)),  # ln(fp_cc)
        (-4.0, 4.0),  # n_cc
        (55, 80),  # H0
        (0.15, 0.70),  # Ωm
        (-20, -19),  # M
    ],
)

prior_normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return prior_normalization


z_pivot = 0.6142


@njit
def log_likelihood(params):
    fp_cc, n_cc = np.exp(params[0]), params[1]
    fz_cc = fp_cc * ((1.0 + z_cc_vals) / (1.0 + z_pivot))**n_cc
    cov_mat_cc = cov_mat_sys_cc + np.diag(H_err**2 * fz_cc**2)
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
    from corner_plot import plot_corner_and_chains
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from ohd.plot_predictions import plot_cc_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.2),
        (emcee.moves.DEMove(), 0.8),
    ]

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

    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)

    [
        (f_cc_16, f_cc_50, f_cc_84),
        (n_cc_16, n_cc_50, n_cc_84),
        (h0_16, h0_50, h0_84),
        (Om_16, Om_50, Om_84),
        (M_16, M_50, M_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = samples[np.argmax(log_probs)]
    DOF = len(z_cmb) + len(z_cc_vals) - len(best_fit)

    Omh2_samples = samples[:, 3] * (samples[:, 2] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    fz_cc = np.exp(best_fit[0]) * ((1.0 + z_cc_vals) / (1.0 + z_pivot))**best_fit[1]
    cov_mat_cc = cov_mat_sys_cc + np.diag(H_err**2 * fz_cc**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)

    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"ln(f_cc): {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"n_cc: {n_cc_50:.2f} +{(n_cc_84 - n_cc_50):.2f} -{(n_cc_50 - n_cc_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit, L_cc):.2f}")
    print(f"DOF: {DOF}")

    labels=["$ln(fp_{CC})$", "$n_{CC}$", "$H_0$", "$Ω_m$", "M"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=H_err,
        label=f"{cc_legend}: $H_0$={h0_50:.1f} km/s/Mpc",
        err_scaling=1 / fz_cc,
    )
    plot_sn_predictions(
        legend=legend,
        x=z_cmb,
        y=mB_vals - M_50,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mB_theory(best_fit) - M_50,
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# Flat ΛCDM: w(z) = -1
# H0: 66.2 +2.9 -2.9 km/s/Mpc
# M: -19.472 +0.092 -0.094 mag
# Ωm: 0.328 +0.017 -0.017
# ωm: 0.144 +0.013 -0.012
# ln(f_cc): -0.50 +0.12 -0.11
# n_cc: 1.33 +0.48 -0.46
# Chi squared: 1441.04
# DOF: 1624
# ---------------------------------

# Flat wCDM: w(z) = w0
# H0: 66.4 +3.0 -2.9 km/s/Mpc
# M: -19.464 +0.095 -0.097 mag
# Ωm: 0.309 +0.043 -0.048
# ωm: 0.1357 +0.0195 -0.0209
# w0: -0.948 +0.110 -0.117
# ln(f_cc): -0.49 +0.12 -0.11
# n_cc: 1.37 +0.48 -0.47
# Chi squared: 1440.40
# DOF: 1623
# ---------------------------------
