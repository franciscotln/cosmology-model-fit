from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_ivp import solve_ivp
from solve_triangular import solve_triangular
import y2018fs8.data as fs8_data
from y2026union3_1.data import get_data

c = c0 / 1000  # km/s

legend, z_cmb, z_hel, mu_vals, cov_matrix = get_data()
L_cho = np.linalg.cholesky(cov_matrix)

data = fs8_data.data
z_vals = data["z"]
a_vals = 1 / (1.0 + z_vals)
fs8_vals = data["fs8"]
cho = cho_factor(fs8_data.cov_mat, lower=True)[0]

z_grid = np.linspace(0, np.max(np.concatenate([z_vals, z_cmb])) + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]

N_fs8 = len(data)


@njit
def w_de(z, w0, wa):
    return w0 + wa * z / (1.0 + z)


@njit
def Ode_z(z, w0, wa):
    # w0waCDM
    return (1. + z) ** (3 * (1. + w0 + wa)) * np.exp(-3 * wa * z / (1. + z))


@njit
def d_Ode_dz(z, w0, wa):
    return Ode_z(z, w0, wa) * 3 * (1.0 + w_de(z, w0, wa)) / (1.0 + z)


@njit
def Ez(z, Om, w0, wa):
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0, wa))


@njit
def dE_da(z, E_val, Om, w0, wa):
    a = 1 / (1.0 + z)
    numerator = 3 * Om * (1.0 + z) ** 2 + (1.0 - Om) * d_Ode_dz(z, w0, wa)
    denominator = 2 * a**2 * E_val
    return -numerator / denominator


@njit
def DM(z, Om, w0, wa):
    dh_grid = c / Ez(z_grid, Om, w0, wa)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid), dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def growth_ODE(a, integr, Om, w0, wa):
    z = 1 / a - 1.0
    E_val = Ez(z, Om, w0, wa)
    dE_da_val = dE_da(z, E_val, Om, w0, wa)

    delta, d_delta_da = integr

    source = (3 / 2) * (Om / a**5) * (delta / E_val**2)
    friction = -(3 / a + dE_da_val / E_val) * d_delta_da
    d2_delta_da = friction + source

    return np.array([d_delta_da, d2_delta_da])


a_span = np.logspace(-2.303, 0, 2000)
a_init = a_span[0]
a_end = a_span[-1]


@njit
def fs8_theory(a, Om, sigma8_0, w0, wa):
    sol = solve_ivp(
        growth_ODE,
        t_span=(a_init, a_end),
        y0=(a_init, 1.0),  # δ(a_init) = a_init, dδ/da(a_init) = 1.0
        t_eval=a_span,
        rtol=1e-6,
        atol=1e-8,
        args=(Om, w0, wa),
    )
    delta, d_delta_da = sol.y
    delta_0 = delta[-1]
    # f = d(ln δ)/d(ln a) = (a / δ(a)) * d(δ(a))/da
    # sigma8(a) = sigma8(a=1) * δ(a) / δ(a=1)
    return (sigma8_0 / delta_0) * a * interp_pchip(a, a_span, d_delta_da)


Ez_DMz_fid = np.empty(N_fs8, dtype=np.float64)
for i in range(N_fs8):
    z_i = z_vals[i]
    Om_fid_i = data["omega_fid"][i]
    DM_i, = DM(np.atleast_1d(z_i), Om=Om_fid_i, w0=-1.0, wa=0.0)
    Ez_i = Ez(z_i, Om=Om_fid_i, w0=-1.0, wa=0.0)
    Ez_DMz_fid[i] = Ez_i * DM_i


@njit
def AP_factor(z, Om, w0, wa):
    return Ez(z, Om, w0, wa) * DM(z, Om, w0, wa) / Ez_DMz_fid


@njit
def mu_theory(theta, DM_val):
    return theta[-1] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM_val)


@njit
def chi2_sn(theta):
    Om, _, w0, wa, _, _ = theta
    delta = mu_vals - mu_theory(theta, DM(z_cmb, Om, w0, wa))
    y = solve_triangular(L_cho, delta)
    return np.dot(y, y)


@njit
def chi2_fs8(theta):
    Om, sig8, w0, wa, ln_f_err, _ = theta
    q = AP_factor(z_vals, Om, w0, wa)
    delta = fs8_vals - fs8_theory(a_vals, Om, sig8, w0, wa) / q
    y = solve_triangular(cho, delta)
    return np.exp(-2 * ln_f_err) * np.dot(y, y)


@njit
def chi_squared(theta):
    return chi2_fs8(theta) + chi2_sn(theta)


@njit
def log_likelihood(theta):
    return -0.5 * (chi_squared(theta) + 2 * N_fs8 * theta[-2])


names = ["om", "s8", "w0", "wa", "ln_f_err", "M"]
labels = ["Ω_m", "\\sigma_8", "w_0", "w_a", "ln(f_{err})", "M"]
bounds = np.array([
    (0.1, 0.6),  # Ωm: effective clustering matter density
    (0.5, 1.0),  # sigma8
    (-2.0, 0.0),  # w0
    (-4.0, 2.0),  # wa
    (-1.0, 0.0),  # ln(f_err): overestimation factor of the errors
    (-9.3, -9.0),  # supernovae zero point offset
])

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    if theta[2] + theta[3] >= 0.0:
        return -np.inf
    return normalization


@njit
def log_probability_jit(theta):
    lp = log_prior(theta)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def log_probability(theta):
    return log_probability_jit(theta)


def main():
    from multiprocessing import Pool
    from emcee import EnsembleSampler, moves, autocorr
    from getdist import MCSamples, plots
    import matplotlib.pyplot as plt
    from fs8.plot_predictions import plot_predictions
    from sn.plotting import plot_predictions as plot_sn_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 500
    nsteps = 5000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    custom_moves = [(moves.KDEMove(), 0.20), (moves.DEMove(), 0.80)]

    with Pool(8) as pool:
        sampler = EnsembleSampler(nwalkers, ndim, log_probability, pool, custom_moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"})

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("mean acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    samples = sampler.get_chain(discard=burn_in, flat=False)
    flat_log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=False)

    # reshape for getdist
    chain_list = np.moveaxis(samples, 1, 0)
    loglike_list = np.moveaxis(log_probs, 1, 0)

    gd_samples = MCSamples(samples=chain_list, loglikes=-loglike_list, names=names, labels=labels)
    gd_samples.addDerived(gd_samples["s8"] * (gd_samples["om"] / 0.3) ** 0.5, name="S8", label="S_8")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    MAP_params = flat_samples[np.argmax(flat_log_probs)]

    print(f"chi2 (MAP) = {chi_squared(MAP_params):.2f}")
    print(f"log likelihood (MAP) = {log_likelihood(MAP_params):.1f}")
    print(f"DOF = {N_fs8 + len(z_cmb) - len(MAP_params)}")

    plots.get_subplot_plotter().triangle_plot(
        gd_samples,
        title_limit=1,
        filled=True,
        contour_colors=["C0"],
        color=["C0"],
        legend_labels=['$f\\sigma_8$ compilation']
    )
    plt.show()

    om, s8, w0, wa, ln_f_err, _ = MAP_params
    plot_predictions(
        fs8_theory=lambda z: fs8_theory(1 / (1 + z), om, s8, w0, wa),
        data=data,
        q=Ez(z_vals, om, w0, wa) * DM(z_vals, om, w0, wa) / Ez_DMz_fid,
        f_err=1 / np.exp(ln_f_err),
    )
    plot_sn_predictions(
        legend=legend,
        x=z_cmb,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=mu_theory(MAP_params, DM(z_cmb, om, w0, wa)),
        label=f"$Ω_m$={om:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ----------- flat ΛCDM -----------
# Ωm = 0.302 ± 0.015
# sig_8 = 0.774 ± 0.012
# S8 = 0.776 ± 0.016
# ln(f) = -0.398 +0.087 -0.100
# M = -9.225 ± 0.013 mag
#
# log likelihood (MAP) = -19.9
# chi2 (MAP) = 91.41
# DOF = 78
# ---------------------------------


# ----------- flat wCDM -----------
# Ωm = 0.261 ± 0.021
# sig_8 = 0.852 +0.030 -0.040
# S8 = 0.793 ± 0.017
# w0 = -0.812 +0.066 -0.059
# ln(f) = -0.432 +0.087 -0.098
# M = -9.190 ± 0.018 mag
#
# log likelihood (MAP) = -15.9
# chi2 (MAP) = 86.68
# DOF = 77
# ---------------------------------


# ---------- flat w0waCDM ---------
# Ωm = 0.283 +0.039 -0.034
# sig_8 = 0.822 +0.036 -0.067
# S8 = 0.793 ± 0.017
# w0 = -0.754 +0.092 -0.110
# wa = -0.61 +1.0 -0.60
# ln(f) = -0.423 +0.088 -0.100
# M = -9.184 ± 0.020 mag
#
# log likelihood (MAP) = -15.8
# chi2 (MAP) = 84.18
# DOF = 76
# ---------------------------------