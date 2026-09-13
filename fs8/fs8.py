from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_ivp import solve_ivp
from solve_triangular import solve_triangular
import y2018fs8.data as fs8_data

c = c0 / 1000  # km/s

data = fs8_data.data
z_vals = data["z"]
a_vals = 1 / (1.0 + z_vals)
fs8_vals = data["fs8"]
cho = cho_factor(fs8_data.cov_mat, lower=True)[0]

z_grid = np.linspace(0, np.max(z_vals) + 0.1, num=4000)
dz = np.diff(z_grid)

N = len(data)


@njit
def w_de(z, w0):
    # Thawing quintessence wzCDM
    return -1.0 + 2 * (1.0 + w0) / (1.0 + w0 + (1.0 - w0) * (1.0 + z) ** 3)


@njit
def Ode_z(z, w0):
    # Thawing quintessence wzCDM
    cubic = (1.0 + z) ** 3
    return (2 * cubic / (1.0 + w0 + (1.0 - w0) * cubic)) ** 2


@njit
def d_Ode_dz(z, w0):
    return Ode_z(z, w0) * 3 * (1.0 + w_de(z, w0)) / (1.0 + z)


@njit
def Ez(z, Om, w0):
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0))


@njit
def dE_da(z, E_val, Om, w0):
    a = 1 / (1.0 + z)
    numerator = 3 * Om * (1.0 + z) ** 2 + (1.0 - Om) * d_Ode_dz(z, w0)
    denominator = 2 * a**2 * E_val
    return -numerator / denominator


@njit
def DM(z, Om, w0):
    dh_grid = c / Ez(z_grid, Om, w0)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid), dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def growth_ODE(a, integr, Om, w0):
    z = 1 / a - 1.0
    E_val = Ez(z, Om, w0)
    dE_da_val = dE_da(z, E_val, Om, w0)

    delta, d_delta_da = integr

    source = (3 / 2) * (Om / a**5) * (delta / E_val**2)
    friction = -(3 / a + dE_da_val / E_val) * d_delta_da
    d2_delta_da = friction + source

    return np.array([d_delta_da, d2_delta_da])


a_span = np.logspace(-2.303, 0, 2000)
a_init = a_span[0]
a_end = a_span[-1]


@njit
def fs8_theory(a, Om, sigma8_0, w0):
    sol = solve_ivp(
        growth_ODE,
        t_span=(a_init, a_end),
        y0=(a_init, 1.0),  # δ(a_init) = a_init, dδ/da(a_init) = 1.0
        t_eval=a_span,
        rtol=1e-6,
        atol=1e-8,
        args=(Om, w0),
    )
    delta, d_delta_da = sol.y
    delta_0 = delta[-1]
    # f = d(ln δ)/d(ln a) = (a / δ(a)) * d(δ(a))/da
    # sigma8(a) = sigma8(a=1) * δ(a) / δ(a=1)
    return (sigma8_0 / delta_0) * a * interp_pchip(a, a_span, d_delta_da)


Ez_DMz_fid = np.zeros(N, dtype=np.float64)
for i in range(N):
    w0_fid = -1.0
    Om_fid_i = data["omega_fid"][i]
    z_i = z_vals[i]
    DM_i = DM(np.array([z_i]), Om_fid_i, w0_fid)[0]
    Ez_i = Ez(z_i, Om_fid_i, w0_fid)
    Ez_DMz_fid[i] = Ez_i * DM_i


@njit
def AP_factor(z, Om, w0):
    return Ez(z, Om, w0) * DM(z, Om, w0) / Ez_DMz_fid


@njit
def chi_squared(theta):
    Om, sig8, w0, f_err = theta
    q = AP_factor(z_vals, Om, w0)
    delta = fs8_vals - fs8_theory(a_vals, Om, sig8, w0) / q
    y = solve_triangular(cho, delta)
    return f_err**2 * np.dot(y, y)


@njit
def log_likelihood(theta):
    f_err = theta[-1]
    return -0.5 * (chi_squared(theta) - 2 * N * np.log(f_err))


names = ["om", "s8", "w0", "f_err"]
labels = ["Ω_m", "\\sigma_8", "w_0", "f_{err}"]
bounds = np.array(
    [
        (0.1, 0.6),  # Ωm: effective clustering matter density
        (0.5, 1.0),  # sigma8
        (-1.0, 0.0),  # w0
        (0.2, 3.2),  # f_err: overestimation factor of the errors
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
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

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 100
    nsteps = 4000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    custom_moves = [(moves.KDEMove(), 0.20), (moves.DEMove(), 0.80)]

    with Pool(8) as pool:
        sampler = EnsembleSampler(nwalkers, ndim, log_probability, pool, custom_moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

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

    gd_samples = MCSamples(
        samples=chain_list,
        loglikes=-loglike_list,
        names=names,
        labels=labels,
    )
    gd_samples.addDerived(gd_samples["s8"] * (gd_samples["om"] / 0.3) ** 0.5, name="S8", label="S_8")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    MAP_params = flat_samples[np.argmax(flat_log_probs)]

    print(f"chi2 (MAP) = {chi_squared(MAP_params):.2f}")
    print(f"log likelihood (MAP) = {log_likelihood(MAP_params):.1f}")
    print(f"DOF = {N - len(MAP_params)}")

    plots.get_subplot_plotter().triangle_plot(
        gd_samples,
        title_limit=1,
        filled=True,
        contour_colors=["C0"],
        color=["C0"],
        legend_labels=['$f\\sigma_8$ compilation']
    )
    plt.show()

    om, s8, w0, f_err = MAP_params
    plot_predictions(
        fs8_theory=lambda z: fs8_theory(1 / (1 + z), om, s8, w0),
        data=data,
        q=Ez(z_vals, om, w0) * DM(z_vals, om, w0) / Ez_DMz_fid,
        f_err=f_err,
    )


if __name__ == "__main__":
    main()


# ----------- flat ΛCDM -----------
# Ωm = 0.314 +- 0.020
# σ8 = 0.787 +- 0.011
# S8 = 0.804 +- 0.020
# f_err = 1.78 +- 0.17
# chi2 = 55.86
# log likelihood = 5.2
# degs of freedom = 53
# ---------------------------------


# ----------- flat wCDM -----------
# Ωm = 0.282 +0.019 -0.022
# σ8 = 0.897 +0.041 -0.052
# S8 = 0.868 +- 0.028
# w0 = -0.693 +0.100 -0.076 (prior ~ U[-1.5, 0])
# f_err = 1.93 +- 0.19
# chi2 = 56.04
# log likelihood = 10.1
# degs of freedom = 52
# ---------------------------------


# ---------- flat wzCDM -----------
# Ωm = 0.316 +- 0.018
# σ8 = 0.846 +0.022 -0.025
# S8 = 0.867 +- 0.028
# w0 = -0.58 +0.13 -0.10 (prior ~ U[-1, 0])
# f_err = 1.93 +- 0.19
# chi2 = 56.09
# log likelihood = 10.1
# degs of freedom = 52
# ---------------------------------
