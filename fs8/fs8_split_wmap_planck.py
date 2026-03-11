from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
import y2018fs8.data as fs8_data

c = c0 / 1000  # km/s

data = fs8_data.data
z_vals = data["z"]

inv_cov = np.linalg.inv(fs8_data.cov_mat)

z_max = np.max(z_vals) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
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
def Ez(z, params):
    Om, w0 = params[0], params[3]
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0))


@njit
def dE_da(z, params):
    Om, w0 = params[0], params[3]
    Ode = 1.0 - Om
    a = 1 / (1.0 + z)

    matter = Om * (1.0 + z) ** 3 * (3 * (1.0 + 0.0) / (1.0 + z))
    dark_engy = Ode * d_Ode_dz(z, w0)

    numerator = matter + dark_engy
    denominator = -2 * a**2 * Ez(z, params)
    return numerator / denominator


@njit
def DM(z, params):
    dh_grid = c / Ez(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid), dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


Ez_DMz_fiducial = np.zeros(N, dtype=np.float64)
for i in range(N):
    z_i = z_vals[i]
    Om_fid_i = data["omega_fid"][i]
    sig8_fid_i = data["s8_fid"][i]
    g8_fid_i = sig8_fid_i * (Om_fid_i / 0.3) ** 0.5
    w0_fid = -1.0
    f_err_fid = 1.0
    params_fid = [Om_fid_i, sig8_fid_i, g8_fid_i, w0_fid, f_err_fid]

    DM_i = DM(np.array([z_i]), params_fid)[0]
    Ez_i = Ez(z_i, params_fid)
    Ez_DMz_fiducial[i] = Ez_i * DM_i


@njit
def growth_ode(a, y, params):
    z = 1 / a - 1.0
    E_val = Ez(z, params)
    dE_da_val = dE_da(z, params)

    delta, d_delta_da = y

    Om = params[0]

    source = (3 / 2) * (Om / a**5) * (delta / E_val**2)
    friction = -(3 / a + dE_da_val / E_val) * d_delta_da
    d2_delta_da = friction + source

    return [d_delta_da, d2_delta_da]


max_z = 500
a_vals = np.logspace(np.log10(1 / (1.0 + max_z)), 0, 2000)


def fs8_theory(z, params):
    sol = solve_ivp(
        growth_ode,
        t_span=(a_vals[0], a_vals[-1]),
        y0=(a_vals[0], 1.0),
        t_eval=a_vals,
        rtol=1e-8,
        atol=1e-10,
        args=(params,),
    )
    delta, d_delta_da = sol.y

    a = 1 / (1.0 + z)
    sig8 = params[1]
    delta0 = interp_hermite(np.array([1.0]), a_vals, delta, d_delta_da)[0]
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    return sig8 * a * interp_pchip(a, a_vals, d_delta_da) / delta0


PLANCK_MASK = data["omega_fid"] >= 0.3
S8_fid = data["s8_fid"] * (data["omega_fid"] / 0.3) ** 0.5


def chi_squared(theta):
    g8, f_err = theta[2], theta[-1]

    # fiducial template WMAP vs Planck factor
    shape = np.where(PLANCK_MASK, 1.0, S8_fid / g8)

    # Alcock-Paczynski factor
    q = Ez(z_vals, theta) * DM(z_vals, theta) / Ez_DMz_fiducial

    diff = data["fs8"] - fs8_theory(z_vals, theta) * shape / q

    return diff @ (inv_cov * f_err**2) @ diff


def log_likelihood(theta):
    f_err = theta[-1]
    return -0.5 * (chi_squared(theta) - 2 * N * np.log(f_err))


bounds = np.array(
    [
        (0.1, 0.6),  # Ωm: effective clustering matter density
        (0.5, 1.0),  # sigma8
        (0.5, 1.5),  # g8
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


def log_probability(theta):
    lp = log_prior(theta)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def main():
    from multiprocessing import Pool
    import emcee
    from fs8.plot_predictions import plot_predictions
    from corner_plot import plot_corner_and_chains

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 1000
    nsteps = 2500 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.25),
        (emcee.moves.DEMove(), 0.75),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("mean acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    MAP_samples = samples[np.argmax(log_probs)]

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    [
        (Om_16, Om_50, Om_84),
        (s8_16, s8_50, s8_84),
        (g8_16, g8_50, g8_84),
        (w0_16, w0_50, w0_84),
        (f_16, f_50, f_84),
    ] = pct

    S8_samples = samples[:, 1] * (samples[:, 0] / 0.3) ** 0.5
    S8_chains_samples = chains_samples[:, :, 1] * (chains_samples[:, :, 0] / 0.3) ** 0.5

    S8_16, S8_50, S8_84 = np.percentile(S8_samples, [15.9, 50, 84.1])
    best_fit = np.percentile(samples, 50, axis=0)

    samples = np.column_stack((S8_samples, samples))
    chains_samples = np.concatenate(
        (S8_chains_samples[:, :, np.newaxis], chains_samples), axis=2
    )

    print(f"Ωm = {Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}")
    print(f"σ8 = {s8_50:.3f} +{s8_84-s8_50:.3f} -{s8_50-s8_16:.3f}")
    print(f"S8 = {S8_50:.3f} +{S8_84-S8_50:.3f} -{S8_50-S8_16:.3f}")
    print(f"g8 = {g8_50:.3f} +{g8_84-g8_50:.3f} -{g8_50-g8_16:.3f}")
    print(f"f_err = {f_50:.2f} +{f_84-f_50:.2f} -{f_50-f_16:.2f}")
    print(f"w0 = {w0_50:.3f} +{w0_84-w0_50:.3f} -{w0_50-w0_16:.3f}")
    print(f"chi2 = {chi_squared(MAP_samples):.2f}")
    print(f"log likelihood = {log_likelihood(MAP_samples):.1f}")
    print(f"degs of freedom = {N - len(best_fit)}")

    labels = ["$S_8$", "$Ω_m$", "$\sigma_8$", "$g_8$", "$w_0$", "$f_{err}$"]
    plot_corner_and_chains(labels, samples, chains_samples)
    plot_predictions(
        fs8_theory=lambda z: fs8_theory(z, best_fit),
        data=data,
        q=(Ez(z_vals, best_fit) * DM(z_vals, best_fit) / Ez_DMz_fiducial)
        / np.where(PLANCK_MASK, 1.0, S8_fid / g8_50),
        f_err=f_50,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
================================================

Planck (+Lensing):
Ωm = 0.3153 ±0.0073
σ8 = 0.8111 ±0.0060
S8 = 0.832 ±0.013

================================================

fs8 compilation with split WMAP and Planck data:

without f_err:
Ωm = 0.316 +0.033 -0.031
σ8 = 0.785 +0.020 -0.020
S8 = 0.805 +0.037 -0.036
g8 = 0.797 +0.033 -0.031
chi2 = 28.89
log likelihood = -14.4
degs of freedom = 59
chi2/dof = 0.49

-----

with f_err:
Ωm = 0.316 +0.022 -0.021
σ8 = 0.785 +0.014 -0.014
S8 = 0.805 +0.026 -0.025
g8 = 0.796 +0.023 -0.022
f_err = 1.43 +0.13 -0.13
chi2 = 61.41
log likelihood = -7.3
degs of freedom = 58
chi2/dof = 1.06
"""

"""
Flat wCDM

without f_err:
Ωm = 0.305 +0.038 -0.036
σ8 = 0.814 +0.078 -0.055
S8 = 0.827 +0.051 -0.049
g8 = 0.796 +0.033 -0.032
w0 = -0.882 +0.199 -0.231 (prior ~ U(-1, 0))
chi2 = 28.54
log likelihood = -14.3
degs of freedom = 58
chi2/dof = 0.49

-----

with f_err:
Ωm = 0.303 +0.027 -0.027
σ8 = 0.818 +0.056 -0.043
S8 = 0.826 +0.037 -0.035
g8 = 0.793 +0.023 -0.022
f_err = 1.43 +0.13 -0.13
w0 = -0.869 +0.148 -0.161 (prior ~ U(-2, 0))
chi2 = 62.60
log likelihood = -7.0
degs of freedom = 57
chi2/dof = 1.10
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)

without f_err:
Ωm = 0.317 +0.032 -0.031
σ8 = 0.813 +0.037 -0.029
S8 = 0.839 +0.044 -0.042
g8 = 0.790 +0.033 -0.031
w0 = -0.749 +0.206 -0.166 (prior ~ U(-1, 0))
chi2 = 28.62
log likelihood = -14.3
degs of freedom = 58
chi2/dof = 0.49

-----

with f_err:
Ωm = 0.316 +0.022 -0.022
σ8 = 0.808 +0.026 -0.021
S8 = 0.831 +0.032 -0.030
g8 = 0.791 +0.023 -0.022
f_err = 1.43 +0.13 -0.13
w0 = -0.790 +0.160 -0.135 (prior ~ U(-1, 0))
chi2 = 61.46
log likelihood = -7.1
degs of freedom = 57
chi2/dof = 1.08
"""
