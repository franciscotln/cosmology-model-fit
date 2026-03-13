from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
import y2018fs8.data as fs8_data

c = c0 / 1000  # km/s

data = fs8_data.data
z_vals = data["z"]
a_vals = 1 / (1.0 + z_vals)
fs8_vals = data["fs8"]
inv_cov_mat = np.linalg.inv(fs8_data.cov_mat)

z_max = np.max(data["z"]) + 0.1
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
def Ez(z, Om, w0):
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0))


@njit
def dE_da(z, Om, w0):
    a = 1 / (1.0 + z)
    numerator = 3 * Om * (1.0 + z) ** 2 + (1.0 - Om) * d_Ode_dz(z, w0)
    denominator = 2 * a**2 * Ez(z, Om, w0)
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
    dE_da_val = dE_da(z, Om, w0)

    delta, d_delta_da = integr

    source = (3 / 2) * (Om / a**5) * (delta / E_val**2)
    friction = -(3 / a + dE_da_val / E_val) * d_delta_da
    d2_delta_da = friction + source

    return [d_delta_da, d2_delta_da]


a_span = np.logspace(-2.15, 0, 1000)
a_init = a_span[0]
a_end = a_span[-1]


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


def chi_squared(theta):
    Om, sig8, w0, f_err = theta
    q = AP_factor(z_vals, Om, w0)
    delta = fs8_vals - fs8_theory(a_vals, Om, sig8, w0) / q
    return delta @ (inv_cov_mat * f_err**2) @ delta


def log_likelihood(theta):
    f_err = theta[-1]
    return -0.5 * (chi_squared(theta) - 2 * N * np.log(f_err))


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
    burn_in = 500
    nsteps = 2000 + burn_in
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
    print(f"w0 = {w0_50:.3f} +{w0_84-w0_50:.3f} -{w0_50-w0_16:.3f}")
    print(f"f_err = {f_50:.2f} +{f_84-f_50:.2f} -{f_50-f_16:.2f}")
    print(f"chi2 = {chi_squared(MAP_samples):.2f}")
    print(f"log likelihood = {log_likelihood(MAP_samples):.1f}")
    print(f"degs of freedom = {N - len(best_fit)}")

    labels = ["$S_8$", "$Ω_m$", "$\sigma_8$", "$w_0$", "$f_{err}$"]
    plot_corner_and_chains(labels, samples, chains_samples)
    plot_predictions(
        fs8_theory=lambda z: fs8_theory(1 / (1 + z), Om_50, s8_50, w0_50),
        data=data,
        q=Ez(z_vals, Om_50, w0_50) * DM(z_vals, Om_50, w0_50) / Ez_DMz_fid,
        f_err=f_50,
    )


if __name__ == "__main__":
    main()


"""
flat ΛCDM

without f_err:
Ωm = 0.297 +0.033 -0.030
σ8 = 0.783 +0.021 -0.020
S8 = 0.779 +0.032 -0.031
chi2 = 21.54
log likelihood = -10.8
degs of freedom = 57

---

with f_err:
Ωm = 0.296 +0.020 -0.019
σ8 = 0.784 +0.012 -0.012
S8 = 0.778 +0.019 -0.019
f_err = 1.63 +0.15 -0.15
chi2 = 59.04
log likelihood = 0.2
degs of freedom = 56
"""

"""
flat wCDM

without f_err:
Ωm = 0.276 +0.036 -0.032
σ8 = 0.858 +0.079 -0.067
S8 = 0.827 +0.046 -0.048
w0 = -0.762 +0.149 -0.190 (prior: U(-1.4, 0.0))
chi2 = 19.79
log likelihood = -9.9
degs of freedom = 56

---

with f_err:
Ωm = 0.271 +0.023 -0.023
σ8 = 0.874 +0.059 -0.049
S8 = 0.832 +0.032 -0.031
w0 = -0.729 +0.104 -0.118 (prior: U(-1.4, 0.0))
f_err = 1.69 +0.16 -0.16
chi2 = 59.16
log likelihood = 2.7
degs of freedom = 55
"""

"""
flat wzCDM

without f_err:
Ωm = 0.300 +0.033 -0.030
σ8 = 0.835 +0.050 -0.038
S8 = 0.838 +0.049 -0.045
w0 = -0.607 +0.213 -0.220 (prior: U(-1.0, 0.0))
chi2 = 19.67
log likelihood = -9.8
degs of freedom = 56
---

with f_err:
Ωm = 0.299 +0.019 -0.018
σ8 = 0.835 +0.030 -0.027
S8 = 0.835 +0.031 -0.030
w0 = -0.613 +0.138 -0.152 (prior: U(-1.0, 0.0))
f_err = 1.69 +0.16 -0.16
chi2 = 58.53
log likelihood = 2.9
degs of freedom = 55
"""
