from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
import y2018fs8.data as fs8_data

c = c0 / 1000  # km/s
H0 = 70.0  # km/s/Mpc

"""
Mask applied on data which refers to the data based on Planck cosmology:
Ωm_fid >= 0.3
"""

data = fs8_data.data
mask = data["omega_fid"] >= 0.3

data = data[mask]
cov_mat = fs8_data.cov_mat[np.ix_(mask, mask)]

inv_cov = np.linalg.inv(cov_mat)
z_vals = data["z"]
fs8_vals = data["fs8"]

z_grid = np.linspace(0, np.max(z_vals) + 0.1, num=4000)
dz = np.diff(z_grid)

N = len(z_vals)


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
def Hz(z, H0, Om, w0):
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0))


@njit
def dH_da(z, H0, Om, w0):
    a = 1 / (1.0 + z)
    numerator = 3 * Om * (1.0 + z) ** 2 + (1.0 - Om) * d_Ode_dz(z, w0)
    denominator = 2 * a**2 * Hz(z, H0, Om, w0)
    return -numerator * H0**2 / denominator


@njit
def DM(z, H0, Om, w0):
    dh_grid = c / Hz(z_grid, H0, Om, w0)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid), dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


Hz_DMz_fid = np.zeros(N, dtype=np.float64)
for i in range(N):
    w0_fid = -1.0
    Om_fid_i = data["omega_fid"][i]
    H0_fid_i = data["H0_fid"][i]
    z_i = data["z"][i]
    DM_i = DM(np.array([z_i]), H0_fid_i, Om_fid_i, w0_fid)[0]
    Hz_i = Hz(z_i, H0_fid_i, Om_fid_i, w0_fid)
    Hz_DMz_fid[i] = Hz_i * DM_i


@njit
def growth_ode(a, y, Om, w0):
    z = 1 / a - 1.0
    H_val = Hz(z, H0, Om, w0)
    dH_da_val = dH_da(z, H0, Om, w0)

    delta, d_delta_da = y

    source = (3 / 2) * H0**2 * (Om / a**5) * (delta / H_val**2)
    friction = -(3 / a + dH_da_val / H_val) * d_delta_da
    d2_delta_da = friction + source

    return [d_delta_da, d2_delta_da]


max_z = 500
a_vals = np.logspace(np.log10(1 / (1.0 + max_z)), 0, 2000)


def fs8_theory(z, Om, sig8, w0):
    sol = solve_ivp(
        growth_ode,
        t_span=(a_vals[0], a_vals[-1]),
        y0=(a_vals[0], 1.0),
        t_eval=a_vals,
        rtol=1e-8,
        atol=1e-10,
        args=(Om, w0),
    )
    delta, d_delta_da = sol.y

    delta0 = interp_hermite(np.array([1.0]), a_vals, delta, d_delta_da)[0]
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    a = 1 / (1.0 + z)
    return sig8 * a * interp_pchip(a, a_vals, d_delta_da) / delta0


def chi_squared(theta):
    Om, sig8, w0, f_err = theta
    q = Hz(z_vals, H0, Om, w0) * DM(z_vals, H0, Om, w0) / Hz_DMz_fid
    delta = fs8_vals - fs8_theory(z_vals, Om, sig8, w0) / q

    return delta @ (inv_cov * f_err**2) @ delta


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
    burn_in = 1000
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
    MAP_samples = samples[np.argmax(log_probs)]

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
        fs8_theory=lambda z: fs8_theory(z, Om_50, s8_50, w0_50),
        data=data,
        q=Hz(z_vals, H0, Om_50, w0_50) * DM(z_vals, H0, Om_50, w0_50) / Hz_DMz_fid,
        f_err=f_50,
    )


if __name__ == "__main__":
    main()


"""
Data points with Ωm_fid >= 0.3
Sample size: 36
==============================
"""

"""
flat ΛCDM

without f_err:
Ωm = 0.392 +0.049 -0.046
σ8 = 0.771 +0.020 -0.020
S8 = 0.881 +0.049 -0.048
chi2 = 13.39
log likelihood = -6.7
degs of freedom = 34
chi2/dof = 0.39

---

with f_err:
Ωm = 0.390 +0.030 -0.028
σ8 = 0.771 +0.013 -0.012
S8 = 0.879 +0.031 -0.030
f_err = 1.60 +0.20 -0.19 (error overestimation factor)
chi2 = 35.84
log likelihood = -0.2
degs of freedom = 33
chi2/dof = 1.09
"""

"""
flat wCDM

without f_err:
Ωm = 0.372 +0.046 -0.047
σ8 = 0.754 +0.074 -0.043
S8 = 0.851 +0.062 -0.060
w0 = -1.114 +0.354 -0.415 (prior: U(-2.0, 0.0))
chi2 = 14.64
log likelihood = -7.3
degs of freedom = 35
chi2/dof = 0.42

---

with f_err:
Ωm = 0.373 +0.030 -0.031
σ8 = 0.756 +0.049 -0.033
S8 = 0.848 +0.042 -0.041
w0 = -1.098 +0.253 -0.282 (prior: U(-2.0, 0.0))
f_err = 1.55 +0.19 -0.18 (error overestimation factor)
chi2 = 37.96
log likelihood = -0.9
degs of freedom = 34
chi2/dof = 1.12
"""

"""
flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)

without f_err:
Ωm = 0.373 +0.043 -0.041
σ8 = 0.794 +0.035 -0.028
S8 = 0.887 +0.050 -0.047
w0 = -0.764 +0.244 -0.167 (prior: U(-1.0, 0.0))
chi2 = 14.70
log likelihood = -7.4
degs of freedom = 35
chi2/dof = 0.42
---

with f_err:
Ωm = 0.372 +0.028 -0.027
σ8 = 0.785 +0.022 -0.017
S8 = 0.875 +0.033 -0.031
w0 = -0.850 +0.175 -0.107 (prior: U(-1.0, 0.0))
f_err = 1.54 +0.19 -0.18 (error overestimation factor)
chi2 = 38.49
log likelihood = -1.0
degs of freedom = 34
chi2/dof = 1.13
"""
