from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as c0
import y2018fs8.data as fs8_data

c = c0 / 1000  # km/s

data = fs8_data.data
inv_cov_mat = np.linalg.inv(fs8_data.cov_mat)

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def rho_de_z(z, w0):
    cubic = (1 + z) ** 3
    return (2 * cubic / (1 + w0 + (1 - w0) * cubic)) ** 2


@njit
def E(z, om, w0=-1):
    return np.sqrt(om * (1 + z) ** 3 + (1 - om) * rho_de_z(z, w0))


@njit
def dE_da(z, om, w0=-1):
    a = 1 / (1 + z)
    dz = 1e-05
    Ez_plus = E(z + dz, om, w0)
    Ez_minus = E(z - dz, om, w0)
    dE_dz = (Ez_plus - Ez_minus) / (2 * dz)
    return -dE_dz / a**2


@njit
def DM(z, om, w0=-1):
    dh_grid = c / E(z_grid, om, w0)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid))
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


denominator_fiducial = E(data["z"], data["omega_fid"]) * np.array(
    [DM(zi, om_fid) for zi, om_fid in zip(data["z"], data["omega_fid"])]
)


@njit
def growth_ode(a, y, om, w0):
    z = 1 / a - 1
    E_val = E(z, om, w0)
    dE_da_val = dE_da(z, om, w0)

    delta, d_delta_da = y

    source = (3 / 2) * (om / a**5) * (delta / E_val**2)
    friction = -(3 / a + dE_da_val / E_val) * d_delta_da
    d2_delta_da = friction + source

    return [d_delta_da, d2_delta_da]


a_vals = np.logspace(-2.3, 0, 1000)


def compute_fs8(zs, om, sig8, w0):
    sol = solve_ivp(
        growth_ode,
        t_span=(a_vals[0], a_vals[-1]),
        y0=(a_vals[0], 1.0),
        t_eval=a_vals,
        rtol=1e-8,
        atol=1e-10,
        args=(om, w0),
    )
    delta, d_delta_da = sol.y

    delta0 = np.interp(1.0, a_vals, delta)
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    a_z = 1 / (1 + zs)
    delta_vals = np.interp(a_z, a_vals, delta)
    f_vals = a_z * np.interp(a_z, a_vals, d_delta_da) / delta_vals
    sigma8_zs = sig8 * delta_vals / delta0

    return f_vals * sigma8_zs


def chi_squared(theta):
    Om, sig8, w0, f_err = theta
    q = E(data["z"], Om, w0) * DM(data["z"], Om, w0) / denominator_fiducial
    delta = data["fs8"] - compute_fs8(data["z"], Om, sig8, w0) / q
    return f_err**2 * np.dot(delta, np.dot(inv_cov_mat, delta))


N = len(data["z"])


def log_likelihood(theta):
    return -0.5 * chi_squared(theta) + N * np.log(theta[-1])


bounds = np.array(
    [
        (0.1, 0.6),  # Om
        (0.2, 1.2),  # sigma8
        (-1.0, 0.0),  # w0
        (0.4, 2.4),  # f_err: overstimation factor of the errors
    ],
    dtype=np.float64,
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
    import matplotlib.pyplot as plt
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 250
    nsteps = 2500 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.70),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

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
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

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
    print(f"f = {f_50:.2f} +{f_84-f_50:.2f} -{f_50-f_16:.2f}")
    print(f"chi2 = {chi_squared(best_fit):.2f}")
    print(f"log evidence = {log_evd:.1f}")

    labels = ["$S_8$", "$Ω_m$", "$\sigma_8$", "$w_0$", "$f_{err}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)

    z_plot = np.linspace(0, np.max(data["z"]), 200)
    fs8_plot = compute_fs8(z_plot, *best_fit[0:-1])

    q_vals = (
        E(data["z"], Om_50, w0_50) * DM(data["z"], Om_50, w0_50) / denominator_fiducial
    )
    fs8_data_corrected = data["fs8"] * q_vals
    err_data_corrected = data["fs8_err"] * q_vals

    plt.errorbar(
        data["z"],
        fs8_data_corrected,
        yerr=err_data_corrected / f_50,
        fmt=".",
        label="data",
    )
    plt.plot(z_plot, fs8_plot, label="best-fit", color="C1")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


"""
flat ΛCDM

without f_err:
Ωm = 0.277 +0.027 -0.025
σ8 = 0.784 +0.019 -0.018
S8 = 0.754 +0.028 -0.027
w0: -1
f = 1
chi2 = 38.74
log evidence = -24.8
degrees of freedom = 63

---
with f_err:
Ωm = 0.276 +0.021 -0.020
σ8 = 0.785 +0.015 -0.014
S8 = 0.753 +0.021 -0.021
w0: -1
f = 1.29 +0.12 -0.11
chi2 = 64.31
log evidence = -23.3
degrees of freedom = 62

===============================

flat wCDM
Ωm = 0.250 +0.025 -0.027
σ8 = 0.892 +0.083 -0.058
S8 = 0.819 +0.040 -0.036
w0 = -0.698 +0.127 -0.128 (prior: U(-1.4, 0.0))
f = 1.33 +0.12 -0.12
chi2 = 63.38
log evidence = -22.1
degrees of freedom = 61

===============================

flat wzCDM
Ωm = 0.282 +0.020 -0.019
σ8 = 0.834 +0.031 -0.028
S8 = 0.810 +0.034 -0.033
w0 = -0.623 +0.150 -0.163 (prior: U(-1.0, 0.0))
f = 1.32 +0.12 -0.12
chi2 = 63.53
log evidence = -22.5
degrees of freedom = 61
"""
