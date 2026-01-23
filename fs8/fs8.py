from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
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
def Ez(z, Om_eff, w0=-1):
    return np.sqrt(Om_eff * (1 + z) ** 3 + (1 - Om_eff) * rho_de_z(z, w0))


@njit
def dE_da(z, Om_eff, w0=-1):
    a = 1 / (1 + z)
    dz = 1e-05
    Ez_plus = Ez(z + dz, Om_eff, w0)
    Ez_minus = Ez(z - dz, Om_eff, w0)
    dE_dz = (Ez_plus - Ez_minus) / (2 * dz)
    return -dE_dz / a**2


@njit
def DM(z, Om_eff, w0):
    dh_grid = c / Ez(z_grid, Om_eff, w0)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid), dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


denominator_fiducial = np.zeros(len(data["z"]), dtype=np.float64)
for i in range(len(data["z"])):
    z = data["z"][i]
    om_fid = data["omega_fid"][i]
    w0_fid = -1.0
    DM_i = DM(np.array([z]), om_fid, w0_fid)[0]
    denominator_fiducial[i] = Ez(z, om_fid, w0_fid) * DM_i


@njit
def growth_ode(a, y, Om_eff, w0):
    z = 1 / a - 1
    E_val = Ez(z, Om_eff, w0)
    dE_da_val = dE_da(z, Om_eff, w0)

    delta, d_delta_da = y

    source = (3 / 2) * (Om_eff / a**5) * (delta / E_val**2)
    friction = -(3 / a + dE_da_val / E_val) * d_delta_da
    d2_delta_da = friction + source

    return [d_delta_da, d2_delta_da]


max_z = 100
a_vals = np.logspace(np.log10(1 / (1 + max_z)), 0, 1000)


def fs8_theory(z, Om_eff, sig8, w0):
    sol = solve_ivp(
        growth_ode,
        t_span=(a_vals[0], a_vals[-1]),
        y0=(a_vals[0], 1.0),
        t_eval=a_vals,
        rtol=1e-8,
        atol=1e-10,
        args=(Om_eff, w0),
    )
    delta, d_delta_da = sol.y

    delta0 = interp_hermite(np.array([1.0]), a_vals, delta, d_delta_da)[0]
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    a = 1 / (1 + z)
    return sig8 * a * interp_pchip(a, a_vals, d_delta_da) / delta0


def chi_squared(theta):
    Om_eff, sig8, w0, f_err = theta
    q = Ez(data["z"], Om_eff, w0) * DM(data["z"], Om_eff, w0) / denominator_fiducial
    delta = data["fs8"] - fs8_theory(data["z"], Om_eff, sig8, w0) / q
    return f_err**2 * np.dot(delta, np.dot(inv_cov_mat, delta))


N = len(data["z"])
logdet = np.linalg.slogdet(fs8_data.cov_mat)[1]
norm_fact_1 = N * np.log(2 * np.pi) + logdet


def log_likelihood(theta):
    norm_fact = norm_fact_1 - 2 * N * np.log(theta[-1])
    return -0.5 * (chi_squared(theta) + norm_fact)


bounds = np.array(
    [
        (0.1, 0.6),  # Ωm_eff: effective clustering matter density
        (0.2, 1.2),  # sigma8
        (-1.0, 0.0),  # w0
        (0.5, 2.2),  # f_err: overestimation factor of the errors
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
    import matplotlib.pyplot as plt
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 1000
    nsteps = 2500 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.30),
        (emcee.moves.DEMove(), 0.70),
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

    print(f"Ωm_eff = {Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}")
    print(f"σ8 = {s8_50:.3f} +{s8_84-s8_50:.3f} -{s8_50-s8_16:.3f}")
    print(f"S8 = {S8_50:.3f} +{S8_84-S8_50:.3f} -{S8_50-S8_16:.3f}")
    print(f"w0 = {w0_50:.3f} +{w0_84-w0_50:.3f} -{w0_50-w0_16:.3f}")
    print(f"f = {f_50:.2f} +{f_84-f_50:.2f} -{f_50-f_16:.2f}")
    print(f"chi2 = {chi_squared(best_fit):.2f}")
    print(f"log likelihood = {log_likelihood(best_fit):.1f}")
    print(f"log evidence = {log_evd:.1f}")
    print(f"degs of freedom = {N - len(best_fit)}")

    labels = ["$S_8$", "$Ω_m$", "$\sigma_8$", "$w_0$", "$f_{err}$"]
    plot_corner_and_chains(labels, samples, chains_samples)

    z_plot = np.linspace(0, np.max(data["z"]), 200)
    fs8_plot = fs8_theory(z_plot, Om_50, s8_50, w0_50)

    q = Ez(data["z"], Om_50, w0_50) * DM(data["z"], Om_50, w0_50) / denominator_fiducial

    plt.errorbar(
        data["z"],
        data["fs8"] * q,
        yerr=data["fs8_err"] * q / f_50,
        fmt=".",
        label="data",
    )
    plt.plot(z_plot, fs8_plot, label="best-fit", color="C1")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.legend()
    plt.show()

    residuals = q * data["fs8"] - fs8_theory(data["z"], Om_50, s8_50, w0_50)
    plt.errorbar(
        data["z"],
        residuals,
        yerr=data["fs8_err"] * q / f_50,
        fmt=".",
        label="residuals",
    )
    plt.axhline(0, color="k", ls="--")
    plt.xlabel("z")
    plt.ylabel("residuals")
    plt.legend()
    plt.show()

    plt.hist(residuals, bins=12, density=True)
    plt.xlabel("residuals")
    plt.ylabel("density")
    plt.show()


if __name__ == "__main__":
    main()


"""
flat ΛCDM

without f_err:
Ωm_eff = 0.249 +0.026 -0.024
σ8 = 0.816 +0.020 -0.019
S8 = 0.742 +0.028 -0.028
w0: -1
f = 1
chi2 = 37.50
log likelihood = 83.2
log evidence = 77.8
degs of freedom = 61

---

with f_err:
Ωm_eff = 0.274 +0.020 -0.019
σ8 = 0.787 +0.014 -0.014
S8 = 0.752 +0.022 -0.021
w0: -1
f = 1.32 +0.12 -0.12
chi2 = 61.41
log likelihood = 101.5
log evidence = 93.7
degs of freedom = 60

===============================

flat wCDM
Ωm_eff = 0.254 +0.023 -0.024
σ8 = 0.875 +0.072 -0.051
S8 = 0.809 +0.038 -0.035
w0 = -0.742 +0.124 -0.123 (prior: U(-1.4, 0.0))
f = 1.35 +0.12 -0.12
chi2 = 60.35
log likelihood = 103.6
log evidence = 94.5
degs of freedom = 59

===============================

flat wzCDM
Ωm_eff = 0.280 +0.020 -0.019
σ8 = 0.828 +0.029 -0.026
S8 = 0.800 +0.033 -0.032
w0 = -0.686 +0.148 -0.155 (prior: U(-1.0, 0.0))
f = 1.34 +0.13 -0.12
chi2 = 60.52
log likelihood = 103.1
log evidence = 94.2
degs of freedom = 59
"""
