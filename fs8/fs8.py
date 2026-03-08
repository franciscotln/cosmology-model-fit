from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
import y2018fs8.data as fs8_data

c = c0 / 1000  # km/s

data = fs8_data.data

no_cov_mask = data["cov_id"] == -1
std_no_cov = data["fs8_err"][no_cov_mask]

cov1_mask = data["cov_id"] == 1
cov2_mask = data["cov_id"] == 2
cov3_mask = data["cov_id"] == 3

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


denominator_fiducial = np.zeros(N, dtype=np.float64)
for i in range(N):
    w0_fid = -1.0
    Om_fid_i = data["omega_fid"][i]
    z_i = data["z"][i]
    DM_i = DM(np.array([z_i]), Om_fid_i, w0_fid)[0]
    Ez_i = Ez(z_i, Om_fid_i, w0_fid)
    denominator_fiducial[i] = Ez_i * DM_i


@njit
def growth_ode(a, y, Om, w0):
    z = 1 / a - 1.0
    E_val = Ez(z, Om, w0)
    dE_da_val = dE_da(z, Om, w0)

    delta, d_delta_da = y

    source = (3 / 2) * (Om / a**5) * (delta / E_val**2)
    friction = -(3 / a + dE_da_val / E_val) * d_delta_da
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
    q = Ez(data["z"], Om, w0) * DM(data["z"], Om, w0) / denominator_fiducial
    delta = data["fs8"] - fs8_theory(data["z"], Om, sig8, w0) / q

    delta_no_cov = delta[no_cov_mask]
    delta_cov1 = delta[cov1_mask]
    delta_cov2 = delta[cov2_mask]
    delta_cov3 = delta[cov3_mask]

    chi2_no_cov = np.sum((delta_no_cov / (std_no_cov / f_err)) ** 2)
    chi2_1 = delta_cov1 @ (fs8_data.inv_cov1 * f_err**2) @ delta_cov1
    chi2_2 = delta_cov2 @ (fs8_data.inv_cov2 * f_err**2) @ delta_cov2
    chi2_3 = delta_cov3 @ (fs8_data.inv_cov3 * f_err**2) @ delta_cov3

    return chi2_no_cov + chi2_1 + chi2_2 + chi2_3


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
        fs8_theory=lambda z: fs8_theory(z, Om_50, s8_50, w0_50),
        data=data,
        q=Ez(data["z"], Om_50, w0_50)
        * DM(data["z"], Om_50, w0_50)
        / denominator_fiducial,
        f_err=f_50,
    )


if __name__ == "__main__":
    main()


"""
flat ΛCDM

without f_err:
Ωm = 0.302 +0.033 -0.031
σ8 = 0.783 +0.019 -0.018
S8 = 0.785 +0.033 -0.032 (1.33 sigma tension with Planck)
chi2 = 19.50
log likelihood = -9.8
degs of freedom = 56
chi2/deg = 0.348 (too low)

---

with f_err:
Ωm = 0.300 +0.019 -0.018
σ8 = 0.783 +0.011 -0.011
S8 = 0.784 +0.019 -0.019 (2 sigma tension with Planck)
f_err = 1.70 +0.16 -0.16
chi2 = 57.67
log likelihood = 2.6
degs of freedom = 55
chi2/deg = 1.05
"""

"""
flat wCDM

without f_err:
Ωm = 0.279 +0.036 -0.033
σ8 = 0.863 +0.077 -0.065
S8 = 0.835 +0.047 -0.047
w0 = -0.752 +0.140 -0.176 (prior: U(-1.4, 0.0))
chi2 = 17.46
log likelihood = -8.7
degs of freedom = 55
chi2/deg = 0.317

---

with f_err:
Ωm = 0.273 +0.022 -0.021
σ8 = 0.879 +0.055 -0.045
S8 = 0.840 +0.029 -0.029
w0 = -0.722 +0.096 -0.104 (prior: U(-1.4, 0.0))
f_err = 1.78 +0.17 -0.17
chi2 = 58.30
log likelihood = 5.8
degs of freedom = 54
chi2/deg = 1.08
"""

"""
flat wzCDM

without f_err:
Ωm = 0.303 +0.033 -0.030
σ8 = 0.836 +0.047 -0.038
S8 = 0.843 +0.047 -0.045
w0 = -0.614 +0.203 -0.211 (prior: U(-1.0, 0.0))
chi2 = 17.45
log likelihood = -8.7
degs of freedom = 55
chi2/deg = 0.317

---

with f_err:
Ωm = 0.302 +0.018 -0.017
σ8 = 0.836 +0.027 -0.025
S8 = 0.840 +0.029 -0.028
w0 = -0.616 +0.124 -0.140 (prior: U(-1.0, 0.0))
f_err = 1.78 +0.17 -0.16
chi2 = 58.10
log likelihood = 5.8
degs of freedom = 54
chi2/deg = 1.08
"""
