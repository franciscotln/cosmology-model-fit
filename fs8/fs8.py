import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import cho_solve, cho_factor
from scipy.interpolate import interp1d
import y2018fs8.data as fs8_data
from numba import njit


covariance = fs8_data.cov_mat
data = fs8_data.data
z_data = data["z"]
fs8_data = data["fs8"]
err_data = data["fs8_err"]
Om_fid = data["omega_fid"]

cho_cov = cho_factor(covariance)


@njit
def E(z, om, w0):
    inv_a = 1 + z
    rho_de = (2 * inv_a**3 / (1 + inv_a**3)) ** (2 * (1 + w0))
    return np.sqrt(om * inv_a**3 + (1 - om) * rho_de)


def DM(z, om, w0):
    integrand = lambda zp: 1 / E(zp, om, w0)
    return quad(integrand, 0, z)[0]


def compute_q(z, om, w0, om_fid):
    return (E(z, om, w0) * DM(z, om, w0)) / (E(z, om_fid, -1) * DM(z, om_fid, -1))


def growth_deriv(y, a, om, w0):
    if a == 0:
        return [0, 0]
    z = 1 / a - 1
    H = E(z, om, w0)
    HH = H**2
    dHHda = -3 * om / a**4
    Hprime = (1 / 2) * dHHda / H
    ddelta = y[1]
    ddeltada = -(3 / a + Hprime / H) * y[1] + (3 / 2) * (om / a**5) / HH * y[0]
    return [ddelta, ddeltada]


a_vals = np.logspace(-3, 0, 1000)


def compute_fs8(zs, om, s8, w0):
    sol = solve_ivp(
        fun=lambda a, y: growth_deriv(y, a, om, w0),
        t_span=(a_vals[0], a_vals[-1]),
        y0=[a_vals[0], 1.0],
        t_eval=a_vals,
        rtol=1e-8,
        atol=1e-10,
    )
    delta = sol.y[0]
    ddelta = sol.y[1]

    delta_func = interp1d(a_vals, delta)
    ddelta_func = interp1d(a_vals, ddelta)
    fs8 = np.empty(zs.size, dtype=np.float64)
    for i, z in enumerate(zs):
        a_z = 1 / (1 + z)
        fs8[i] = s8 * a_z * ddelta_func(a_z) / delta_func(1.0)
    return fs8


def chi_squared(theta):
    Om, s8, w0, f_err = theta
    fs8_th = compute_fs8(z_data, Om, s8, w0)
    q = np.array([compute_q(zi, Om, w0, Omfi) for zi, Omfi in zip(z_data, Om_fid)])
    fs8_corr = fs8_data * q
    delta = fs8_corr - fs8_th
    return f_err**-2 * delta.dot(cho_solve(cho_cov, delta))


N = len(z_data)


def log_likelihood(theta):
    return -0.5 * chi_squared(theta) - N * np.log(theta[-1])


bounds = np.array(
    [
        [0.1, 0.6],  # Om
        [0.2, 1.2],  # sigma8
        [-2.5, 0.0],  # w0
        [0.1, 1.5],  # f_err: overstimation factor of the errors
    ],
    dtype=np.float64,
)


@njit
def log_prior(theta):
    if np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_probability(theta):
    lp = log_prior(theta)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def main():
    from multiprocessing import Pool
    import emcee, corner
    import matplotlib.pyplot as plt

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 40
    burn_in = 100
    nsteps = 900 + burn_in

    initial_pos = np.zeros((nwalkers, ndim))
    for dim, (low, high) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(low, high, nwalkers)

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.30),
                (emcee.moves.DEMove(), 0.56),
                (emcee.moves.DESnookerMove(), 0.14),
            ],
        )
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

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    Om_16, Om_50, Om_84 = pct[0]
    s8_16, s8_50, s8_84 = pct[1]
    w0_16, w0_50, w0_84 = pct[2]
    f_16, f_50, f_84 = pct[3]

    S8_samples = samples[:, 1] * (samples[:, 0] / 0.3) ** 0.5
    S8_16, S8_50, S8_84 = np.percentile(S8_samples, [15.9, 50, 84.1])
    best_fit = np.array([Om_50, s8_50, w0_50, f_50])

    print(f"Ωm = {Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}")
    print(f"σ8 = {s8_50:.3f} +{s8_84-s8_50:.3f} -{s8_50-s8_16:.3f}")
    print(f"S8 = {S8_50:.3f} +{S8_84-S8_50:.3f} -{S8_50-S8_16:.3f}")
    print(f"w0 = {w0_50:.3f} +{w0_84-w0_50:.3f} -{w0_50-w0_16:.3f}")
    print(f"f = {f_50:.2f} +{f_84-f_50:.2f} -{f_50-f_16:.2f}")
    print(f"chi2 = {chi_squared(best_fit):.2f}")

    labels = ["$Ω_m$", "$\sigma_8$", "$w_0$", "$f_{err}$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=2.0,
        smooth1d=2.0,
        bins=50,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    plt.figure(figsize=(16, 1.5 * ndim))
    for n in range(ndim):
        plt.subplot2grid((ndim, 1), (n, 0))
        plt.plot(chains_samples[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()

    z_plot = np.linspace(0, np.max(z_data), 200)
    fs8_plot = compute_fs8(z_plot, *best_fit[0:-1])

    q_vals = np.array(
        [compute_q(zi, Om_50, w0_50, Omfi) for zi, Omfi in zip(z_data, Om_fid)]
    )
    fs8_data_corrected = fs8_data * q_vals
    err_data_corrected = err_data * q_vals

    plt.errorbar(
        z_data,
        fs8_data_corrected,
        yerr=err_data_corrected * f_50,
        fmt=".",
        label="data",
    )
    plt.plot(z_plot, fs8_plot, label="best-fit model", color="C1")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


"""
flat ΛCDM
Ωm = 0.268 +0.020 -0.019
σ8 = 0.789 +0.015 -0.014
S8 = 0.746 +0.019 -0.021
w0 = -1
f = 0.78 +0.07 -0.06
chi2 = 62.21
63 degs of freedom

===============================

flat wCDM
Ωm = 0.284 +0.023 -0.022
σ8 = 0.861 +0.071 -0.056
S8 = 0.840 +0.084 -0.073
w0 = -0.796 +0.136 -0.146
f = 0.78 +0.07 -0.06
chi2 = 61.53
62 deg of freedom

===============================

flat wzCDM
Ωm = 0.299 +0.033 -0.032
σ8 = 0.830 +0.039 -0.037
S8 = 0.830 +0.074 -0.072
w0 = -0.713 +0.221 -0.232
f = 0.78 +0.07 -0.06
chi2 = 61.51
62 deg of freedom
"""
