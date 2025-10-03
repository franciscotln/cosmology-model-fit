from numba import njit
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from multiprocessing import Pool
from .plot_predictions import plot_cc_predictions
from y2005cc.data import get_data

c = 299792.458  # Speed of light in km/s

legend, z_values, H_values, cov_matrix = get_data()
cho = cho_factor(cov_matrix)
logdet = np.linalg.slogdet(cov_matrix)[1]


@njit
def H_z(z, h0, Om, exp_w0):
    cubed = (1 + z) ** 3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + np.log(exp_w0)))
    return h0 * np.sqrt(Om * cubed + (1 - Om) * rho_de)


bounds = np.array(
    [
        (30, 120),  # H0
        (0, 0.7),  # Om
        (0.0001, 1.2),  # exp(w0)
        (0.4, 3),  # f
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f = params[-1]
    delta = H_values - H_z(z_values, *params[0:3])
    return f**2 * np.dot(delta, cho_solve(cho, delta, check_finite=False))


def log_likelihood(params):
    N = len(z_values)
    normalization = N * np.log(2 * np.pi) + logdet - 2 * N * np.log(params[-1])
    return -0.5 * (chi_squared(params) + normalization)


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2500 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(5) as pool:
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
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", nwalkers * (nsteps - burn_in) * ndim / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    print("correlation matrix:")
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [exp_w0_16, exp_w0_50, exp_w0_84],
        [f_16, f_50, f_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    w0_samples = np.log(samples[:, 2])
    w0_16, w0_50, w0_84 = np.percentile(w0_samples, [15.9, 50, 84.1])

    best_fit = np.array([H0_50, Om_50, exp_w0_50, f_50], dtype=np.float64)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"exp(w0): {exp_w0_50:.3f} +{(exp_w0_84 - exp_w0_50):.3f} -{(exp_w0_50 - exp_w0_16):.3f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"f: {f_50:.3f} +{(f_84 - f_50):.3f} -{(f_50 - f_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Degs of freedom: {z_values.size  - len(best_fit)}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, *best_fit[0:3]),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix)) / f_50,
        label=f"{legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
    )
    labels = ["$H_0$", "$Ω_m$", "$exp(w0)$", "$f$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=2.0,
        smooth1d=2.0,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    plt.figure(figsize=(16, 1.5 * ndim))
    for n in range(ndim):
        plt.subplot2grid((ndim, 1), (n, 0))
        plt.plot(chains_samples[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""
*******************************
Results for data from
https://arxiv.org/pdf/2307.09501
and one data point from
https://arxiv.org/pdf/2506.03836
*******************************

Flat ΛCDM: w(z) = -1
With f:
H0: 67.1 +3.7 -3.8 km/s/Mpc
Ωm: 0.328 +0.052 -0.044
w0: -1 (fixed)
f: 1.456 +0.188 -0.179
Chi squared: 31.34
Log likelihood: -130.53
Degs of freedom: 30
Correlation matrix:
[[ 1.      -0.808    0.03132]
 [-0.808    1.      -0.05145]
 [ 0.03132 -0.05145  1.     ]]

Without f:
H0: 66.7 +5.4 -5.5 km/s/Mpc
Ωm: 0.334 +0.079 -0.062
w0: -1 (fixed)
f: 1
Chi squared: 14.82
Log likelihood: -134.65
Degs of freedom: 31
correlation matrix:
[[ 1.       -0.809697]
 [-0.809697  1.      ]]

Log likelihood ratio test:
-2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
-2 * (-134.65) + 2 * (-130.53) = 8.24
p-value = 0.0016
We are 99.84% confident that the model with f is better than the one without f.
So the uncertainties in the H(z) dataset are overestimated by a factor of 1.46 ± 0.18.
2.5 sigma between f=1 and f=1.46.

===============================

Flat wCDM: w(z) = w0
H0: 68.2 +6.4 -5.5 km/s/Mpc
Ωm: 0.312 +0.053 -0.057
exp(w0): 0.325 +0.183 -0.134
w0: -1.124 +0.446 -0.531
f: 1.441 +0.187 -0.180
Chi squared: 30.98
Log likelihood: -130.68
Degs of freedom: 29

===============================

Flat w(z) = -1 + 2 * (1 + w0) / ((1 + z)**3 + 1)
H0: 68.6 +7.2 -6.2 km/s/Mpc
Ωm: 0.317 +0.056 -0.048
exp(w0): 0.309 +0.208 -0.144
w0: -1.173 +0.514 -0.626
f: 1.443 +0.187 -0.181
Chi squared: 30.51
Log likelihood: -130.40
Degs of freedom: 29
"""
