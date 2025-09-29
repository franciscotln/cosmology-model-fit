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
def H_z(z, h0, Om, w0):
    cubed = (1 + z) ** 3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return h0 * np.sqrt(Om * cubed + (1 - Om) * rho_de)


bounds = np.array(
    [(40, 120), (0, 0.7), (-4.0, 1), (0.4, 3)], dtype=np.float64
)  # H0, Om, w0, f


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
    nwalkers = 200 * ndim
    burn_in = 100
    nsteps = 1000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.5),
                (emcee.moves.DEMove(), 0.4),
                (emcee.moves.DESnookerMove(), 0.1),
            ],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    print("correlation matrix:")
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
        [f_16, f_50, f_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([H0_50, Om_50, w0_50, f_50], dtype=np.float64)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
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
    labels = ["$H_0$", "$Ω_m$", "$w_0$", "f"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    chains_samples = sampler.get_chain(discard=0, flat=False)
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    axes[ndim - 1].set_xlabel("chain step")
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

Flat ΛCDM
With f:
H0: 67.1 +3.7 -3.8 km/s/Mpc
Ωm: 0.328 +0.051 -0.043
w0: -1
f: 1.455 +0.187 -0.178
Chi squared: 31.32
Log likelihood: -130.53
Degs of freedom: 30
Correlation matrix:
[[ 1.          -8.07442e-01  2.91985e-02]
 [-8.07442e-01  1.          -4.79678e-02]
 [ 2.91985e-02 -4.79678e-02  1.         ]]

Without f:
H0: 66.7 +5.4 -5.5 km/s/Mpc
Ωm: 0.334 +0.079 -0.062
w0: -1
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

Flat wCDM
H0: 70.7 +7.5 -6.1 km/s/Mpc
Ωm: 0.312 +0.052 -0.050
w0: -1.401 +0.515 -0.644
f: 1.440 +0.188 -0.179
Chi squared: 30.78
Log likelihood: -130.61
Degs of freedom: 29
Correlation matrix:
[[ 1.      -0.46261 -0.79425 -0.04056]
 [-0.46261  1.       0.105    0.0491 ]
 [-0.79425  0.105    1.       0.07453]
 [-0.04056  0.0491   0.07453  1.     ]]

===============================

Flat w(z) = -1 + 2 * (1 + w0) / ((1 + z)**3 + 1)
H0: 72.3 +8.7 -7.0 km/s/Mpc
Ωm: 0.301 +0.056 -0.049
w0: -1.577 +0.627 -0.794
f: 1.444 +0.188 -0.180
Chi squared: 30.43
Log likelihood: -130.35
Degs of freedom: 29
Correlation matrix:
[[ 1.      -0.82775 -0.82181 -0.02437]
 [-0.82775  1.       0.54686 -0.00677]
 [-0.82181  0.54686  1.       0.05416]
 [-0.02437 -0.00677  0.05416  1.     ]]
"""
