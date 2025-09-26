from numba import njit
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
from .plot_predictions import plot_cc_predictions
from y2005cc.data import get_data

c = 299792.458  # Speed of light in km/s

legend, z_values, H_values, cov_matrix = get_data()
inv_cov = np.linalg.inv(cov_matrix)
logdet = np.linalg.slogdet(cov_matrix)[1]


@njit
def H_z(z, h0, Om, w0):
    cubed = (1 + z) ** 3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return h0 * np.sqrt(Om * cubed + (1 - Om) * rho_de)


bounds = np.array(
    [(50, 100), (0, 0.7), (-3.0, 0), (0.4, 3)], dtype=np.float64
)  # H0, Om, w0, f


def chi_squared(params):
    f = params[-1]
    delta = H_values - H_z(z_values, *params[0:3])
    return np.dot(delta, np.dot(inv_cov * f**2, delta))


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
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 20000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
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

    best_fit = [H0_50, Om_50, w0_50, f_50]

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
    corner.corner(
        samples,
        labels=["$H_0$", "$\Omega_m$", "$w_0$", "f"],
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
H0: 67.0 +3.7 -3.8 km/s/Mpc
Ωm: 0.329 +0.052 -0.044
w0: -1
f: 1.456 +0.186 -0.179
Chi squared: 31.34
Log likelihood: -130.44
Degs of freedom: 30
Correlation matrix:
[[ 1.      -0.80842  0.03265]
 [-0.80842  1.      -0.05184]
 [ 0.03265 -0.05184  1.     ]]

Without f:
H0: 66.6 +5.4 -5.5 km/s/Mpc
Ωm: 0.335 +0.080 -0.063
w0: -1
Chi squared: 14.81
Log likelihood: -134.57
Degs of freedom: 31
correlation matrix:
[[ 1.      -0.80645]
 [-0.80645  1.     ]]

Log likelihood ratio test:
-2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
-2 * (-134.57) + 2 * (-130.44) = 8.26
p-value = 0.0016
We are 99.84% confident that the model with f is better than the one without f.
So the uncertainties in the H(z) dataset are overestimated by a factor of 1.46 ± 0.18.
2.5 sigma between f=1 and f=1.46.

===============================

Flat wCDM
H0: 70.6 +7.3 -6.2 km/s/Mpc
Ωm: 0.313 +0.052 -0.050
w0: -1.409 +0.521 -0.629
f: 1.441 +0.188 -0.179
Chi squared: 30.83
Log likelihood: -130.52
Degs of freedom: 29
Correlation matrix:
[[ 1.      -0.39663 -0.76288 -0.00373]
 [-0.39663  1.      -0.01346  0.02089]
 [-0.76288 -0.01346  1.       0.03699]
 [-0.00373  0.02089  0.03699  1.     ]]

===============================

Flat w(z) = -1 + 2 * (1 + w0) / ((1 + z)**3 + 1)
H0: 71.9 +7.9 -6.8 km/s/Mpc
Ωm: 0.304 +0.055 -0.046
w0: -1.546 +0.612 -0.715
f: 1.449 +0.188 -0.180
Chi squared: 30.60
Log likelihood: -130.23
Degs of freedom: 29
Correlation matrix:
[[ 1.      -0.811   -0.78965  0.03278]
 [-0.811    1.       0.48468 -0.05246]
 [-0.78965  0.48468  1.      -0.00735]
 [ 0.03278 -0.05246 -0.00735  1.     ]]
"""
