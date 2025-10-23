from numba import njit
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from y2005cc.data import get_data

c = 299792.458  # Speed of light in km/s

legend, z_values, H_values, cov_matrix = get_data()
cho = cho_factor(cov_matrix)
logdet = np.linalg.slogdet(cov_matrix)[1]

# Planck prior
Omh2_planck = 0.1430
Omh2_planck_sigma = 0.0011


@njit
def H_z(z, h0, Om, w0):
    cubed = (1 + z) ** 3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return h0 * np.sqrt(Om * cubed + (1 - Om) * rho_de)


bounds = np.array(
    [
        (50, 90),  # H0
        (0.10, 0.60),  # Om
        (-2.5, 0.0),  # w0
        (0.4, 1.4),  # f
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f = params[-1]
    delta = H_values - H_z(z_values, *params[0:3])
    chi2_cc = f**-2 * np.dot(delta, cho_solve(cho, delta, check_finite=False))

    Omh2 = params[0] ** 2 * params[1] / 10000
    delta_Omh2 = Omh2_planck - Omh2
    chi2_planck = (delta_Omh2 / Omh2_planck_sigma) ** 2

    return chi2_cc + chi2_planck


def log_likelihood(params):
    N = len(z_values)
    normalization = N * np.log(2 * np.pi) + logdet + 2 * N * np.log(params[-1])
    return -0.5 * (chi_squared(params) + normalization)


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee, corner
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from log_evidence import log_evidence
    from .plot_predictions import plot_cc_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))

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
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

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
    print(f"Log evidence: {log_evidence(samples, log_probs, log_probability, bounds):.2f}")
    print(f"Degs of freedom: {1 + z_values.size - len(best_fit)}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, *best_fit[0:3]),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix)) * f_50,
        label=f"{legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
    )
    labels = ["$H_0$", "$Ω_m$", "$w_0$", "$f$"]
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
H0: 67.1 +3.9 -3.9 km/s/Mpc
Ωm: 0.328 +0.053 -0.045
w0: -1
f: 0.71 +0.10 -0.08
Chi squared: 29.31
Log likelihood: -130.62
Log evidence: -135.58 (Bayes factor with f: 1.76)
Degs of freedom: 31

Without f:
H0: 66.7 +5.4 -5.5
Ωm: 0.334 +0.078 -0.062
w0: -1
f: 1
Chi squared: 14.82
Log likelihood: -134.65
Log evidence: -137.34
Degs of freedom: 32

Log likelihood ratio test:
-2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
-2 * (-134.65) + 2 * (-130.62) = 8.06
Degrees of freedom = 1
p-value = 0.0044
We are 99.84% confident that the model with f is better than the one without f.
So the uncertainties in the H(z) dataset are overestimated and should be scaled down by 29%
3.00 - 3.75 sigma between f=1 and f=0.71.

===============================

Flat wCDM: w(z) = w0
H0: 67.4 +4.5 -4.3 km/s/Mpc
Ωm: 0.315 +0.045 -0.038
w0: -1.054 +0.187 -0.271
f: 0.71 +0.10 -0.08
Chi squared: 29.29
Log likelihood: -130.75
Degs of freedom: 30

===============================

Flat w(z) = -1 + 2 * (1 + w0) / ((1 + z)**3 + 1)
H0: 68.2 +5.7 -5.3 km/s/Mpc
Ωm: 0.308 +0.054 -0.046
w0: -1.137 +0.366 -0.457
f: 0.71 +0.10 -0.08
Chi squared: 29.39
Log likelihood: -130.69
Degs of freedom: 30
"""
