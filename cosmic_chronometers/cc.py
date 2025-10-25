from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2005cc.data import get_data

c = 299792.458  # Speed of light in km/s

legend, z_values, H_values, cov_matrix = get_data()

cho = cho_factor(cov_matrix, lower=True)[0]
logdet = np.linalg.slogdet(cov_matrix)[1]


@njit
def H_z(z, params):
    h0, Om = params[0], params[1]
    return h0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


bounds = np.array(
    [
        (40, 100),  # H0
        (0.0, 1.0),  # Om
        (0.3, 2.5),  # f
    ],
    dtype=np.float64,
)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    f = params[-1]

    delta = H_values - H_z(z_values, params)
    chi2_cc = f**2 * solve_triang(cho, delta)

    return chi2_cc


def log_likelihood(params):
    N = len(z_values)
    normalization = N * np.log(2 * np.pi) + logdet - 2 * N * np.log(params[-1])
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
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence
    from .plot_predictions import plot_cc_predictions

    ndim = len(bounds)
    nwalkers = 100
    burn_in = 500
    nsteps = 15000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", nwalkers * (nsteps - burn_in) * ndim / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (f_16, f_50, f_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"f: {f_50:.2f} +{(f_84 - f_50):.2f} -{(f_50 - f_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {1 + z_values.size - len(best_fit)}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix)) * f_50,
        label=f"{legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
    )
    plot_corner_and_chains(
        labels=["$H_0$", "$Ω_m$", "$f$"],
        flat_samples=samples,
        samples=chains_samples,
    )


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
With varying f:
H0: 67.1 +3.7 -3.8
Ωm: 0.328 +0.051 -0.044
f: 1.45 +0.19 -0.18
Chi squared: 31.29
Log likelihood: -130.53
Log evidence: -136.66
Degs of freedom: 31

-------------------------------

With fixed f = 1:
H0: 66.6 +5.4 -5.5
Ωm: 0.335 +0.079 -0.063
f: 1
Chi squared: 14.82
Log likelihood: -134.65
Log evidence: -138.44
Degs of freedom: 32

-------------------------------

Log likelihood ratio test:
-2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
-2 * (-134.65) + 2 * (-130.62) = 8.06
Degrees of freedom = 1
p-value = 0.0044
We are 99.84% confident that the model with f is better than the one without f.
So the uncertainties in the H(z) dataset are overestimated and should be scaled down by 29%
2.37 - 2.50 sigma between f=1 and f=1.45
"""
