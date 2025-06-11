import numpy as np
from scipy.linalg import cho_factor, cho_solve
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
from .plot_predictions import plot_predictions
from y2005cc.data import get_data

# Speed of light in km/s
c = 299792.458

legend, z_values, H_values, cov_matrix = get_data()
cho = cho_factor(cov_matrix)


def H_z(z, h0, Om, w0=-1):
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return h0 * np.sqrt(Om * one_plus_z**3 + (1 - Om) * evolving_de)


bounds = np.array([(40, 110), (0.1, 0.6), (-3.5, 0.5)])  # H0, Om, w0


def chi_squared(params):
    delta = H_values - H_z(z_values, *params)
    return np.dot(delta, cho_solve(cho, delta))


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


L, _ = cho
log_det = 2 * np.sum(np.log(np.diag(L)))
pi_const = len(H_values) * np.log(2 * np.pi)


def log_likelihood(params):
    return -0.5 * (chi_squared(params) + log_det + pi_const)


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

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [H0_50, Om_50, w0_50]

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Degs of freedom: {z_values.size  - len(best_fit)}")

    plot_predictions(
        H_z=lambda z: H_z(z, *best_fit),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix)),
        label=f"{legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
    )
    corner.corner(
        samples,
        labels=["$H_0$", "$\Omega_m$", "$w_0$"],
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
******************************
Results for data from
https://arxiv.org/pdf/2307.09501
*****************************

Flat ΛCDM
H0: 66.3 +5.4 -5.4 km/s/Mpc
Ωm: 0.338 +0.079 -0.063
w0: -1
wa: 0
Chi squared: 14.59
Log likelihood: -129.99
Degs of freedom: 30

===============================

Flat wCDM
H0: 71.4 +11.1 -8.6 km/s/Mpc
Ωm: 0.306 +0.080 -0.068
w0: -1.599 +0.734 -0.920
wa: 0
Chi squared: 14.90
Log likelihood: -130.15
Degs of freedom: 29

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 72.0 +11.3 -9.3 km/s/Mpc
Ωm: 0.304 +0.081 -0.065
w0: -1.683 +0.821 -0.965
wa: 0
Chi squared: 14.47
Log likelihood: -129.94
Degs of freedom: 29

=============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 72.6 +11.6 -9.6
Ωm: 0.310 +0.102 -0.082
w0: -1.705 +0.901 -0.978
wa: -0.583 +2.936 - 2.997 (unconstrained)
Chi squared: 14.88
Log likelihood: -130.14
Degs of freedom: 28
"""
