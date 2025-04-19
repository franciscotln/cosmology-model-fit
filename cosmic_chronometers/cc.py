import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2005cc.data import get_data

# source: https://arxiv.org/abs/1802.01505

# Speed of light in km/s
c = 299792.458

legend, z_values, H_values, dH_values = get_data()


def H_z(z, params):
    h0, o_m, w0, _ = params
    return h0 * np.sqrt(o_m * (1 + z)**3 + (1 - o_m) * ((2*(1 + z)**2)/(1 + (1 + z)**2))**(3*(1 + w0)))


bounds = np.array([
    (50, 100), # H0
    (0, 1), # Ωm
    (-4, 1), # w0
    (-5, 5), # wa
])


def chi_squared(params):
    delta = H_values - H_z(z_values, params)
    return np.sum(delta**2 / dH_values**2)


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 200
    burn_in = 500
    nsteps = 5000 + burn_in
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

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [H0_16, H0_50, H0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [H0_50, omega_50, w0_50, wa_50]

    print(f"H0: {H0_50:.4f} +{(H0_84 - H0_50):.4f} -{(H0_50 - H0_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degs of freedom: {z_values.size  - len(best_fit)}")

    labels = [f"$H_0$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]

    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=2,
        smooth1d=2,
        bins=50,
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit[i], color='white', linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
H0: 67.8132 +3.0532 -3.0992
Ωm: 0.3259 +0.0656 -0.0553
w0: -1
wa: 0
Chi squared: 14.5125
Degs of freedom: 29

===============================

Flat wCDM
H0: 71.5943 +10.7436 -7.2949
Ωm: 0.2969 +0.0701 -0.0682
w0: -1.3918 +0.6784 -0.8596 (0.51 sigma)
wa: 0
Chi squared: 16.2025
Degs of freedom: 28

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 72.6131 +11.0187 -8.2044
Ωm: 0.2979 +0.0699 -0.0588
w0: -1.4915 +0.7359 -0.9404 (0.59 sigma)
wa: 0
Chi squared: 14.8610
Degs of freedom: 28

===============================

Flat w(z) = w0 + wa * (1 - np.exp(0.5 - 0.5*(1 + z)**2))
H0: 73.7263 +11.5453 -8.4807
Ωm: 0.2993 +0.0947 -0.0820
w0: -1.5086 +0.7635 -0.9814 (0.58 sigma)
wa: -0.1650 +1.6758 -1.8906
Chi squared: 15.2133
Degs of freedom: 27

================================

Flat w(z) = w0 + wa * np.tanh(z)
H0: 73.8512 +11.5379 -8.7214
Ωm: 0.2994 +0.0952 -0.0816
w0: -1.5286 +0.7932 -0.9916 (0.59 sigma)
wa: -0.1583 +1.7003 -1.9123
Chi squared: 15.2418
Degs of freedom: 27
"""