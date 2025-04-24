import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2005cc.compilation_data import get_data

# Speed of light in km/s
c = 299792.458

legend, z_values, H_values, dH_values = get_data()


def H_z(z, params):
    h0, o_m, w0, _ = params
    return h0 * np.sqrt(o_m * (1 + z)**3 + (1 - o_m) * ((2*(1 + z)**2)/(1 + (1 + z)**2))**(3*(1 + w0)))


bounds = np.array([
    (40, 110), # H0
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
    nsteps = 8000 + burn_in
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
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3, lw=0.5)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit[i], color='white', linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()


"""
*****************************
Compilation data (37 data points)
*****************************

Flat ΛCDM
H0: 70.5575 +2.1501 -2.1998
Ωm: 0.2560 +0.0264 -0.0233
w0: -1
wa: 0
Chi squared: 22.4221
Degs of freedom: 35

===============================

Flat wCDM
H0: 65.7016 +4.6575 -4.1304
Ωm: 0.2467 +0.0320 -0.0438
w0: -0.7299 +0.2411 -0.2393 (1.12 sigma)
wa: 0
Chi squared: 22.1189
Degs of freedom: 34

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 65.9320 +5.3727 -4.8331
Ωm: 0.2777 +0.0382 -0.0341
w0: -0.7251 +0.2796 -0.2997 (0.95 sigma)
wa: 0
Chi squared: 21.5051
Degs of freedom: 34

==================================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 65.6112 +5.9901 -5.6083
Ωm: 0.2691 +0.0868 -0.0920
w0: -0.6582 +0.4708 -0.3727 (0.81 sigma)
wa: -0.3067 +1.0713 -2.0416
Chi squared: 21.5502
Degs of freedom: 33

******************************
Results for data from
https://arxiv.org/abs/1802.01505
*****************************

Flat ΛCDM
H0: 67.7511 +3.0747 -3.1238
Ωm: 0.3271 +0.0658 -0.0556
w0: -1
wa: 0
Chi squared: 14.5113
Degs of freedom: 29

===============================

Flat wCDM
H0: 71.9539 +11.1382 -7.3525
Ωm: 0.2969 +0.0693 -0.0674
w0: -1.4156 +0.6610 -0.8845
wa: 0
Chi squared: 16.0122
Degs of freedom: 28

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 72.7292 +11.6313 -8.2271
Ωm: 0.2961 +0.0703 -0.0606
w0: -1.5073 +0.7512 -0.9802
wa: 0
Chi squared: 14.9783
Degs of freedom: 28
"""