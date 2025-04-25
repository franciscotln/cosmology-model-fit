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
    h0, o_m, w0, _, _ = params
    return h0 * np.sqrt(o_m * (1 + z)**3 + (1 - o_m) * ((2*(1 + z)**2)/(1 + (1 + z)**2))**(3*(1 + w0)))


bounds = np.array([
    (40, 110), # H0
    (0, 1), # Ωm
    (-4, 1), # w0
    (-5, 5), # wa
    (0.01, 1.5), # f - overestimation of the uncertainties
])


def chi_squared(params):
    f = params[-1]
    delta = H_values - H_z(z_values, params)
    scaled_error = dH_values * f
    return np.sum(delta**2 / scaled_error**2)


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f = params[-1]
    return -0.5 * chi_squared(params) - len(H_values) * np.log(f)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 200
    burn_in = 500
    nsteps = 6000 + burn_in
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
        [f_16, f_50, f_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [H0_50, omega_50, w0_50, wa_50, f_50]

    print(f"H0: {H0_50:.4f} +{(H0_84 - H0_50):.4f} -{(H0_50 - H0_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"f: {f_50:.4f} +{(f_84 - f_50):.4f} -{(f_50 - f_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degs of freedom: {z_values.size  - len(best_fit)}")

    labels = [f"$H_0$", f"$\Omega_m$", r"$w_0$", f"$w_a$", f"$f$"]

    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=2,
        smooth1d=2,
        bins=50,
        plot_datapoints=False,
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
Here we are considering the uncertainties to be overestimated
We use the parameter f to account for this
The results show consistent values for f and the corner plot shows
that the parameters are weakly correlated

*****************************
Compilation data (37 data points)
*****************************

Flat ΛCDM
H0: 70.6237 +1.7631 -1.8060 km/s/Mpc
Ωm: 0.2554 +0.0211 -0.0192
w0: -1
wa: 0
f: 0.8206 +0.1117 -0.0912
Chi squared: 33.2933
Degs of freedom: 34

===============================

Flat wCDM
H0: 65.6996 +3.7489 -3.4417 km/s/Mpc
Ωm: 0.2496 +0.0257 -0.0339
w0: -0.7257 +0.1968 -0.1938 (1.40 sigma)
wa: 0
f: 0.8093 +0.1108 -0.0904
Chi squared: 33.0334
Degs of freedom: 33

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 65.8954 +4.3637 -3.9796 km/s/Mpc
Ωm: 0.2782 +0.0308 -0.0285
w0: -0.7200 +0.2300 -0.2422 (1.16 - 1.22 sigma)
wa: 0
f: 0.8151 +0.1128 -0.0916
Chi squared: 32.3398
Degs of freedom: 33

==================================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 66.1623 +4.8368 -4.7809 km/s/Mpc
Ωm: 0.2494 +0.0802 -0.0961
w0: -0.7062 +0.3496 -0.2904 (0.92 sigma)
wa: 0.0417 +0.7609 -1.6116 (0.04 sigma)
f: 0.8158 +0.1130 -0.0924
Chi squared: 32.4815
Degs of freedom: 32

******************************
Results for data from
https://arxiv.org/pdf/2307.09501
*****************************

Flat ΛCDM
H0: 67.9528 +2.1981 -2.2491 km/s/Mpc
Ωm: 0.3234 +0.0455 -0.0404
w0: -1
wa: 0
f: 0.7154 +0.1060 -0.0853
Chi squared: 28.3968
Degs of freedom: 29

===============================

Flat wCDM
H0: 71.2987 +7.5434 -5.6833 km/s/Mpc
Ωm: 0.3046 +0.0478 -0.0499
w0: -1.3207 +0.5019 -0.6063
wa: 0
f: 0.7262 +0.1091 -0.0879
Chi squared: 28.5601
Degs of freedom: 28

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 72.2793 +8.3092 -6.2258 km/s/Mpc
Ωm: 0.3022 +0.0491 -0.0458
w0: -1.4294 +0.5592 -0.7090
wa: 0
f: 0.7258 +0.1102 -0.0882
Chi squared: 27.6494
Degs of freedom: 28

=============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 72.1762 +8.9081 -6.7251 km/s/Mpc
Ωm: 0.3101 +0.0829 -0.0811
w0: -1.3200 +0.6521 -0.8190
wa: -0.3113 +2.3473 -3.0546
f: 0.7274 +0.1121 -0.0878
Chi squared: 27.6750
Degs of freedom: 27
"""