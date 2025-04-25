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
H0: 65.9044 +4.3232 -3.9700 km/s/Mpc
Ωm: 0.2780 +0.0309 -0.0282
w0: -0.7197 +0.2296 -0.2419 (1.19 sigma)
wa: 0
f: 0.8149 +0.1123 -0.0912
Chi squared: 32.3601
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
https://arxiv.org/abs/1802.01505
*****************************

Flat ΛCDM
H0: 67.9130 +2.2279 -2.2794 km/s/Mpc
Ωm: 0.3241 +0.0469 -0.0412
w0: -1
wa: 0
f: 0.7277 +0.1108 -0.0879
Chi squared: 27.3841
Degs of freedom: 28

===============================

Flat wCDM
H0: 71.2090 +7.7199 -5.7079 km/s/Mpc
Ωm: 0.3059 +0.0494 -0.0518
w0: -1.3228 +0.5083 -0.6281 (0.57 sigma)
wa: 0
f: 0.7391 +0.1144 -0.0902
Chi squared: 27.5378
Degs of freedom: 27

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 72.3738 +8.4821 -6.3151 km/s/Mpc
Ωm: 0.3016 +0.0500 -0.0465
w0: -1.4404 +0.5708 -0.7245 (0.68 sigma)
wa: 0
f: 0.7373 +0.1139 -0.0908
Chi squared: 26.7646
Degs of freedom: 27
"""