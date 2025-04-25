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
    (0, 0.6), # Ωm
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
    n = len(H_values)
    norm_term = n * np.log(2 * np.pi) + 2 * n * np.log(f) + 2 * np.sum(np.log(dH_values))
    return -0.5 * (chi_squared(params) + norm_term)


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
Compilation data (41 data points)
https://arxiv.org/pdf/1709.00646
*****************************

Flat ΛCDM
H0: 69.7883 +1.3872 -1.3937 km/s/Mpc
Ωm: 0.2612 +0.0161 -0.0150
w0: -1
wa: 0
f: 0.7042 +0.0902 -0.0749
Chi squared: 37.3390
Degs of freedom: 38

===============================

Flat wCDM
H0: 68.8820 +3.7148 -3.4348 km/s/Mpc
Ωm: 0.2596 +0.0190 -0.0191
w0: -0.9516 +0.1823 -0.1828 (0.26 - 0.27 - sigma)
wa: 0
f: 0.7143 +0.0937 -0.0771
Chi squared: 36.6480
Degs of freedom: 37

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 69.5990 +4.2308 -3.8838 km/s/Mpc
Ωm: 0.2619 +0.0267 -0.0253
w0: -0.9898 +0.2233 -0.2324 (0.05 sigma)
wa: 0
f: 0.7137 +0.0934 -0.0763
Chi squared: 36.3607
Degs of freedom: 37

==================================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 69.9303 +4.7256 -4.6145 km/s/Mpc
Ωm: 0.2362 +0.0630 -0.0679
w0: -0.9819 +0.3267 -0.2896 (0.06 sigma)
wa: 0.4612 +0.7379 -1.6136
f: 0.7143 +0.0945 -0.0766
Chi squared: 36.4226
Degs of freedom: 36

******************************
Results for data from
https://arxiv.org/pdf/2307.09501
*****************************

Flat ΛCDM
H0: 67.9926 +2.1899 -2.2426 km/s/Mpc
Ωm: 0.3229 +0.0454 -0.0406
w0: -1
wa: 0
f: 0.7167 +0.1063 -0.0858
Chi squared: 28.2945
Degs of freedom: 29

===============================

Flat wCDM
H0: 71.2178 +7.4944 -5.5947 km/s/Mpc
Ωm: 0.3053 +0.0480 -0.0498
w0: -1.3176 +0.5007 -0.6055
wa: 0
f: 0.7258 +0.1094 -0.0874
Chi squared: 28.5953
Degs of freedom: 28

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 72.4238 +8.2836 -6.2933
Ωm: 0.3012 +0.0491 -0.0455
w0: -1.4340 +0.5595 -0.7065
f: 0.7251 +0.1096 -0.0875
Chi squared: 27.6688
Degs of freedom: 28

=============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 72.2010 +8.8488 -6.7042 km/s/Mpc
Ωm: 0.3091 +0.0821 -0.0808
w0: -1.3259 +0.6470 -0.8179
wa: -0.2805 +2.2945 -3.0230
f: 0.7292 +0.1108 -0.0881
Chi squared: 27.5744
Degs of freedom: 27
"""