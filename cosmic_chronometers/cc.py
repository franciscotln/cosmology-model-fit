import numpy as np
import emcee
from getdist import MCSamples, plots
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2005cc.data import get_data

# Speed of light in km/s
c = 299792.458

legend, z_values, H_values, dH_values = get_data()


def H_z(z, params):
    h0, o_m, w0 = params[0], params[1], params[2]
    return h0 * np.sqrt(o_m * (1 + z)**3 + (1 - o_m) * ((2*(1 + z)**2)/(1 + (1 + z)**2))**(3*(1 + w0)))


bounds = np.array([
    (40, 110),  # H0
    (0, 0.6),   # Ωm
    (-1.0, 0.5),# w0
    (-5, 5),    # wa
    (0.1, 1.5), # f - overestimation of the uncertainties
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
    return -0.5 * chi_squared(params) - n * np.log(f)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def plot_predictions(params):
    h0, f = params[0], params[-1]
    z_smooth = np.linspace(0, max(z_values), 100)
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=z_values,
        y=H_values,
        yerr=dH_values,
        fmt='.',
        color='red',
        alpha=0.4,
        label="CC data",
        capsize=2,
        linestyle="None",
    )
    plt.errorbar(
        x=z_values,
        y=H_values,
        yerr=dH_values * f,
        fmt='.',
        color='green',
        alpha=0.4,
        label=f"CC data - corrected uncertainties f={f:.3f}",
        capsize=2,
        linestyle="None",
    )
    plt.plot(z_smooth, H_z(z_smooth, params), color='blue', alpha=0.5)
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z_values) + 0.2)
    plt.legend()
    plt.title(f"Cosmic chronometers: $H_0$={h0:.2f} km/s/Mpc")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=z_values,
        y=H_values - H_z(z_values, params),
        yerr=dH_values * f,
        fmt='.',
        color='blue',
        alpha=0.4,
        label="Residuals",
        capsize=2,
        linestyle="None",
    )
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z) - H_{model}(z)$")
    plt.xlim(0, np.max(z_values) + 0.2)
    plt.title(f"Residuals")
    plt.legend()
    plt.show()


def main():
    ndim = len(bounds)
    nwalkers = 6 * ndim
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

    plot_predictions(best_fit)

    labels = ["H_0", "\Omega_m", "w_0", "w_a", "f"]
    gdsamples = MCSamples(
        samples=samples,
        names=labels,
        labels=labels,
        settings={"fine_bins_2D": 128, "smooth_scale_2D": 0.9}
    )
    g = plots.get_subplot_plotter()
    g.triangle_plot(
        gdsamples,
        Filled=False,
        contour_levels=[0.68, 0.95],
        diag1d_kwargs={"density": True},
    )
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
https://github.com/ja-vazquez/SimpleMC/blob/master/simplemc/data/HDiagramCompilacion-data.txt
*****************************

Flat ΛCDM
H0: 70.6130 +1.7770 -1.7968 km/s/Mpc
Ωm: 0.2551 +0.0213 -0.0190
w0: -1
wa: 0
f: 0.8212 +0.1111 -0.0912 (1.61 - 1.96 sigma)
Chi squared: 33.2463
Degs of freedom: 34

===============================

Flat wCDM
H0: 65.8011 +3.7739 -3.5234 km/s/Mpc
Ωm: 0.2498 +0.0255 -0.0339
w0: -0.7327 +0.2017 -0.1966 (1.33 - 1.36 sigma)
wa: 0
f: 0.8095 +0.1102 -0.0897 (1.73 - 2.12 sigma)
Chi squared: 33.0736
Degs of freedom: 33

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 65.8573 +4.4043 -3.9929 km/s/Mpc
Ωm: 0.2784 +0.0311 -0.0285
w0: -0.7174 +0.2305 -0.2436 (1.16 - 1.23 sigma)
wa: 0
f: 0.8152 +0.1132 -0.0918 (1.63 - 2.01 sigma)
Chi squared: 32.3295
Degs of freedom: 33

==================================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 66.1453 +4.7985 -4.7897 km/s/Mpc
Ωm: 0.2516 +0.0789 -0.0968
w0: -0.7023 +0.3500 -0.2851
wa: 0.0055 +0.7851 -1.5913
f: 0.8190 +0.1144 -0.0940
Chi squared: 32.2067
Degs of freedom: 32

******************************
Results for data from
https://arxiv.org/pdf/2307.09501
*****************************

Flat ΛCDM
H0: 67.9657 +2.2066 -2.2381 km/s/Mpc
Ωm: 0.3229 +0.0455 -0.0408
w0: -1
wa: 0
f: 0.7178 +0.1076 -0.0854 (2.62 - 3.30 sigma)
Chi squared: 28.2103
Degs of freedom: 29

===============================

Flat wCDM
H0: 71.3986 +7.7268 -5.6946 km/s/Mpc
Ωm: 0.3050 +0.0479 -0.0494
w0: -1.3285 +0.4978 -0.6250 (0.53 - 0.66 sigma)
wa: 0
f: 0.7267 +0.1102 -0.0879 (2.48 - 3.11 sigma)
Chi squared: 28.4272
Degs of freedom: 28

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 65.2403 +2.8317 -3.2932 km/s/Mpc
Ωm: 0.3215 +0.0508 -0.0464
w0: -0.7721 +0.2647 -0.1640 (0.86 - 1.39 sigma)
wa: 0
f: 0.7327 +0.1106 -0.0895 (2.41 - 2.99 sigma)
Chi squared: 27.9593
Degs of freedom: 28

=============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 72.1181 +8.9491 -6.6845 km/s/Mpc
Ωm: 0.3085 +0.0837 -0.0808
w0: -1.3233 +0.6594 -0.8206 (0.39 - 0.49 sigma)
wa: -0.2320 +2.2742 -3.0988
f: 0.7276 +0.1092 -0.0872 (2.49 - 3.12 sigma)
Chi squared: 27.6948
Degs of freedom: 27
"""