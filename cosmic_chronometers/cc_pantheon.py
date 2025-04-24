import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2022pantheonSHOES.data import get_data
from y2005cc.compilation_data import get_data as get_cc_data
from hubble.plotting import plot_predictions as plot_sn_predictions

_, z_cc_vals, H_cc_vals, dH_cc_vals = get_cc_data()
legend, z_vals, apparent_mag_values, cov_matrix_sn = get_data()
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)

c = 299792.458 # Speed of light in km/s


def h_over_h0_model(z, params):
    _, _, O_m, w0, _ = params
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def integral_e_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)


def model_apparent_mag(z, params):
    h0, M = params[0], params[1]
    comoving_distance = (c/h0) * integral_e_z(z, params)
    return M + 25 + 5 * np.log10((1 + z) * comoving_distance)


def plot_cc_predictions(params):
    h0 = params[0]
    z_smooth = np.linspace(0, max(z_cc_vals), 100)
    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=dH_cc_vals,
        fmt='.',
        color='blue',
        alpha=0.4,
        label="CC data",
        capsize=2,
        linestyle="None",
    )
    plt.plot(z_smooth, H_z(z_smooth, params), color='green', alpha=0.5)
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z_cc_vals) + 0.2)
    plt.legend()
    plt.title(f"Cosmic chronometers: $H_0$={h0:.2f} km/s/Mpc")
    plt.show()


def H_z(z, params):
    h0 = params[0]
    return h0 * h_over_h0_model(z, params)


bounds = np.array([
    (60, 80), # H0
    (-20, -19), # M
    (0.15, 0.7), # Ωm
    (-3, 0), # w0
    (-3.5, 3.5), # wa
])


def chi_squared(params):
    delta_sn = apparent_mag_values - model_apparent_mag(z_vals, params)
    chi_sn = np.dot(delta_sn, np.dot(inverse_cov_sn, delta_sn))

    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.sum(cc_delta**2 / dH_cc_vals**2)
    return chi_sn + chi_cc


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
    nwalkers = 100
    burn_in = 500
    nsteps = 5000 + burn_in
    initial_pos = np.random.default_rng().uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

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
        [h0_16, h0_50, h0_84],
        [M_16, M_50, M_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, M_50, omega_50, w0_50, wa_50]

    deg_of_freedom = z_vals.size + z_cc_vals.size - len(best_fit)

    print(f"H0: {h0_50:.4f} +{(h0_84 - h0_50):.4f} -{(h0_50 - h0_16):.4f}")
    print(f"M: {M_50:.4f} +{(M_84 - M_50):.4f} -{(M_50 - M_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_apparent_mag(z_vals, best_fit),
        label=f"Best fit: $\Omega_m$={omega_50:.4f}, $w_0$={w0_50:.4f}, $w_a$={wa_50:.4f}",
        x_scale="log"
    )

    labels = [r"$H_0$", r"$M$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]
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

    _, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3, lw=0.4)
        axes[i].set_ylabel(labels[i])
        axes[i].axvline(x=burn_in, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit[i], color='white', linestyle='--', alpha=0.5)
    axes[ndim - 1].set_xlabel("chain step")
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
H0: 66.6355 +1.4123 -1.3656
M: -19.4650 +0.0422 -0.0413
Ωm: 0.3100 +0.0158 -0.0152
w0: -1
wa: 0
Chi squared: 1431.5870
Degrees of freedom: 1624

==============================

Flat wCDM: w(z) = w0
H0: 67.6556 +1.4180 -1.4375 km/s/Mpc
M: -19.4197 +0.0435 -0.0447
Ωm: 0.2627 +0.0244 -0.0244
w0: -0.8449 +0.0550 -0.0584 (2.74 sigma)
wa: 0
Chi squared: 1424.9563
Degrees of freedom: 1623

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 67.5648 +1.4409 -1.4105 km/s/Mpc
M: -19.4216 +0.0442 -0.0446
Ωm: 0.2741 +0.0211 -0.0202
w0: -0.8427 +0.0566 -0.0623 (2.65 sigma)
wa: 0
Chi squared: 1425.4876
Degrees of freedom: 1623

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
H0: 67.5160 +1.4111 -1.4069 km/s/Mpc
M: -19.4261 +0.0437 -0.0446
Ωm: 0.2428 +0.0407 -0.0533
w0: -0.8404 +0.0599 -0.0651 (2.55 sigma)
wa: 0.2968 +0.3470 -0.5623
Chi squared: 1425.0559
Degrees of freedom: 1622
"""
