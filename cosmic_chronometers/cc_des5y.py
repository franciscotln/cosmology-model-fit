import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data
from y2005cc.compilation_data import get_data as get_cc_data
from hubble.plotting import plot_predictions as plot_sn_predictions

_, z_cc_vals, H_cc_vals, dH_cc_vals = get_cc_data()
legend, z_vals, distance_moduli_values, cov_matrix_sn = get_data()
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)

c = 299792.458 # Speed of light in km/s


def h_over_h0_model(z, params):
    _, _, O_m, w0, _, _ = params
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def integral_e_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)


def model_distance_modulus(z, params):
    delta_M, h0 = params[0], params[1]
    comoving_distance = (c / h0) * integral_e_z(z, params)
    return delta_M + 25 + 5 * np.log10((1 + z) * comoving_distance)


def plot_cc_predictions(params):
    h0, f = params[1], params[-1]
    z_smooth = np.linspace(0, max(z_cc_vals), 100)
    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=dH_cc_vals * f,
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
    h0 = params[1]
    return h0 * h_over_h0_model(z, params)


bounds = np.array([
    (-0.5, 0.5), # ΔM
    (60, 80),    # H0
    (0.1, 0.5), # Ωm
    (-3, 0),     # w0
    (-3.5, 3.5), # wa
    (0.01, 1.5), # f - overestimation of the uncertainties in the CC data
])


def chi_squared(params):
    delta_sn = distance_moduli_values - model_distance_modulus(z_vals, params)
    chi_sn = delta_sn @ inverse_cov_sn @ delta_sn

    f = params[-1]
    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    escaled_error = dH_cc_vals * f
    chi_cc = np.sum(cc_delta**2 / escaled_error**2)

    return chi_sn + chi_cc


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f = params[-1]
    return -0.5 * chi_squared(params) - z_cc_vals.size * np.log(f)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 500
    nsteps = 4000 + burn_in
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
        [delta_M_16, delta_M_50, delta_M_84],
        [h0_16, h0_50, h0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
        [f_16, f_50, f_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [delta_M_50, h0_50, omega_50, w0_50, wa_50, f_50]
    DES5Y_EFF_SAMPLE = 1735
    deg_of_freedom = DES5Y_EFF_SAMPLE + z_cc_vals.size - len(best_fit)

    print(f"ΔM: {delta_M_50:.4f} +{(delta_M_84 - delta_M_50):.4f} -{(delta_M_50 - delta_M_16):.4f}")
    print(f"H0: {h0_50:.4f} +{(h0_84 - h0_50):.4f} -{(h0_50 - h0_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"f: {f_50:.4f} +{(f_84 - f_50):.4f} -{(f_50 - f_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_distance_modulus(z_vals, best_fit),
        label=f"Best fit: $\Omega_m$={omega_50:.4f}, $w_0$={w0_50:.4f}, $w_a$={wa_50:.4f}",
        x_scale="log"
    )

    labels = [r"$\Delta_M$", r"$H_0$", f"$\Omega_m$", r"$w_0$", r"$w_a$", f"$f$"]
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
*******************************
Here we are considering the uncertainties to be overestimated
in the CC data. We then use the parameter f to account for this.
The results show consistent values for f and the corner plot shows
that the parameters are weakly correlated
*******************************

Flat ΛCDM: w(z) = -1
ΔM: -0.1337 +0.0314 -0.0325
H0: 65.8129 +1.1592 -1.1826 km/s/Mpc
Ωm: 0.3156 +0.0147 -0.0138
w0: -1
wa: 0
f: 0.7878 +0.1049 -0.0886 (2.02 - 2.40 sigma)
Chi squared: 1682.5273
Degrees of freedom: 1772

==============================

Flat wCDM: w(z) = w0
ΔM: -0.0866 +0.0299 -0.0320
H0: 66.3751 +1.0362 -1.0887 km/s/Mpc
Ωm: 0.2710 +0.0177 -0.0167
w0: -0.8192 +0.0451 -0.0484 (3.74 - 4.01 sigma)
wa: 0
f: 0.7125 +0.0964 -0.0759 (2.98 - 3.79 sigma)
Chi squared: 1676.0966
Degrees of freedom: 1771

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.0854 +0.0292 -0.0297
H0: 66.3504 +1.0084 -1.0175 km/s/Mpc
Ωm: 0.2825 +0.0142 -0.0136
w0: -0.8047 +0.0461 -0.0477 (4.09 - 4.24 sigma)
wa: 0
f: 0.7090 +0.0914 -0.0736 (3.18 - 3.95 sigma)
Chi squared: 1676.2809
Degrees of freedom: 1771

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
ΔM: -0.0863 +0.0302 -0.0299
H0: 66.2713 +1.0532 -1.0320 km/s/Mpc
Ωm: 0.2846 +0.0257 -0.0452
w0: -0.7780 +0.0822 -0.0656 (2.70 - 3.38 sigma)
wa: -0.3390 +0.6980 -0.7713 (0.44 - 0.49 sigma)
f: 0.7121 +0.0917 -0.0749 (3.14 - 3.84 sigma)
Chi squared: 1676.3774
Degrees of freedom: 1770
"""
