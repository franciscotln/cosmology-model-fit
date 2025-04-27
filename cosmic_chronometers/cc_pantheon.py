import numpy as np
import emcee
from getdist import MCSamples, plots
from scipy.integrate import cumulative_trapezoid, quad
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2022pantheonSHOES.data import get_data
from y2005cc.data import get_data as get_cc_data
from hubble.plotting import plot_predictions as plot_sn_predictions

_, z_cc_vals, H_cc_vals, dH_cc_vals = get_cc_data()
legend, z_vals, apparent_mag_values, cov_matrix_sn = get_data()
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


def model_apparent_mag(z, params):
    h0, M = params[0], params[1]
    comoving_distance = (c/h0) * integral_e_z(z, params)
    return M + 25 + 5 * np.log10((1 + z) * comoving_distance)


def plot_cc_predictions(params):
    h0, f = params[0], params[-1]
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
    h0 = params[0]
    return h0 * h_over_h0_model(z, params)


bounds = np.array([
    (60, 80),    # H0
    (-20, -19),  # M
    (0.15, 0.7), # Ωm
    (-3, 0),     # w0
    (-3.5, 3.5), # wa
    (0.1, 1.5), # f - overestimation of the uncertainties in the CC data
])


def chi_squared(params):
    delta_sn = apparent_mag_values - model_apparent_mag(z_vals, params)
    chi_sn = np.dot(delta_sn, np.dot(inverse_cov_sn, delta_sn))

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
    nwalkers = 6 * ndim
    burn_in = 500
    nsteps = 15000 + burn_in
    initial_pos = np.random.default_rng().uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

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
        [h0_16, h0_50, h0_84],
        [M_16, M_50, M_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
        [f_16, f_50, f_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, M_50, omega_50, w0_50, wa_50, f_50]

    deg_of_freedom = z_vals.size + z_cc_vals.size - len(best_fit)

    print(f"H0: {h0_50:.4f} +{(h0_84 - h0_50):.4f} -{(h0_50 - h0_16):.4f}")
    print(f"M: {M_50:.4f} +{(M_84 - M_50):.4f} -{(M_50 - M_16):.4f}")
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
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_apparent_mag(z_vals, best_fit),
        label=f"Best fit: $H_0$={h0_50} $\Omega_m$={omega_50:.4f}",
        x_scale="log"
    )

    labels = ["H_0", "M", "Omega_m", "w_0", "w_a", "f"]
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
        title_limit=True,
        diag1d_kwargs={"density": True},
    )
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
H0: 67.7235 +1.3281 -1.3134
M: -19.4234 +0.0394 -0.0401
Ωm: 0.3293 +0.0169 -0.0165
w0: -1
wa: 0
f: 0.7072 +0.1019 -0.0831 (2.87 - 3.52 sigma)
Chi squared: 1432.8168
Degrees of freedom: 1618

==============================

Flat wCDM: w(z) = w0
H0: 67.7652 +1.3628 -1.3692 km/s/Mpc
M: -19.4203 +0.0418 -0.0426
Ωm: 0.3158 +0.0375 -0.0407
w0: -0.9602 +0.0982 -0.1063 (0.37 - 0.41 sigma)
wa: 0
f: 0.7119 +0.1056 -0.0839 (2.73 - 3.43 sigma)
Chi squared: 1432.2390
Degrees of freedom: 1617

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 67.7419 +1.3238 -1.3605 km/s/Mpc
M: -19.4211 +0.0405 -0.0424
Ωm: 0.3201 +0.0312 -0.0313
w0: -0.9655 +0.0946 -0.1038 (0.33 - 0.36 sigma)
wa: 0
f: 0.7135 +0.1047 -0.0852 (2.74 - 3.36 sigma)
Chi squared: 1432.1188
Degrees of freedom: 1617

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
H0: 67.5435 +1.3550 -1.3776 km/s/Mpc
M: -19.4252 +0.0413 -0.0429
Ωm: 0.3371 +0.0529 -0.0850
w0: -0.9321 +0.0990 -0.1089 (0.62 - 0.69 sigma)
wa: -0.3567 +0.9664 -1.3853
f: 0.7145 +0.1046 -0.0844 (2.73 - 3.38 sigma)
Chi squared: 1433.5840
Degrees of freedom: 1616
"""
