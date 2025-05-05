import numpy as np
import emcee
from getdist import MCSamples, plots
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from y2005cc.data import get_data as get_cc_data
from hubble.plotting import plot_predictions as plot_sn_predictions

_, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
legend, z_vals, distance_moduli_values, cov_matrix_sn = get_data()
inverse_cov_cc = np.linalg.inv(cov_matrix_cc)
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)

c = 299792.458 # Speed of light in km/s


def h_over_h0_model(z, params):
    O_m, w0 = params[2], params[3]
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def integral_e_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)


def model_distance_modulus(z, params):
    delta_M, h0 = params[0], params[1]
    comoving_distance = (c/h0) * integral_e_z(z, params)
    return delta_M + 25 + 5 * np.log10((1 + z) * comoving_distance)


def plot_cc_predictions(params):
    h0 = params[1]
    z_smooth = np.linspace(0, max(z_cc_vals), 100)
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=np.sqrt(np.diag(cov_matrix_cc)),
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
    (-0.55, 0.45), # ΔM
    (55, 80),      # H0
    (0.1, 0.7),    # Ωm
    (-5.0, 3.0),   # w0
    (-4.5, 4.5),   # wa
])


def chi_squared(params):
    delta_sn = distance_moduli_values - model_distance_modulus(z_vals, params)
    chi_sn = np.dot(delta_sn, np.dot(inverse_cov_sn, delta_sn))

    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(cc_delta, np.dot(inverse_cov_cc, cc_delta))

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
    nwalkers = 25
    burn_in = 500
    nsteps = 20000 + burn_in
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
        [delta_M_16, delta_M_50, delta_M_84],
        [h0_16, h0_50, h0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [delta_M_50, h0_50, omega_50, w0_50, wa_50]

    deg_of_freedom = z_vals.size + z_cc_vals.size - len(best_fit)

    print(f"ΔM: {delta_M_50:.4f} +{(delta_M_84 - delta_M_50):.4f} -{(delta_M_50 - delta_M_16):.4f}")
    print(f"H0: {h0_50:.4f} +{(h0_84 - h0_50):.4f} -{(h0_50 - h0_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_distance_modulus(z_vals, best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f} km/s/Mpc, $\Omega_m$={omega_50:.4f}",
        x_scale="log"
    )

    labels = ["ΔM", "H_0", "Ωm", "w_0", "w_a"]
    gdsamples = MCSamples(
        samples=samples,
        names=labels,
        labels=labels,
        settings={"fine_bins_2D": 256, "smooth_scale_2D": 0.9}
    )
    g = plots.get_subplot_plotter()
    g.triangle_plot(
        gdsamples,
        Filled=False,
        contour_levels=[0.68, 0.95],
        title_limit=True,
        diag1d_kwargs={"normed": True},
    )
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
ΔM: -0.2074 +0.1405 -0.1414 mag
H0: 65.7154 +3.4595 -3.3408 km/s/Mpc
Ωm: 0.3519 +0.0261 -0.0245
w0: -1
wa: 0
Chi squared: 38.6630
Degrees of freedom: 51

==============================

Flat wCDM: w(z) = w0
ΔM: -0.1675 +0.1446 -0.1473 mag
H0: 66.6949 +3.5828 -3.5556 km/s/Mpc
Ωm: 0.2914 +0.0592 -0.0695
w0: -0.8189 +0.1332 -0.1529 (1.18 - 1.36 sigma)
wa: 0
Chi squared: 37.2113

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.1674 +0.1465 -0.1457 mag
H0: 66.5698 +3.6045 -3.4910 km/s/Mpc
Ωm: 0.3094 +0.0442 -0.0452
w0: -0.8207 +0.1257 -0.1453 (1.24 - 1.43 sigma)
wa: 0
Chi squared: 36.9608
Degrees of freedom: 50

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
ΔM: -0.2060 +0.1446 -0.1477 mag
H0: 64.8501 +3.6739 -3.4148 km/s/Mpc
Ωm: 0.3947 +0.0438 -0.0709
w0: -0.6903 +0.1656 -0.1708
wa: -2.4214 +1.8205 -1.4098
Chi squared: 35.3016
Degrees of freedom: 49
"""
