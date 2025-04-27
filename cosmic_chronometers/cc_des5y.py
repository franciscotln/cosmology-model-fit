import numpy as np
import emcee
from getdist import MCSamples, plots
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data
from y2005cc.data import get_data as get_cc_data
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
        label=f"Best fit: $\Omega_m$={omega_50:.4f}, $H_0$={h0_50:.4f} km/s/Mpc",
        x_scale="log"
    )

    labels = ["ΔM", "H_0", "Ωm", "w_0", "w_a", "f"]
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
*******************************
Here we are considering the uncertainties to be overestimated
in the CC data. We then use the parameter f to account for this.
The results show consistent values for f and the corner plot shows
that the parameters are weakly correlated
*******************************

Flat ΛCDM: w(z) = -1
ΔM: -0.0770 +0.0370 -0.0376
H0: 66.9694 +1.2736 -1.2658 km/s/Mpc
Ωm: 0.3461 +0.0159 -0.0154
w0: -1
wa: 0
f: 0.7110 +0.1040 -0.0844 (2.78 - 3.42 sigma)
Chi squared: 1669.4875
Degrees of freedom: 1763

==============================

Flat wCDM: w(z) = w0
ΔM: -0.0636 +0.0404 -0.0405
H0: 67.1397 +1.3141 -1.3027 km/s/Mpc
Ωm: 0.3072 +0.0404 -0.0452
w0: -0.8876 +0.0982 -0.1058 (1.06 - 1.14 sigma)
wa: 0
f: 0.7185 +0.1047 -0.0848 (2.69 - 3.32 sigma)
Chi squared: 1667.6740
Degrees of freedom: 1762

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.0627 +0.0399 -0.0398
H0: 67.1286 +1.3042 -1.2813 km/s/Mpc
Ωm: 0.3152 +0.0307 -0.0321
w0: -0.8830 +0.0902 -0.0983 (1.19 - 1.30 sigma)
wa: 0
f: 0.7140 +0.1064 -0.0843 (2.69 - 3.39 sigma)
Chi squared: 1667.7513
Degrees of freedom: 1762

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
ΔM: -0.0774 +0.0394 -0.0396
H0: 66.3841 +1.3213 -1.2944 km/s/Mpc
Ωm: 0.3925 +0.0293 -0.0445
w0: -0.7874 +0.1053 -0.1163 (1.83 - 2.02 sigma)
wa: -2.2735 +1.3661 -0.8710 (1.66 - 2.61 sigma)
f: 0.7140 +0.1045 -0.0849 (2.74 - 3.37 sigma)
Chi squared: 1664.7084
Degrees of freedom: 1761
"""
