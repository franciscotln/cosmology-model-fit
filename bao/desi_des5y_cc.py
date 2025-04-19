import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data
from y2005cc.data import get_data as get_cc_data
from hubble.plotting import plot_predictions as plot_sn_predictions

_, z_cc_vals, H_cc_vals, dH_cc_vals = get_cc_data()
legend, z_vals, distance_moduli_values, cov_matrix_sn = get_data()
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)

c = 299792.458 # Speed of light in km/s

# Load BAO data
data = np.genfromtxt(fname="bao/raw-data/data.txt", delimiter=" ", names=True,
    dtype=[("z", float), ("value", float), ("quantity", "U10")])
cov_matrix = np.loadtxt("bao/raw-data/covariance.txt", delimiter=" ", dtype=float)
inv_cov_matrix = np.linalg.inv(cov_matrix)


def h_over_h0_model(z, params):
    _, _, O_m, w0, _ = params
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def integral_e_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)


def model_distance_modulus(z, params):
    h0 = params[0]
    comoving_distance = (c/h0) * integral_e_z(z, params)
    return 25 + 5 * np.log10((1 + z) * comoving_distance)


def plot_bao_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(cov_matrix))

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    h0, r_d, omega_m, w0, wa = params
    z_smooth = np.linspace(0, max(z_values), 100)
    plt.figure(figsize=(8, 6))
    for q in unique_quantities:
        mask = quantity_types == q
        plt.errorbar(
            x=z_values[mask],
            y=observed_values[mask],
            yerr=errors[mask],
            fmt='.',
            color=colors[q],
            label=f"Data: {q}",
            capsize=2,
            linestyle="None",
        )
        model_smooth = []
        for z in z_smooth:
            if q == "DV_over_rs":
                model_smooth.append(DV_z(z, params)/r_d)
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, params)/r_d)
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, params))/r_d)
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO model: $r_d$={r_d:.2f}, $\Omega_M$={omega_m:.4f}, $w_0$={w0:.4f}, $w_a$={wa:.4f}")
    plt.show()

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


def DM_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    return cumulative_trapezoid(c / H_z(z, params), z, initial=0)[-1]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)


def model_predictions(params):
    r_d = params[1]
    predictions = []
    for z, _, quantity in data:
        if quantity == "DV_over_rs":
            predictions.append(DV_z(z, params) / r_d)
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params) / r_d)
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params)) / r_d)
    return np.array(predictions)


bounds = np.array([
    (60, 80), # H0
    (115, 160), # r_d
    (0.2, 0.7), # omega_m
    (-3, 0), # w0
    (-3.5, 3.5), # wa
])


def chi_squared(params):
    delta_sn = distance_moduli_values - model_distance_modulus(z_vals, params)
    chi_sn = np.dot(delta_sn, np.dot(inverse_cov_sn, delta_sn))

    delta_bao = data['value'] - model_predictions(params)
    chi_bao = np.dot(delta_bao, np.dot(inv_cov_matrix, delta_bao))

    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.sum(cc_delta**2 / dH_cc_vals**2)
    return chi_sn + chi_bao + chi_cc


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
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, rd_50, omega_50, w0_50, wa_50]
    DES5Y_EFF_SAMPLE = 1735
    deg_of_freedom = DES5Y_EFF_SAMPLE + data['value'].size + z_cc_vals.size - len(best_fit)

    print(f"h0: {h0_50:.4f} +{(h0_84 - h0_50):.4f} -{(h0_50 - h0_16):.4f}")
    print(f"r_d: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_distance_modulus(z_vals, best_fit),
        label=f"Best fit: $w_0$={w0_50:.4f}, $\Omega_m$={omega_50:.4f}",
        x_scale="log"
    )

    labels = [r"$H_0$", r"$r_d$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]
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
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit[i], color='white', linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
h0: 70.1105 +0.2118 -0.2018
r_d: 143.5180 +0.7036 -0.7023
Ωm: 0.3094 +0.0075 -0.0079
w0: -1
wa: 0
Chi squared: 1674.0150
Degrees of freedom: 1776

==============================

Flat wCDM: w(z) = w0
h0: 69.2248 +0.3486 -0.3498
r_d: 143.1237 +0.7199 -0.7138
Ωm: 0.2980 +0.0086 -0.0089
w0: -0.8813 +0.0394 -0.0387 (3.04 sigma)
wa: 0
Chi squared: 1664.5022
Degrees of freedom: 1775

==============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
h0: 69.1029 +0.3717 -0.3687
r_d: 143.1781 +0.7141 -0.7093
Ωm: 0.3042 +0.0080 -0.0077
w0: -0.8614 +0.0420 -0.0423 (3.29 sigma)
wa: 0
Chi squared: 1663.5389
Degrees of freedom: 1775

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
h0: 68.8360 +0.4414 -0.4402
r_d: 143.4270 +0.7432 -0.7410
Ωm: 0.3194 +0.0131 -0.0155
w0: -0.7984 +0.0725 -0.0678 (2.87 sigma)
wa: -0.6686 +0.4515 -0.4573 (1.47 sigma)
Chi squared: 1662.2341
Degrees of freedom: 1774

============================

Flat linear: w(z) = w0 + wa * z
h0: 68.8757 +0.4015 -0.4072
r_d: 143.4345 +0.7330 -0.7264
Ωm: 0.3254 +0.0121 -0.0134
w0: -0.8302 +0.0542 -0.0495 (3.27 sigma)
wa: -0.4315 +0.2358 -0.2510 (1.77 sigma)
Chi squared: 1662.2735
Degrees of freedom: 1774

============================

Flat non-linear: w(z) = w0 + wa * (1 - 1/np.exp(z))
h0: 68.8720 +0.4379 -0.4316
r_d: 143.4004 +0.7469 -0.7266
Ωm: 0.3194 +0.0134 -0.0174
w0: -0.8136 +0.0648 -0.0593 (3.00 sigma)
wa: -0.5215 +0.3876 -0.3691 (1.38 sigma)
Chi squared: 1662.2767
Degrees of freedom: 1774

============================

Flat non-linear: w(z) = w0 + wa * (1 - np.exp(0.5 - 0.5 * (1 + z)**2))
h0: 68.8768 +0.4291 -0.4194
r_d: 143.4185 +0.7426 -0.7487
Ωm: 0.3190 +0.0131 -0.0164
w0: -0.8210 +0.0602 -0.0561 (3.08 sigma)
wa: -0.4185 +0.3009 -0.3050 (1.38 sigma)
Chi squared: 1662.3222
Degrees of freedom: 1774

=============================

Flat non-linear: w(z) = w0 + wa * np.tanh(z)
h0: 68.8907 +0.4211 -0.4222
r_d: 143.4194 +0.7504 -0.7504
Ωm: 0.3190 +0.0132 -0.0158
w0: -0.8217 +0.0605 -0.0558 (3.07 sigma)
wa: -0.4272 +0.3030 -0.3014 (1.42 sigma)
Chi squared: 1662.3258
Degrees of freedom: 1774

=============================

Flat non-linear: w(z) = w0 + wa * np.tanh(0.5*((1 + z) **2 - 1))
h0: 68.8950 +0.4183 -0.4268
r_d: 143.4160 +0.7610 -0.7436
Ωm: 0.3177 +0.0130 -0.0154
w0: -0.8269 +0.0560 -0.0539 (3.15 sigma)
wa: -0.3352 +0.2380 -0.2407 (1.40 sigma)
Chi squared: 1662.4143
Degrees of freedom: 1774
"""
