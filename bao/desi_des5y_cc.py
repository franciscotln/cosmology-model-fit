import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, distance_moduli_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

inverse_cov_cc = np.linalg.inv(cov_matrix_cc)
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)
inverse_cov_bao = np.linalg.inv(cov_matrix_bao)

c = 299792.458 # Speed of light in km/s

z_grid_sn = np.linspace(0, np.max(z_sn_vals), num=3000)


def h_over_h0_model(z, params):
    O_m, w0 = params[3], params[4]
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def integral_e_z(params):
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z_grid_sn, params), z_grid_sn, initial=0)
    return np.interp(z_sn_vals, z_grid_sn, integral_values)


def model_distance_modulus(params):
    delta_M, h0 = params[0], params[1]
    return delta_M + 25 + 5 * np.log10((1 + z_sn_vals) * (c / h0) * integral_e_z(params))


def plot_bao_predictions(params):
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }
    h0, r_d, omega_m = params[1], params[2], params[3]
    z_smooth = np.linspace(0, max(bao_data["z"]), 100)

    plt.figure(figsize=(8, 6))
    for q in set(bao_data["quantity"]):
        mask = bao_data["quantity"] == q
        plt.errorbar(
            x=bao_data["z"][mask],
            y=bao_data["value"][mask],
            yerr=np.sqrt(np.diag(cov_matrix_bao))[mask],
            fmt='.',
            color=colors[q],
            label=f"Data: {q}",
            capsize=2,
            alpha=0.6,
        )
        model_smooth = []
        for z in z_smooth:
            if q == "DV_over_rs":
                model_smooth.append(DV_z(z, params)/r_d)
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, params)/r_d)
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, params))/r_d)
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.6)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"{bao_legend}: $H_0$={h0:.2f} km/s/Mpc, $r_d$={r_d:.2f} Mpc, $\Omega_M$={omega_m:.3f}")
    plt.show()

    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=np.sqrt(np.diag(cov_matrix_cc)),
        fmt='.',
        color='blue',
        alpha=0.4,
        label="CCH data",
        capsize=2,
    )
    plt.plot(z_smooth, H_z(z_smooth, params), color='red', alpha=0.5, label="Model")
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z_cc_vals) + 0.2)
    plt.legend()
    plt.title(f"{cc_legend}: $H_0$={h0:.2f} km/s/Mpc")
    plt.show()


def H_z(z, params):
    h0 = params[1]
    return h0 * h_over_h0_model(z, params)


def DM_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    return cumulative_trapezoid(c / H_z(z, params), z, initial=0)[-1]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)


def bao_predictions(params):
    r_d = params[2]
    predictions = []
    for z, _, quantity in bao_data:
        if quantity == "DV_over_rs":
            predictions.append(DV_z(z, params) / r_d)
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params) / r_d)
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params)) / r_d)
    return np.array(predictions)


bounds = np.array([
    (-0.55, 0.55), # ΔM
    (50, 80),      # H0
    (110, 175),    # r_d
    (0.2, 0.7),    # Ωm
    (-1.1, -0.4),  # w0
])


def chi_squared(params):
    delta_sn = distance_moduli_values - model_distance_modulus(params)
    chi_sn = np.dot(delta_sn, np.dot(inverse_cov_sn, delta_sn))

    delta_bao = bao_data['value'] - bao_predictions(params)
    chi_bao = np.dot(delta_bao, np.dot(inverse_cov_bao, delta_bao))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, np.dot(inverse_cov_cc, delta_cc))
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
    nwalkers = 10 * ndim
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
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [delta_M_50, h0_50, rd_50, omega_50, w0_50]
    DES5Y_EFF_SAMPLE = 1735
    deg_of_freedom = DES5Y_EFF_SAMPLE + bao_data['value'].size + z_cc_vals.size - len(best_fit)

    print(f"ΔM: {delta_M_50:.3f} +{(delta_M_84 - delta_M_50):.3f} -{(delta_M_50 - delta_M_16):.3f}")
    print(f"H0: {h0_50:.2f} +{(h0_84 - h0_50):.2f} -{(h0_50 - h0_16):.2f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"Ωm: {omega_50:.3f} +{(omega_84 - omega_50):.3f} -{(omega_50 - omega_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_distance_modulus(best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f}, $\Omega_m$={omega_50:.3f}",
        x_scale="log"
    )

    labels = ["ΔM", r"$H_0$", r"$r_d$", "Ωm", r"$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864), # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
ΔM: -0.061 +0.101 -0.105 mag
H0: 68.18 +3.28 -3.25 km/s/Mpc
r_d: 147.51 +7.32 -6.69 Mpc
Ωm: 0.310 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1673.12
Degrees of freedom: 1776

==============================

Flat wCDM: w(z) = w0
ΔM: -0.061 +0.101 -0.105 mag
H0: 67.28 +3.24 -3.22 km/s/Mpc
r_d: 147.08 +7.25 -6.67 Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.877 +0.038 -0.038 (3.2 sigma)
wa: 0
Chi squared: 1663.20
Degrees of freedom: 1775

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.060 +0.101 -0.108 mag
H0: 67.19 +3.23 -3.29 km/s/Mpc
r_d: 147.07 +7.46 -6.65 Mpc
Ωm: 0.305 +0.008 -0.008
w0: -0.858 +0.042 -0.042 (3.4 sigma)
wa: 0
Chi squared: 1662.25
Degrees of freedom: 1775

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
ΔM: -0.056 +0.101 -0.107 mag
H0: 67.05 +3.25 -3.25 km/s/Mpc
r_d: 147.09 +7.39 -6.68 Mpc
Ωm: 0.319 +0.013 -0.017
w0: -0.800 +0.074 -0.067 (2.7 - 3.0 sigma)
wa: -0.6385 +0.4721 -0.4662
Chi squared: 1661.08
Degrees of freedom: 1774
"""
