import os
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2022pantheonSHOES.data import get_data
from hubble.plotting import plot_predictions as plot_sn_predictions

legend, z_vals, apparent_mag_values, cov_matrix_sn = get_data()
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'

c = 299792.458 # Speed of light in km/s
H0 = 70.0 # Hubble constant in km/s/Mpc

data = np.genfromtxt(
    path_to_data + "data.txt",
    dtype=[("z", float), ("value", float), ("quantity", "U10")],
    delimiter=" ",
    names=True,
)
bao_cov_matrix = np.loadtxt(path_to_data + "covariance.txt", delimiter=" ", dtype=float)
inv_bao_cov_matrix = np.linalg.inv(bao_cov_matrix)


def w_de(z, params):
    _, _, _, w0, wa = params
    return w0 + (wa - w0) * np.tanh(0.5*(1 + z - 1/(1 + z)))


def rho_de(zs, params):
    z = np.linspace(0, np.max(zs), num=2000)
    integral_values = cumulative_trapezoid(3*(1 + w_de(z, params))/(1 + z), z, initial=0)
    return np.exp(np.interp(zs, z, integral_values))


def h_over_h0_model(z, params):
    _, _, O_m, w0, wa = params
    return np.sqrt(O_m * (1 + z)**3 + (1 - O_m) * rho_de(z, params))


def wcdm_integral_of_e_z(zs, params):
    z = np.linspace(0, np.max(zs), num=2000)
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)


def wcdm_apparent_mag(z, params):
    M0 = params[1]
    a0_over_ae = 1 + z
    luminosity_distance = a0_over_ae * (c/H0) * wcdm_integral_of_e_z(z, params)
    return M0 + 25 + 5 * np.log10(luminosity_distance)


def plot_bao_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(bao_cov_matrix))

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    r_d, M0, omega_m, w0, wa = params
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
                model_smooth.append(DV_z(z, params)/(H0*r_d))
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, params)/(H0*r_d))
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, params))/(H0*r_d))
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO model: $r_d * h$={r_d:.2f}, $M_0$={M0:.3f}, $\Omega_M$={omega_m:.4f}, $w_0$={w0:.4f}, $w_a$={wa:.4f}")
    plt.show()


def H_z(z, params):
    return h_over_h0_model(z, params)
    # return np.sqrt(omega_m * sum**3 + (1 - omega_m)) # LCDM


def DM_z(zs, params):
    z = np.linspace(0, np.max(zs), num=2000)
    return cumulative_trapezoid(c / H_z(z, params), z, initial=0)[-1]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)


def model_predictions(params):
    r_d = params[0]
    predictions = []
    for z, _, quantity in data:
        if quantity == "DV_over_rs":
            predictions.append(DV_z(z, params) / (r_d*H0))
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params) / (r_d*H0))
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params)) / (r_d*H0))
    return np.array(predictions)


bounds = np.array([
    (115, 160), # r_d
    (-20, -19), # M
    (0.2, 0.7), # Ωm
    (-2, 0), # w0
    (-4, 2), # wa
])


def chi_squared(params):
    delta_sn = apparent_mag_values - wcdm_apparent_mag(z_vals, params)
    chi_sn = np.dot(delta_sn, np.dot(inverse_cov_sn, delta_sn))
    delta_bao = data['value'] - model_predictions(params)
    chi_bao = np.dot(delta_bao, np.dot(inv_bao_cov_matrix, delta_bao))
    return chi_sn + chi_bao


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
        [rd_16, rd_50, rd_84],
        [M_16, M_50, M_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, M_50, omega_50, w0_50, wa_50]

    print(f"r_d: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(f"M0: {M_50:.4f} +{(M_84 - M_50):.4f} -{(M_50 - M_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {data['value'].size + z_vals.size - len(best_fit)}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=wcdm_apparent_mag(z_vals, best_fit),
        label=f"Best fit: $w_0$={w0_50:.4f}, $\Omega_m$={omega_50:.4f}",
        x_scale="log"
    )

    labels = [r"$r_d$", f"$M_0$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]
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

    fig, axes = plt.subplots(ndim, figsize=(10, 7))
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
Flat ΛCDM
r_d: 144.3023 +0.9670 -0.9510
M0: -19.3601 +0.0046 -0.0046
Ωm: 0.3042 +0.0080 -0.0078
w0: -1.0
wa: 0.0
Chi squared: 1416.77
Degrees of freedom: 1600

====================

Flat wCDM
r_d: 142.5986 +1.2425 -1.2467
M0: -19.3480 +0.0072 -0.0072
Ωm: 0.2977 +0.0086 -0.0085
w0: -0.9159 +0.0393 -0.0396
wa: 0.0
Chi squared: 1412.41
Degrees of freedom: 1599

====================

Flat w0waCDM
r_d: 142.5619 +1.2552 -1.2665
M0: -19.3464 +0.0095 -0.0092
Ωm: 0.3021 +0.0154 -0.0228
w0: -0.8964 +0.0607 -0.0561
wa: -0.1376 +0.4676 -0.4503
Chi squared: 1412.36
Degrees of freedom: 1598

=========================

Flat w0 + (wa - w0) * z
r_d: 142.4138 +1.2649 -1.2587
M0: -19.3431 +0.0082 -0.0080
Ωm: 0.3155 +0.0118 -0.0124
w0: -0.8891 +0.0464 -0.0443
wa: -0.2567 +0.1710 -0.2163
Chi squared: 1412.1772
Degrees of freedom: 1598

=========================

Flat w0 + (wa - w0)*tanh(z)
r_d: 142.5354 +1.2467 -1.2518
M0: -19.3459 +0.0090 -0.0089
Ωm: 0.3045 +0.0149 -0.0229
w0: -0.8963 +0.0521 -0.0492
wa: -1.0303 +0.2875 -0.2563
Chi squared: 1412.20
Degrees of freedom: 1598

==============================

Flat w0 + (wa - w0)*tanh(0.5*(1 + z - 1/(1 + z)))
r_d: 142.5106 +1.2621 -1.2396
M0: -19.3459 +0.0089 -0.0091
Ωm: 0.3034 +0.0155 -0.0260
w0: -0.8954 +0.0550 -0.0515
wa: -1.0312 +0.3718 -0.3226
Chi squared: 1412.28
Degrees of freedom: 1598
"""