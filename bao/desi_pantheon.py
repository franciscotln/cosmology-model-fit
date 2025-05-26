import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data import get_data
from hubble.plotting import plot_predictions as plot_sn_predictions

legend, z_vals, z_hel_vals, apparent_mag_values, cov_matrix_sn = get_data()
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)
_, data, bao_cov_matrix = get_bao_data()
inv_bao_cov_matrix = np.linalg.inv(bao_cov_matrix)

c = 299792.458 # Speed of light in km/s
H0 = 70.0 # Hubble constant in km/s/Mpc


def h_over_h0_model(z, params):
    _, _, O_m, w0, _ = params
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


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

    r_d, M0, omega_m = params[0], params[1], params[2]
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
    plt.title(f"BAO model: $r_d * h$={r_d:.2f}, $M_0$={M0:.3f}, $\Omega_M$={omega_m:.4f}")
    plt.show()


def H_z(z, params):
    return h_over_h0_model(z, params)


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
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),
        fill_contours=False,
        plot_datapoints=False,
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
w0: -0.9159 +0.0393 -0.0396 (2.13 sigma)
wa: 0.0
Chi squared: 1412.41
Degrees of freedom: 1599

====================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
r_d: 142.5480 +1.2637 -1.2506
M0: -19.3465 +0.0077 -0.0079
Ωm: 0.3020 +0.0082 -0.0078
w0: -0.9059 +0.0438 -0.0442 (2.14 sigma)
wa: 0
Chi squared: 1412.2302
Degrees of freedom: 1599

====================

Flat w0waCDM
r_d: 142.5619 +1.2552 -1.2665
M0: -19.3464 +0.0095 -0.0092
Ωm: 0.3021 +0.0154 -0.0228
w0: -0.8964 +0.0607 -0.0561 (1.77 sigma)
wa: -0.1376 +0.4676 -0.4503 (0.30 sigma)
Chi squared: 1412.36
Degrees of freedom: 1598

=========================

Flat w0 + wa * z
r_d: 142.4138 +1.2649 -1.2587
M0: -19.3431 +0.0082 -0.0080
Ωm: 0.3155 +0.0118 -0.0124
w0: -0.8891 +0.0464 -0.0443 (2.45 sigma)
wa: -0.2567 +0.1710 -0.2163 (1.33 sigma)
Chi squared: 1412.18
Degrees of freedom: 1598

=========================

Flat w0 + wa * np.tanh(z)
r_d: 142.5392 +1.2591 -1.2540
M0: -19.3457 +0.0091 -0.0090
Ωm: 0.3040 +0.0151 -0.0220
w0: -0.8955 +0.0526 -0.0497 (2.04 sigma)
wa: -0.1259 +0.3063 -0.2957 (0.42 sigma)
Chi squared: 1412.2583
Degrees of freedom: 1598

Flat w0 + wa * np.tanh(0.5*((1 + z)**2 - 1))
r_d: 142.5459 +1.2656 -1.2598
M0: -19.3456 +0.0088 -0.0087
Ωm: 0.3045 +0.0143 -0.0181
w0: -0.8966 +0.0500 -0.0485 (2.10 sigma)
wa: -0.1160 +0.2250 -0.2297 (0.51 sigma)
Chi squared: 1412.1484
Degrees of freedom: 1598

Flat w0 + wa * (1 - np.exp(0.5 - 0.5 * (1 + z)**2))
r_d: 142.5389 +1.2515 -1.2728
M0: -19.3459 +0.0090 -0.0088
Ωm: 0.3044 +0.0149 -0.0225
w0: -0.8960 +0.0512 -0.0484 (2.09 sigma)
wa: -0.1279 +0.3058 -0.2844 (0.43 sigma)
Chi squared: 1412.2664
Degrees of freedom: 1598

==============================

Flat w0 + wa * np.tanh(0.5*(1 + z - 1/(1 + z)))
r_d: 142.5545 +1.2534 -1.2440
M0: -19.3461 +0.0093 -0.0092
Ωm: 0.3039 +0.0155 -0.0248
w0: -0.8950 +0.0561 -0.0519 (1.94 sigma)
wa: -0.1469 +0.3997 -0.3616 (0.39 sigma)
Chi squared: 1412.3338
Degrees of freedom: 1598
"""