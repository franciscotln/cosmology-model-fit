import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions

legend, z_vals, distance_moduli_values, cov_matrix_sn = get_data()
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)
_, data, bao_cov_matrix = get_bao_data()
inv_bao_cov_matrix = np.linalg.inv(bao_cov_matrix)

c = 299792.458 # Speed of light in km/s
H0 = 70 # Hubble constant in km/s/Mpc


def h_over_h0_model(z, params):
    _, _, O_m, w0, _ = params
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def integral_of_e_z(zs, params):
    z = np.linspace(0, np.max(zs), num=2000)
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)


def model_distance_modulus(z, params):
    delta_M = params[0]
    a0_over_ae = 1 + z
    comoving_distance = (c / H0) * integral_of_e_z(z, params)
    return delta_M + 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def plot_bao_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(bao_cov_matrix))

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    _, r_d, omega_m, w0, wa = params
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
                model_smooth.append(DV_z(z, params)/(H0 * r_d))
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, params)/(H0 * r_d))
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, params))/(H0 * r_d))
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO model: $\Omega_M$={omega_m:.4f}, $w_0$={w0:.4f}, $w_a$={wa:.4f}")
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
    r_d = params[1]
    predictions = []
    for z, _, quantity in data:
        if quantity == "DV_over_rs":
            predictions.append(DV_z(z, params) / (r_d * H0))
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params) / (r_d * H0))
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params)) / (r_d * H0))
    return np.array(predictions)


bounds = np.array([
    (-0.5, 0.5), # delta M
    (115, 160), # r_d
    (0.2, 0.7), # omega_m
    (-3, 1), # w0
    (-4, 4), # wa
])


def chi_squared(params):
    delta_sn = distance_moduli_values - model_distance_modulus(z_vals, params)
    chi_sn = delta_sn @ inverse_cov_sn @ delta_sn

    delta_bao = data['value'] - model_predictions(params)
    chi_bao = delta_bao @ inv_bao_cov_matrix @ delta_bao
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
    nsteps = 5000 + burn_in
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
        [delta_M_16, delta_M_50, delta_M_84],
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [delta_M_50, rd_50, omega_50, w0_50, wa_50]

    print(f"ΔM: {delta_M_50:.4f} +{(delta_M_84 - delta_M_50):.4f} -{(delta_M_50 - delta_M_16):.4f}")
    print(f"r_d: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {data['value'].size + z_vals.size - len(best_fit)}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_distance_modulus(z_vals, best_fit),
        label=f"Best fit: $\Omega_m$={omega_50:.4f}, $w_0$={w0_50:.4f}, $w_a$={wa_50:.4f}",
        x_scale="log"
    )

    labels = [r"$\Delta_M$", r"$r_d$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
        levels=(0.393, 0.864), # 1 and 2 sigmas in 2D
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3, lw=0.4)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit[i], color='white', linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM model
ΔM: -0.0795 +0.0892 -0.0894
r_d: 144.3610 +1.0056 -1.0115
Ωm: 0.3037 +0.0085 -0.0082
w0: -1
wa: 0
Chi squared: 38.8150
Degrees of freedom: 32

=============================

Flat wCDM
ΔM: -0.0665 +0.0897 -0.0894
r_d: 141.0234 +1.5981 -1.5449
Ωm: 0.2977 +0.0093 -0.0090
w0: -0.8657 +0.0510 -0.0514 (2.62 sigma)
wa: 0
Chi squared: 32.1772
Degrees of freedom: 31

==============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.0605 +0.0885 -0.0890
r_d: 140.4293 +1.6787 -1.6629
Ωm: 0.3064 +0.0086 -0.0083
w0: -0.8298 +0.0591 -0.0598 (2.86 sigma)
wa: 0
Chi squared: 30.9553
Degrees of freedom: 31

=============================

Flat w0waCDM
ΔM: -0.0455 +0.0895 -0.0899
r_d: 139.0781 +1.8874 -1.8501
Ωm: 0.3310 +0.0154 -0.0179
w0: -1.7117 +0.4692 -0.4614 (1.53 sigma)
wa: -1.0165 +0.5654 -0.5614 (1.80 sigma)
Chi squared: 28.7904
Degrees of freedom: 30
"""