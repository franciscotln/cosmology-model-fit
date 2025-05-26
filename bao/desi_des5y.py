import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions

legend, z_vals, z_hel_vals, distance_moduli_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)
_, data, cov_matrix = get_bao_data()
inv_cov_matrix = np.linalg.inv(cov_matrix)

c = 299792.458  # Speed of light in km/s
H0 = 70  # Hubble constant in km/s/Mpc as per DES5Y


def h_over_h0_model(z, params):
    O_m, w0 = params[2], params[3]
    sum = 1 + z
    evolving_de = sum ** (3 * (1 + w0))  # ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0))
    return np.sqrt(O_m * sum**3 + (1 - O_m) * evolving_de)


def integral_of_e_z(params):
    z = np.linspace(0, np.max(z_vals), num=2000)
    integral_values = cumulative_trapezoid(1 / h_over_h0_model(z, params), z, initial=0)
    return np.interp(z_vals, z, integral_values)


def model_distance_modulus(params):
    delta_M = params[0]
    comoving_distance = (c / H0) * integral_of_e_z(params)
    return delta_M + 25 + 5 * np.log10((1 + z_hel_vals) * comoving_distance)


def plot_bao_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(cov_matrix))

    unique_quantities = set(quantity_types)
    colors = {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"}

    r_d, omega_m = params[1], params[2]
    z_smooth = np.linspace(0, max(z_values), 100)
    plt.figure(figsize=(8, 6))
    for q in unique_quantities:
        mask = quantity_types == q
        plt.errorbar(
            x=z_values[mask],
            y=observed_values[mask],
            yerr=errors[mask],
            fmt=".",
            color=colors[q],
            label=f"Data: {q}",
            capsize=2,
            linestyle="None",
        )
        model_smooth = []
        for z in z_smooth:
            if q == "DV_over_rs":
                model_smooth.append(DV_z(z, params) / (H0 * r_d))
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, params) / (H0 * r_d))
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, params)) / (H0 * r_d))
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO model: $\Omega_M$={omega_m:.4f}")
    plt.show()


def H_z(z, params):
    return h_over_h0_model(z, params)


def DM_z(zs, params):
    return quad(lambda z: c / H_z(z, params), 0, zs)[0]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


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


bounds = np.array(
    [
        (-0.5, 0.5),  # delta M
        (115, 160),  # r_d
        (0.1, 0.7),  # omega_m
        (-3, 0),  # w0
    ]
)


def chi_squared(params):
    delta_sn = distance_moduli_values - model_distance_modulus(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = data["value"] - model_predictions(params)
    chi_bao = np.dot(delta_bao, np.dot(inv_cov_matrix, delta_bao))
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
    nwalkers = 8 * ndim
    burn_in = 500
    nsteps = 8000 + burn_in
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
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [delta_M_50, rd_50, omega_50, w0_50]

    print(
        f"ΔM: {delta_M_50:.4f} +{(delta_M_84 - delta_M_50):.4f} -{(delta_M_50 - delta_M_16):.4f}"
    )
    print(f"r_d: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(
        f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}"
    )
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {data['value'].size + z_vals.size - len(best_fit)}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_distance_modulus(best_fit),
        label=f"Best fit: $\Omega_m$={omega_50:.4f}, $w_0$={w0_50:.4f}",
        x_scale="log",
    )

    labels = ["$\Delta_M$", "$r_d$", "$\Omega_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3, lw=0.4)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
ΔM: -0.004 +0.006 -0.006
r_d: 143.62 +0.93 -0.94
Ωm: 0.310 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1658.9670
Degrees of freedom: 1839

==============================

Flat wCDM
ΔM: 0.026 +0.011 -0.011
r_d: 141.19 +1.19 -1.15
Ωm: 0.298 +0.009 -0.009
w0: -0.872 +0.039 -0.039
wa: 0
Chi squared: 1648.1021
Degrees of freedom: 1838

===============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: 0.030 +0.012 -0.012
r_d: 140.99 +1.18 -1.18
Ωm: 0.305 +0.008 -0.0080
w0: -0.849 +0.042 -0.043
wa: 0
Chi squared: 1646.9797
Degrees of freedom: 1838

==============================

Flat w0waCDM
ΔM: 0.0374 +0.0138 -0.0140
r_d: 140.8937 +1.2340 -1.1907 Mpc
Ωm: 0.3199 +0.0132 -0.0167
w0: -0.7942 +0.0733 -0.0685 (2.9 sigma)
wa: -0.6713 +0.4795 -0.4728 (1.41 sigma)
Chi squared: 1646.1246
Degrees of freedom: 1837
"""
