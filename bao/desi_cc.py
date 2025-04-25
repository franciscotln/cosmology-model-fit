import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2005cc.compilation_data import get_data as get_cc_data

_, z_cc_vals, H_cc_vals, dH_cc_vals = get_cc_data()

c = 299792.458 # Speed of light in km/s

data = np.genfromtxt(fname="bao/raw-data/data.txt", delimiter=" ", names=True,
    dtype=[("z", float), ("value", float), ("quantity", "U10")])
bao_cov_matrix = np.loadtxt("bao/raw-data/covariance.txt", delimiter=" ", dtype=float)
inv_bao_cov_matrix = np.linalg.inv(bao_cov_matrix)


def h_over_h0_model(z, params):
    _, _, O_m, w0, _, _ = params
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def H_z(z, params):
    h0 = params[0]
    return h0 * h_over_h0_model(z, params)


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
            predictions.append(DV_z(z, params)/r_d)
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params)/r_d)
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params))/r_d)
    return np.array(predictions)


bounds = np.array([
    (50, 100),   # H0
    (133, 160),  # r_d
    (0.2, 0.7),  # Ωm
    (-2, 0.5),   # w0
    (-4, 4),     # wa
    (0.01, 1.5), # f - overestimation of the uncertainties in the CC data
])


def chi_squared(params):
    f = params[-1]
    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    escaled_error = dH_cc_vals * f
    chi_cc = np.sum(cc_delta**2 / escaled_error**2)

    delta_bao = data['value'] - model_predictions(params)
    chi_bao = np.dot(delta_bao, np.dot(inv_bao_cov_matrix, delta_bao))
    return chi_cc + chi_bao


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[-1]
    return -0.5 * chi_squared(params) - f_cc * z_cc_vals.size


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def plot_all_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(bao_cov_matrix))

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    h0, r_d, omega_m, w0, wa, _ = params
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
    plt.title(f"BAO model: $H_0$={h0:.2f}, $r_d$={r_d:.2f}, $\Omega_M$={omega_m:.4f}, $w_0$={w0:.4f}, $w_a$={wa:.4f}")
    plt.show()

    # Plot Hubble parameter H(z)
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
    plt.ylabel(r"$H(z)$ - km/s/Mpc")
    plt.xlim(0, np.max(z_cc_vals) + 0.2)
    plt.legend()
    plt.title(f"Cosmic chronometers: $H_0$={h0:.2f}")
    plt.show()


def main():
    ndim = len(bounds)
    nwalkers = 100
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
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
        [f_16, f_50, f_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, rd_50, omega_50, w0_50, wa_50, f_50]

    print(f"H0: {h0_50:.2f} +{(h0_84 - h0_50):.2f} -{(h0_50 - h0_16):.2f}")
    print(f"r_d: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"f: {f_50:.4f} +{(f_84 - f_50):.4f} -{(f_50 - f_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {data['value'].size + z_cc_vals.size - len(best_fit)}")

    plot_all_predictions(best_fit)

    labels = [r"$H_0$", r"$r_d$", f"$\Omega_m$", r"$w_0$", r"$w_a$", r"$f$"]
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
Flat ΛCDM model
H0: 67.49 +0.90 -0.91 km/s/Mpc
r_d: 151.1200 +1.6649 -1.6242 Mpc
Ωm: 0.2918 +0.0080 -0.0077
w0: -1
wa: 0
f: 0.8125 +0.0891 -0.0794 (2.10 - 2.36 sigma)
Chi squared: 41.4501
Degrees of freedom: 50

=============================

Flat wCDM model
H0: 66.13 +1.28 -1.24 km/s/Mpc
r_d: 150.9017 +1.6711 -1.6414 Mpc
Ωm: 0.2920 +0.0082 -0.0080
w0: -0.8966 +0.0692 -0.0726 (1.42 - 1.49 sigma)
wa: 0
f: 0.8056 +0.0891 -0.0771 (2.18 - 2.52 sigma)
Chi squared: 39.9010
Degrees of freedom: 49

==============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 65.69 +1.46 -1.40 km/s/Mpc
r_d: 150.9630 +1.6671 -1.6422 Mpc
Ωm: 0.2998 +0.0096 -0.0095
w0: -0.8563 +0.0889 -0.0943 (1.52 - 1.62 sigma)
wa: 0
f: 0.8068 +0.0885 -0.0779 (2.18 - 2.48 sigma)
Chi squared: 39.4676
Degrees of freedom: 49

==============================

Flat w0waCDM w0 + wa * z/(1 + z)
H0: 62.62 +3.23 -2.93 km/s/Mpc
r_d: 151.7635 +1.8577 -1.8058 Mpc
Ωm: 0.3467 +0.0396 -0.0467
w0: -0.5077 +0.3557 -0.3426 (1.38 - 1.44 sigma)
wa: -1.5623 +1.3438 -1.2719 (1.16 - 1.23 sigma)
f: 0.8191 +0.0908 -0.0805 (1.99 - 2.25 sigma)
Chi squared: 37.3247
Degrees of freedom: 48
"""
