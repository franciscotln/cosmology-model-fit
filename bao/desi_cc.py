import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2005cc.data import get_data as get_cc_data

_, z_cc_vals, H_cc_vals, dH_cc_vals = get_cc_data()

c = 299792.458 # Speed of light in km/s

data = np.genfromtxt(fname="bao/raw-data/data.txt", delimiter=" ", names=True,
    dtype=[("z", float), ("value", float), ("quantity", "U10")])
bao_cov_matrix = np.loadtxt("bao/raw-data/covariance.txt", delimiter=" ", dtype=float)
inv_bao_cov_matrix = np.linalg.inv(bao_cov_matrix)


def w_de(z, params):
    _, _, _, w0, wa = params
    return wa + (w0 - wa) * np.exp(0.5 - 0.5*(1 + z)**2)


def rho_de(zs, params):
    z = np.linspace(0, np.max(zs), num=2000)
    integral_values = cumulative_trapezoid(3*(1 + w_de(z, params))/(1 + z), z, initial=0)
    return np.exp(np.interp(zs, z, integral_values))


def h_over_h0_model(z, params):
    O_m = params[2]
    return np.sqrt(O_m * (1 + z)**3 + (1 - O_m) * rho_de(z, params))


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
    (50, 100), # H0
    (133, 160), # r_d
    (0.2, 0.7), # omega_m
    (-2, 0.5), # w0
    (-4, 0), # wa
])


def chi_squared(params):
    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.sum(cc_delta**2 / dH_cc_vals**2)

    delta_bao = data['value'] - model_predictions(params)
    chi_bao = np.dot(delta_bao, np.dot(inv_bao_cov_matrix, delta_bao))
    return chi_cc + chi_bao


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


def plot_all_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(bao_cov_matrix))

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
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, rd_50, omega_50, w0_50, wa_50]

    print(f"H0: {h0_50:.2f} +{(h0_84 - h0_50):.2f} -{(h0_50 - h0_16):.2f}")
    print(f"r_d: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {data['value'].size + z_cc_vals.size - len(best_fit)}")

    plot_all_predictions(best_fit)

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
Flat ΛCDM model
H0: 69.09 +1.63 -1.63
r_d: 146.9092 +3.4305 -3.3268
Ωm: 0.2980 +0.0086 -0.0083
w0: -1
wa: 0
Chi squared: 24.9175
Degrees of freedom: 41

=============================

Flat wCDM model
H0: 67.89 +1.97 -1.89
r_d: 146.9806 +3.4397 -3.3019
Ωm: 0.2974 +0.0090 -0.0087
w0: -0.9162 +0.0761 -0.0792
wa: 0
Chi squared: 23.8128
Degrees of freedom: 40

==============================

Flat w0waCDM w0 + wa * z/(1 + z)
H0: 63.49 +3.25 -2.84
r_d: 147.4375 +3.4612 -3.3443
Ωm: 0.3647 +0.0356 -0.0415
w0: -0.3944 +0.3308 -0.3530
wa: -2.0051 +1.2815 -1.1622
Chi squared: 21.5170
Degrees of freedom: 39

==============================

Flat wa + (w0 - wa) * np.exp(-0.5*((1 + z)**2 - 1))
H0: 63.93 +3.37 -3.16
r_d: 147.3750 +3.4912 -3.3300
Ωm: 0.3596 +0.0401 -0.0436
w0: -0.5007 +0.3310 -0.3176
wa: -1.6810 +0.5453 -0.5318
Chi squared: 21.4762
Degrees of freedom: 39
"""
