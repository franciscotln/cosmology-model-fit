import numpy as np
import emcee
from getdist import MCSamples, plots
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

_, z_cc_vals, H_cc_vals, dH_cc_vals = get_cc_data()
_, data, bao_cov_matrix = get_bao_data()
inv_bao_cov_matrix = np.linalg.inv(bao_cov_matrix)

c = 299792.458 # Speed of light in km/s

def h_over_h0_model(z, params):
    O_m, w0 = params[2], params[3]
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

    h0, r_d, omega_m, f_cc = params[0], params[1], params[2], params[-1]
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
    plt.title(f"BAO model: $H_0$={h0:.2f}, $r_d$={r_d:.2f}, $\Omega_M$={omega_m:.4f}")
    plt.show()

    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=dH_cc_vals * f_cc,
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
    plt.title(f"Cosmic chronometers: $H_0$={h0:.2f} km/s/Mpc")
    plt.show()


def main():
    ndim = len(bounds)
    nwalkers = 6 * ndim
    burn_in = 500
    nsteps = 20000 + burn_in
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

    labels = ["H_0", "r_d", "Omega_m", "w_0", "w_a", "f"]
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
Flat ΛCDM model
H0: 69.07 +1.33 -1.32 km/s/Mpc
r_d: 146.9059 +2.7570 -2.6723 Mpc
Ωm: 0.2984 +0.0086 -0.0083
f: 0.7950 +0.1001 -0.0856 (2.05 - 2.39 sigma)
Chi squared: 33.4751
Degrees of freedom: 41

=============================

Flat wCDM model
H0: 67.97 +1.72 -1.66 km/s/Mpc
r_d: 146.9063 +2.8136 -2.7026 Mpc
Ωm: 0.2976 +0.0087 -0.0086
w0: -0.9195 +0.0748 -0.0783 (1.03 - 1.08 sigma)
wa: 0
f: 0.7976 +0.1008 -0.0868 (2.01 - 2.33 sigma)
Chi squared: 32.2325
Degrees of freedom: 40

==============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 67.60 +1.89 -1.82 km/s/Mpc
r_d: 146.9091 +2.7760 -2.6747 Mpc
Ωm: 0.3038 +0.0101 -0.0098
w0: -0.8841 +0.0982 -0.1052 (1.10 - 1.18 sigma)
wa: 0
f: 0.8005 +0.1007 -0.0875 (1.98 - 2.28 sigma)
Chi squared: 31.7523
Degrees of freedom: 40

==============================

Flat w0waCDM w0 + wa * z/(1 + z)
H0: 64.34 +3.48 -3.02 km/s/Mpc
r_d: 147.2896 +2.8466 -2.7648 Mpc
Ωm: 0.3537 +0.0392 -0.0469
w0: -0.4973 +0.3602 -0.3679
wa: -1.6501 +1.3950 -1.2771
f: 0.8080 +0.1021 -0.0870 (1.88 - 2.21 sigma)
Chi squared: 29.8468
Degrees of freedom: 39
"""
