import numpy as np
import emcee
import corner
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

# Load BAO data
data = np.genfromtxt(fname="bao/raw-data/data.txt", delimiter=" ", names=True,
    dtype=[("z", float), ("value", float), ("quantity", "U10")])
cov_matrix = np.loadtxt("bao/raw-data/covariance.txt", delimiter=" ", dtype=float)
inv_cov_matrix = np.linalg.inv(cov_matrix)


def h_over_h0_model(z, params):
    _, _, _, O_m, w0, _, _ = params
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def integral_e_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)


def model_distance_modulus(z, params):
    delta_M, h0 = params[0], params[1]
    comoving_distance = (c/h0) * integral_e_z(z, params)
    return delta_M + 25 + 5 * np.log10((1 + z) * comoving_distance)


def plot_bao_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(cov_matrix))

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    _, h0, r_d, omega_m, w0, wa, _ = params
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
    h0 = params[1]
    return h0 * h_over_h0_model(z, params)


def DM_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    return cumulative_trapezoid(c / H_z(z, params), z, initial=0)[-1]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)


def model_predictions(params):
    r_d = params[2]
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
    (-0.5, 0.5), # ΔM
    (60, 80),    # H0
    (115, 160),  # r_d
    (0.2, 0.7),  # omega_m
    (-3, 0),     # w0
    (-3.5, 3.5), # wa
    (0.01, 1.5), # f - overestimation of the uncertainties in the CC data
])


def chi_squared(params):
    delta_sn = distance_moduli_values - model_distance_modulus(z_vals, params)
    chi_sn = delta_sn @ inverse_cov_sn @ delta_sn

    delta_bao = data['value'] - model_predictions(params)
    chi_bao = np.dot(delta_bao, np.dot(inv_cov_matrix, delta_bao))

    f = params[-1]
    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    escaled_error = dH_cc_vals * f
    chi_cc = np.sum(cc_delta**2 / escaled_error**2)
    return chi_sn + chi_bao + chi_cc


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[-1]
    return -0.5 * chi_squared(params) - z_cc_vals.size * np.log(f_cc)


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
        [delta_M_16, delta_M_50, delta_M_84],
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
        [f_cc_16, f_cc_50, f_cc_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [delta_M_50, h0_50, rd_50, omega_50, w0_50, wa_50, f_cc_50]
    DES5Y_EFF_SAMPLE = 1735
    deg_of_freedom = DES5Y_EFF_SAMPLE + data['value'].size + z_cc_vals.size - len(best_fit)

    print(f"ΔM: {delta_M_50:.4f} +{(delta_M_84 - delta_M_50):.4f} -{(delta_M_50 - delta_M_16):.4f}")
    print(f"H0: {h0_50:.4f} +{(h0_84 - h0_50):.4f} -{(h0_50 - h0_16):.4f}")
    print(f"r_d: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"f_cc: {f_cc_50:.4f} +{(f_cc_84 - f_cc_50):.4f} -{(f_cc_50 - f_cc_16):.4f}")
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

    labels = [r"$\Delta_M$", r"$H_0$", r"$r_d$", f"$\Omega_m$", r"$w_0$", r"$w_a$", r"$f$"]
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

    _, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3, lw=0.4)
        axes[i].set_ylabel(labels[i])
        axes[i].axvline(x=burn_in, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit[i], color='white', linestyle='--', alpha=0.5)
    axes[ndim - 1].set_xlabel("chain step")
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
ΔM: -0.0480 +0.0357 -0.0360
H0: 68.5659 +1.1699 -1.1575 km/s/Mpc
r_d: 146.6312 +2.4462 -2.3566 Mpc
Ωm: 0.3103 +0.0078 -0.0077
w0: -1
wa: 0
f_cc: 0.7065 +0.1016 -0.0832 (2.89 - 3.53 sigma)
Chi squared: 1687.7283
Degrees of freedom: 1775

==============================

Flat wCDM: w(z) = w0
ΔM: -0.0591 +0.0364 -0.0362
H0: 67.3479 +1.2105 -1.1925 km/s/Mpc
r_d: 146.9376 +2.4711 -2.4285 Mpc
Ωm: 0.2986 +0.0088 -0.0089
w0: -0.8782 +0.0385 -0.0391 (3.12 - 3.16 sigma)
wa: 0
f_cc: 0.7086 +0.1034 -0.0820 (2.82 - 3.55 sigma)
Chi squared: 1677.8143
Degrees of freedom: 1774

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.0585 +0.0363 -0.0367
H0: 67.2391 +1.2287 -1.2073 km/s/Mpc
r_d: 146.9442 +2.5193 -2.3837 Mpc
Ωm: 0.3056 +0.0080 -0.0078
w0: -0.8600 +0.0419 -0.0423 (3.31 - 3.34 sigma)
wa: 0
f_cc: 0.7116 +0.1032 -0.0834 (2.79 - 3.46 sigma)
Chi squared: 1676.6967
Degrees of freedom: 1774

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
ΔM: -0.0569 +0.0362 -0.0367
H0: 67.0657 +1.2252 -1.2250 km/s/Mpc
r_d: 147.1271 +2.4600 -2.4064 Mpc
Ωm: 0.3194 +0.0132 -0.0158
w0: -0.8035 +0.0712 -0.0669 (2.76 - 2.94 sigma)
wa: -0.6235 +0.4500 -0.4533 (1.38 - 1.39 sigma)
f_cc: 0.7116 +0.1049 -0.0836 (2.75 - 3.45 sigma)
Chi squared: 1675.5995
Degrees of freedom: 1773
"""
