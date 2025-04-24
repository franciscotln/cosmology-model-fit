import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2022pantheonSHOES.data import get_data
from y2005cc.compilation_data import get_data as get_cc_data
from hubble.plotting import plot_predictions as plot_sn_predictions

_, z_cc_vals, H_cc_vals, dH_cc_vals = get_cc_data()
legend, z_vals, apparent_mag_values, cov_matrix_sn = get_data()
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)

c = 299792.458 # Speed of light in km/s

# Load BAO data
data = np.genfromtxt(fname="bao/raw-data/data.txt", delimiter=" ", names=True,
    dtype=[("z", float), ("value", float), ("quantity", "U10")])
cov_matrix = np.loadtxt(fname="bao/raw-data/covariance.txt", delimiter=" ", dtype=float)
inv_cov_matrix = np.linalg.inv(cov_matrix)


def h_over_h0_model(z, params):
    _, _, _, O_m, w0, _ = params
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def integral_e_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)


def model_apparent_mag(z, params):
    h0 = params[0]
    M = params[1]
    comoving_distance = (c/h0) * integral_e_z(z, params)
    return M + 25 + 5 * np.log10((1 + z) * comoving_distance)


def plot_bao_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(cov_matrix))

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    h0, M, r_d, omega_m, w0, wa = params
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
    (60, 80), # H0
    (-20, -19), # M
    (115, 160), # r_d
    (0.15, 0.7), # Ωm
    (-3, 0), # w0
    (-3.5, 3.5), # wa
])


def chi_squared(params):
    delta_sn = apparent_mag_values - model_apparent_mag(z_vals, params)
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
        [h0_16, h0_50, h0_84],
        [M_16, M_50, M_84],
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, M_50, rd_50, omega_50, w0_50, wa_50]

    deg_of_freedom = z_vals.size + data['value'].size + z_cc_vals.size - len(best_fit)

    print(f"H0: {h0_50:.4f} +{(h0_84 - h0_50):.4f} -{(h0_50 - h0_16):.4f}")
    print(f"M: {M_50:.4f} +{(M_84 - M_50):.4f} -{(M_50 - M_16):.4f}")
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
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_apparent_mag(z_vals, best_fit),
        label=f"Best fit: $\Omega_m$={omega_50:.4f}, $w_0$={w0_50:.4f}, $w_a$={wa_50:.4f}",
        x_scale="log"
    )

    labels = [r"$H_0$", r"$M$", r"$r_d$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]
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
Flat ΛCDM: w(z) = -1
h0: 67.2600 +1.0534 -1.0707
M: -19.4477 +0.0328 -0.0340
r_d: 150.5823 +2.1687 -2.0875
Ωm: 0.3008 +0.0076 -0.0074
w0: -1
wa: 0
Chi squared: 1442.3460
Degrees of freedom: 1636

==============================

Flat wCDM: w(z) = w0
H0: 66.4920 +1.0777 -1.0959
M: -19.4586 +0.0332 -0.0340
r_d: 150.1105 +2.2099 -2.1298
Ωm: 0.2937 +0.0082 -0.0081
w0: -0.9017 +0.0384 -0.0384 (2.56 sigma)
wa: 0
Chi squared: 1436.0292
Degrees of freedom: 1635

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 66.4188 +1.1212 -1.1135
M: -19.4591 +0.0339 -0.0344
r_d: 150.2107 +2.1980 -2.1258
Ωm: 0.2991 +0.0076 -0.0074
w0: -0.8912 +0.0419 -0.0432 (2.56 sigma)
wa: 0
Chi squared: 1436.0294
Degrees of freedom: 1635

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
H0: 66.4728 +1.1386 -1.1354 km/s/Mpc
M: -19.4588 +0.0339 -0.0346
r_d: 150.1233 +2.2526 -2.1015 Mpc
Ωm: 0.2932 +0.0169 -0.0293
w0: -0.9013 +0.4463 -0.3917 (0.24 sigma)
wa: 0.0008 +0.4736 -0.4398 (0.00 sigma)
Chi squared: 1436.1274
Degrees of freedom: 1634
"""
