import numpy as np
import emcee
from getdist import MCSamples, plots
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
    O_m, w0 = params[3], params[4]
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


z_grid = np.linspace(0, np.max(z_vals), num=3000)

def integral_e_z(params):
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z_grid, params), z_grid, initial=0)
    return np.interp(z_vals, z_grid, integral_values)


def model_distance_modulus(params):
    delta_M, h0 = params[0], params[1]
    return delta_M + 25 + 5 * np.log10((1 + z_vals) * (c / h0) * integral_e_z(params))


def plot_bao_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(cov_matrix))

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    h0, r_d, omega_m, f = params[1], params[2], params[3], params[-1]
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
    plt.title(f"BAO model: $H_0$={h0:.2f} $r_d$={r_d:.2f}, $\Omega_M$={omega_m:.4f}")
    plt.show()

    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=dH_cc_vals * f,
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
    (0.2, 0.7),  # Ωm
    (-3, 0),     # w0
    (-3.5, 3.5), # wa
    (0.1, 1.5),  # f - overestimation of the uncertainties in the CC data
])


def chi_squared(params):
    delta_sn = distance_moduli_values - model_distance_modulus(params)
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
    nwalkers = 8 * ndim
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
        y_model=model_distance_modulus(best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f}, $\Omega_m$={omega_50:.4f}",
        x_scale="log"
    )

    labels = ["ΔM", "H_0", "r_d", "Ωm", "w_0", "w_a", "f"]
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
*******************************
Here we are considering the uncertainties to be overestimated
in the CC data. We then use the parameter f to account for this.
The results show consistent values for f and the corner plot shows
that the parameters are weakly correlated
*******************************

Flat ΛCDM: w(z) = -1
ΔM: -0.048 +0.036 -0.037 mag
H0: 68.56 +1.17 -1.19 km/s/Mpc
r_d: 146.67 +2.49 -2.37 Mpc
Ωm: 0.310 +0.008 -0.008
w0: -1
wa: 0
f_cc: 0.706 +0.103 -0.084 (2.85 - 3.50 sigma)
Chi squared: 1687.7647
Degrees of freedom: 1775

==============================

Flat wCDM: w(z) = w0
ΔM: -0.058 +0.036 -0.037 mag
H0: 67.36 +1.22 -1.20 km/s/Mpc
r_d: 146.90 +2.48 -2.38 Mpc
Ωm: 0.299 +0.009 -0.009
w0: -0.879 +0.038 -0.039 (3.10 - 3.18 sigma)
wa: 0
f_cc: 0.707 +0.102 -0.082 (2.87 - 3.57 sigma)
Chi squared: 1677.9407
Degrees of freedom: 1774

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.058 +0.036 -0.036 mag
H0: 67.25 +1.23 -1.20 km/s/Mpc
r_d: 146.93 +2.47 -2.40 Mpc
Ωm: 0.306 +0.008 -0.008
w0: -0.860 +0.042 -0.043 (3.26 - 3.33 sigma)
wa: 0
f_cc: 0.709 +0.104 -0.083 (2.80 - 3.51 sigma)
Chi squared: 1676.9063
Degrees of freedom: 1774

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
ΔM: -0.056 +0.037 -0.036 mag
H0: 67.11 +1.23 -1.22 km/s/Mpc
r_d: 147.03 +2.48 -2.43 Mpc
Ωm: 0.319 +0.014 -0.017
w0: -0.806 +0.073 -0.066 (2.66 - 2.94 sigma)
wa: -0.612 +0.474 -0.462 (1.29 - 1.32 sigma)
f_cc: 0.713 +0.104 -0.084 (2.76 - 3.40 sigma)
Chi squared: 1675.4677
Degrees of freedom: 1773
"""
