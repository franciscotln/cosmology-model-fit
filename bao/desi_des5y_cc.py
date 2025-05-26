import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, z_sn_hel_vals, distance_moduli_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

inverse_cov_cc = np.linalg.inv(cov_matrix_cc)
cho_sn = cho_factor(cov_matrix_sn)
inverse_cov_bao = np.linalg.inv(cov_matrix_bao)

c = 299792.458 # Speed of light in km/s

z_grid_sn = np.linspace(0, np.max(z_sn_vals), num=3000)


def h_over_h0_model(z, p):
    sum = 1 + z
    return np.sqrt(p['Omega_m'] * sum**3 + (1 - p['Omega_m']) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + p['w_0'])))


def integral_e_z(params):
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z_grid_sn, params), z_grid_sn, initial=0)
    return np.interp(z_sn_vals, z_grid_sn, integral_values)


def model_distance_modulus(p):
    return p['Delta_M'] + 25 + 5 * np.log10((1 + z_sn_hel_vals) * (c / p['H_0']) * integral_e_z(p))


def plot_bao_predictions(p):
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }
    z_smooth = np.linspace(0, max(bao_data["z"]), 100)

    plt.figure(figsize=(8, 6))
    for q in set(bao_data["quantity"]):
        mask = bao_data["quantity"] == q
        plt.errorbar(
            x=bao_data["z"][mask],
            y=bao_data["value"][mask],
            yerr=np.sqrt(np.diag(cov_matrix_bao))[mask],
            fmt='.',
            color=colors[q],
            label=f"BAO {q}",
            capsize=2,
            alpha=0.6,
        )
        model_smooth = []
        for z in z_smooth:
            if q == "DV_over_rs":
                model_smooth.append(DV_z(z, p)/p['r_d'])
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, p)/p['r_d'])
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, p))/p['r_d'])
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.6)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"{bao_legend}: $H_0$={p['H_0']:.2f} km/s/Mpc, $r_d$={p['r_d']:.2f} Mpc, $\Omega_M$={p['Omega_m']:.3f}")
    plt.show()


def plot_cc_predictions(p):
    z_smooth = np.linspace(0, np.max(z_cc_vals), 100)

    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=np.sqrt(np.diag(cov_matrix_cc)),
        fmt='.',
        color='blue',
        alpha=0.4,
        label="CCH data",
        capsize=2,
    )
    plt.plot(z_smooth, H_z(z_smooth, p), color='red', alpha=0.5, label="Model")
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z_cc_vals) + 0.2)
    plt.legend()
    plt.title(f"{cc_legend}: $H_0$={p['H_0']:.2f} km/s/Mpc")
    plt.show()


def H_z(z, p):
    return p['H_0'] * h_over_h0_model(z, p)


def DM_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    return cumulative_trapezoid(c / H_z(z, params), z, initial=0)[-1]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)


def bao_predictions(p):
    predictions = []
    for z, _, quantity in bao_data:
        if quantity == 'DV_over_rs':
            predictions.append(DV_z(z, p) / p['r_d'])
        elif quantity == 'DM_over_rs':
            predictions.append(DM_z(z, p) / p['r_d'])
        elif quantity == 'DH_over_rs':
            predictions.append((c / H_z(z, p)) / p['r_d'])
    return np.array(predictions)


param_bounds = {
    "Delta_M": (-0.55, 0.55),
    "H_0": (50, 80),
    "r_d": (110, 175),
    "Omega_m": (0.2, 0.7),
    "w_0": (-1.1, -0.4),
}

param_names = list(param_bounds.keys())


def array_to_dict(param_array):
    return dict(zip(param_names, param_array))


def chi_squared(params):
    delta_sn = distance_moduli_values - model_distance_modulus(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = bao_data['value'] - bao_predictions(params)
    chi_bao = np.dot(delta_bao, np.dot(inverse_cov_bao, delta_bao))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, np.dot(inverse_cov_cc, delta_cc))
    return chi_sn + chi_bao + chi_cc


def log_prior(param_array):
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        if not (low < param_array[i] < high):
            return -np.inf
    return 0.0


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(param_array):
    lp = log_prior(param_array)
    if not np.isfinite(lp):
        return -np.inf
    return lp - 0.5 * chi_squared(array_to_dict(param_array))


def main():
    ndim = len(param_names)
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 10000 + burn_in
    initial_pos = np.random.default_rng().uniform(
        [param_bounds[name][0] for name in param_names],
        [param_bounds[name][1] for name in param_names],
        size=(nwalkers, ndim)
    )

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True, thin=1)

    percentiles = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    summary = {}
    for name, (p16, p50, p84) in zip(param_names, percentiles):
        summary[name] = (p50, p84 - p50, p50 - p16)
        print(f"{name}: {p50:.3f} +{p84 - p50:.3f} -{p50 - p16:.3f}")

    best_fit_dict = array_to_dict([entry[0] for entry in summary.values()])
    DES5Y_EFF_SAMPLE = 1735
    deg_of_freedom = DES5Y_EFF_SAMPLE + bao_data['value'].size + z_cc_vals.size - ndim

    print(f"Chi squared: {chi_squared(best_fit_dict):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(best_fit_dict)
    plot_cc_predictions(best_fit_dict)
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_distance_modulus(best_fit_dict),
        label=fr"Best fit: $H_0$={summary['H_0'][0]:.2f} km/s/Mpc, $\Omega_m$={summary['Omega_m'][0]:.3f}",
        x_scale="log"
    )

    corner.corner(
        samples,
        labels=[f'${name}$' for name in param_names],
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864), # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
Delta_M: -0.059 +0.103 -0.106
H_0: 68.26 +3.35 -3.26 km/s/Mpc
r_d: 147.29 +7.32 -6.79
Omega_m: 0.310 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1673.52
Degrees of freedom: 1776

==============================

Flat wCDM: w(z) = w0
ΔM: -0.061 +0.101 -0.105 mag
H0: 67.28 +3.24 -3.22 km/s/Mpc
r_d: 147.08 +7.25 -6.67 Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.877 +0.038 -0.038 (3.2 sigma)
wa: 0
Chi squared: 1663.20
Degrees of freedom: 1775

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.063 +0.101 -0.107 mag
H0: 67.06 +3.24 -3.23 km/s/Mpc
r_d: 147.20 +7.40 -6.69 Mpc
Ωm: 0.306 +0.008 -0.008
w0: -0.851 +0.041 -0.043 (3.47 - 3.63 sigma)
wa: 0
Chi squared: 1661.80
Degrees of freedom: 1775

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
ΔM: -0.056 +0.101 -0.107 mag
H0: 67.05 +3.25 -3.25 km/s/Mpc
r_d: 147.09 +7.39 -6.68 Mpc
Ωm: 0.319 +0.013 -0.017
w0: -0.800 +0.074 -0.067 (2.7 - 3.0 sigma)
wa: -0.6385 +0.4721 -0.4662
Chi squared: 1661.08
Degrees of freedom: 1774
"""
