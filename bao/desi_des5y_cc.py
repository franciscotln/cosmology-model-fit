import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import effective_sample_size as sn_sample, get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions
from cosmic_chronometers.plot_predictions import plot_cc_predictions
from .plot_predictions import plot_bao_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, z_sn_hel_vals, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(cov_matrix_bao)
inv_cov_cc = np.linalg.inv(cov_matrix_cc)
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = 299792.458  # Speed of light in km/s

z_grid_sn = np.linspace(0, np.max(z_sn_vals), num=3000)


def Ez(z, p):
    one_plus_z = 1 + z
    rho_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + p["w_0"]))
    return np.sqrt(p["Omega_m"] * one_plus_z**3 + (1 - p["Omega_m"]) * rho_de)


def integral_Ez(params):
    y = 1 / Ez(z_grid_sn, params)
    integral_values = cumulative_trapezoid(y=y, x=z_grid_sn, initial=0)
    return np.interp(z_sn_vals, z_grid_sn, integral_values)


def mu_theory(p):
    return (
        p["Delta_M"]
        + 25
        + 5 * np.log10((1 + z_sn_hel_vals) * (c / p["H_0"]) * integral_Ez(p))
    )


def H_z(z, p):
    return p["H_0"] * Ez(z, p)


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: c / H_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


bao_quantity_funcs = {
    "DV_over_rs": lambda z, p: DV_z(z, p) / p["r_d"],
    "DM_over_rs": lambda z, p: DM_z(z, p) / p["r_d"],
    "DH_over_rs": lambda z, p: DH_z(z, p) / p["r_d"],
}


def theory_predictions(z, qty, params):
    return np.array([(bao_quantity_funcs[qty](z, params)) for z, qty in zip(z, qty)])


param_bounds = {
    "f_cc": (0.4, 2.5),
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
    delta_sn = mu_values - mu_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = bao_data["value"] - theory_predictions(
        bao_data["z"], bao_data["quantity"], params
    )
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, np.dot(inv_cov_cc * params["f_cc"] ** 2, delta_cc))
    return chi_sn + chi_bao + chi_cc


def log_prior(param_array):
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        if not (low < param_array[i] < high):
            return -np.inf
    return 0.0


def log_likelihood(params):
    f_cc = params["f_cc"]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(param_array):
    lp = log_prior(param_array)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(array_to_dict(param_array))


def main():
    ndim = len(param_names)
    nwalkers = 10 * ndim
    burn_in = 500
    nsteps = 10000 + burn_in
    initial_pos = np.random.uniform(
        [param_bounds[name][0] for name in param_names],
        [param_bounds[name][1] for name in param_names],
        size=(nwalkers, ndim),
    )

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    print("correlation matrix:")
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    percentiles = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    summary = {}
    for name, (p16, p50, p84) in zip(param_names, percentiles):
        summary[name] = (p50, p84 - p50, p50 - p16)
        print(f"{name}: {p50:.3f} +{p84 - p50:.3f} -{p50 - p16:.3f}")

    best_fit_dict = array_to_dict([entry[0] for entry in summary.values()])
    deg_of_freedom = sn_sample + bao_data["value"].size + z_cc_vals.size - ndim

    print(f"Chi squared: {chi_squared(best_fit_dict):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_predictions(z, qty, best_fit_dict),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=f"{bao_legend}: $r_d$={best_fit_dict['r_d']:.1f} Mpc",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit_dict),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / best_fit_dict["f_cc"],
        label=f"{cc_legend} $H_0$: {best_fit_dict['H_0']:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit_dict),
        label=rf"Best fit: $H_0$={summary['H_0'][0]:.1f} km/s/Mpc, $\Omega_m$={summary['Omega_m'][0]:.3f}",
        x_scale="log",
    )

    corner.corner(
        samples,
        labels=[f"${name}$" for name in param_names],
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
f_cc: 1.470 +0.190 -0.182
ΔM: -0.058 +0.072 -0.072 mag
H0: 68.3 +2.3 -2.3 km/s/Mpc
r_d: 147.2 +4.9 -4.7 Mpc
Ωm: 0.311 +0.008 -0.008
w0: -1
Chi squared: 1690.38
Degrees of freedom: 1775
Correlation matrix:
[[ 1.       0.01129  0.00657 -0.01832  0.01085]
 [ 0.01129  1.       0.99586 -0.98895 -0.15224]
 [ 0.00657  0.99586  1.      -0.98028 -0.21788]
 [-0.01832 -0.98895 -0.98028  1.       0.04513]
 [ 0.01085 -0.15224 -0.21788  0.04513  1.     ]]

==============================

Flat wCDM: w(z) = w0
f_cc: 1.456 +0.187 -0.178
ΔM: -0.064 +0.071 -0.073 mag
H0: 67.2 +2.3 -2.3 km/s/Mpc
r_d: 147.2 +5.0 -4.7 Mpc
Ωm: 0.299 +0.009 -0.009
w0: -0.875 +0.038 -0.038 (3.29 sigma from -1)
Chi squared: 1679.35
Degrees of freedom: 1774
Correlation matrix:
[[ 1.       0.01496  0.01328 -0.01863  0.0155  -0.02358]
 [ 0.01496  1.       0.98923 -0.98865 -0.11032 -0.03967]
 [ 0.01328  0.98923  1.      -0.96995 -0.10425 -0.15915]
 [-0.01863 -0.98865 -0.96995  1.       0.03661  0.00123]
 [ 0.0155  -0.11032 -0.10425  0.03661  1.      -0.46128]
 [-0.02358 -0.03967 -0.15915  0.00123 -0.46128  1.     ]]

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
f_cc: 1.454 +0.187 -0.179
ΔM: -0.062 +0.071 -0.072 mag
H0: 67.1 +2.3 -2.2 km/s/Mpc
r_d: 147.1 +5.0 -4.6 Mpc
Ωm: 0.306 +0.008 -0.008
w0: -0.854 +0.042 -0.042 (3.48 sigma from -1)
Chi squared: 1678.30
Degrees of freedom: 1774
Correlation matrix:
[[ 1.       0.01182  0.00772 -0.01766  0.01974 -0.0115 ]
 [ 0.01182  1.       0.98738 -0.98857 -0.14577 -0.02838]
 [ 0.00772  0.98738  1.      -0.96839 -0.17764 -0.16215]
 [-0.01766 -0.98857 -0.96839  1.       0.04817 -0.00327]
 [ 0.01974 -0.14577 -0.17764  0.04817  1.      -0.1664 ]
 [-0.0115  -0.02838 -0.16215 -0.00327 -0.1664   1.     ]]
"""
