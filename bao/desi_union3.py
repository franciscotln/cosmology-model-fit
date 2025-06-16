import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

c = 299792.458  # Speed of light in km/s


def Ez(z, params):
    O_m, w0 = params[3], params[4]
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * evolving_de)


grid = np.linspace(0, np.max(z_sn_vals), num=2000)


def integral_of_e_z(params):
    integral_values = cumulative_trapezoid(1 / Ez(grid, params), grid, initial=0)
    return np.interp(z_sn_vals, grid, integral_values)


def distance_modulus(params):
    dM = params[0]
    H0 = params[2]
    dL = (1 + z_sn_vals) * (c / H0) * integral_of_e_z(params)
    return dM + 25 + 5 * np.log10(dL)


def H_z(z, params):
    return params[2] * Ez(z, params)


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": lambda z, params: DV_z(z, params) / params[1],
    "DM_over_rs": lambda z, params: DM_z(z, params) / params[1],
    "DH_over_rs": lambda z, params: DH_z(z, params) / params[1],
}


def theory_predictions(z, qty, params):
    return np.array([(quantity_funcs[qty](z, params)) for z, qty in zip(z, qty)])


def chi_squared(params):
    delta_sn = mu_vals - distance_modulus(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = bao_data["value"] - theory_predictions(
        bao_data["z"], bao_data["quantity"], params
    )
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))
    return chi_sn + chi_bao


bounds = np.array(
    [
        (-0.6, 0.6),  # delta M
        (115, 160),  # r_d
        (50, 80),  # H0
        (0.2, 0.7),  # Ωm
        (-3, 1),  # w0
    ]
)


# Prior from Planck 2018 https://arxiv.org/abs/1807.06209 table 1 (Combined column)
# Ωm x ​h^2 = 0.1428 ± 0.0011. Prior width increased by 70% to 0.00187
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        Om_x_h2 = params[3] * (params[2] / 100) ** 2
        return -0.5 * ((0.1428 - Om_x_h2) / 0.00187) ** 2
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
    nwalkers = 20 * ndim
    burn_in = 200
    nsteps = 10000 + burn_in
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
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    [
        [dM_16, dM_50, dM_84],
        [rd_16, rd_50, rd_84],
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [dM_50, rd_50, H0_50, Om_50, w0_50]

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degs of freedom: {bao_data['value'].size + z_sn_vals.size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_predictions(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=distance_modulus(best_fit),
        label=f"Best fit: $\Omega_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$\Delta_M$", "$r_d$", "$H_0$", "$\Omega_m$", "$w_0$"]
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
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
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
Flat ΛCDM model
ΔM: -0.123 +0.094 -0.094 mag
r_d: 147.41 +1.53 -1.50 Mpc
H0: 68.54 +1.05 -1.04 km/s/Mpc
Ωm: 0.304 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 38.8150
Degrees of freedom: 31
Correlation matrix:
[ 1.      -0.31239  0.33648 -0.29945]
[-0.31239  1.      -0.924    0.71692]
[ 0.33648 -0.924    1.      -0.90124]
[-0.29945  0.71692 -0.90124  1.     ]

=============================

Flat wCDM
ΔM: -0.087 +0.097 -0.097 mag
r_d: 142.65 +2.52 -2.66 Mpc
H0: 69.26 +1.16 -1.14 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.866 +0.051 -0.052
wa: 0
Chi squared: 32.1546
Degs of freedom: 30
Correlation matrix:
[ 1.      -0.3343   0.37572 -0.3488   0.18234]
[-0.3343   1.      -0.80236  0.71785 -0.79907]
[ 0.37572 -0.80236  1.      -0.92003  0.33743]
[-0.3488   0.71785 -0.92003  1.      -0.36249]
[ 0.18234 -0.79907  0.33743 -0.36249  1.     ]
==============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.116 +0.094 -0.094 mag
r_d: 144.03 +1.98 -1.96 Mpc
H0: 68.26 +1.05 -1.05 km/s/Mpc
Ωm: 0.306 +0.009 -0.008
w0: -0.830 +0.059 -0.060
wa: 0
Chi squared: 30.9527
Degs of freedom: 30
Correlation matrix:
[ 1.      -0.26396  0.33963 -0.30653  0.01986]
[-0.26396  1.      -0.66668  0.51032 -0.62091]
[ 0.33963 -0.66668  1.      -0.90575 -0.10322]
[-0.30653  0.51032 -0.90575  1.       0.11762]
[ 0.01986 -0.62091 -0.10322  0.11762  1.     ]

==============================

Flat w0waCDM
ΔM: -0.182 +0.104 -0.101 mag
r_d: 148.39 +2.49 -3.16 Mpc
H0: 65.62 +1.91 -1.55 km/s/Mpc
Ωm: 0.331 +0.016 -0.018
w0: -0.697 +0.115 -0.111
wa: -1.005 +0.568 -0.562
Chi squared: 28.7899
Degs of freedom: 29
Correlation matrix:
[ 1.      -0.48338  0.52216 -0.50216 -0.34395  0.40968]
[-0.48338  1.      -0.89413  0.83896  0.47099 -0.74098]
[ 0.52216 -0.89413  1.      -0.97166 -0.73209  0.82706]
[-0.50216  0.83896 -0.97166  1.       0.76805 -0.85881]
[-0.34395  0.47099 -0.73209  0.76805  1.      -0.89365]
[ 0.40968 -0.74098  0.82706 -0.85881 -0.89365  1.     ]
"""
