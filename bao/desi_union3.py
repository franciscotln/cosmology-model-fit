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
    O_m, w0 = params[2], params[3]
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * rho_de)


grid = np.linspace(0, np.max(z_sn_vals), num=2000)


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(grid, params), grid, initial=0)
    return np.interp(z_sn_vals, grid, integral_values)


def distance_modulus(params):
    dL = (1 + z_sn_vals) * c * integral_Ez(params)
    return params[0] + 25 + 5 * np.log10(dL)


def H_z(z, params):
    return Ez(z, params)


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": lambda z, params: DV_z(z, params) / (params[1] * 100),
    "DM_over_rs": lambda z, params: DM_z(z, params) / (params[1] * 100),
    "DH_over_rs": lambda z, params: DH_z(z, params) / (params[1] * 100),
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
        (-10, -8.5),  # ΔM
        (90, 110),  # r_d * h
        (0.1, 0.6),  # Ωm
        (-2, 0),  # w0
    ]
)


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0
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
    nwalkers = 16 * ndim
    burn_in = 200
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

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    [
        [dM_16, dM_50, dM_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [dM_50, rd_50, Om_50, w0_50]

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d * h: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
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
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$Δ_M$", "$r_d x h$", "$Ω_m$", "$w_0$"]
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
******************************
DESI BAO DR2 2025
******************************

Flat ΛCDM
ΔM: -9.304 +0.088 -0.089 mag
r_d * h: 101.04 +0.71 -0.71 Mpc
Ωm: 0.304 +0.008 -0.008
w0: -1
Chi squared: 38.8155
Degs of freedom: 32
Correlation matrix:
[[ 1.      -0.00769  0.01009]
 [-0.00769  1.      -0.9176]
 [0.01009 -0.9176    1.    ]]

=============================

Flat wCDM
ΔM: -9.291 +0.089 -0.089 mag
r_d * h: 98.71 +1.11 -1.09 Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.865 +0.051 -0.052
Chi squared: 32.1544 (Δ chi2 6.66)
Degs of freedom: 31
Correlation matrix:
[[ 1.      -0.04961 -0.00521  0.05168]
 [-0.04961  1.      -0.19855 -0.81299]
 [-0.00521 -0.19855  1.      -0.35804]
 [ 0.05168 -0.81299 -0.35804  1.     ]]

==============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
ΔM: -9.282 +0.089 -0.089 mag
r_d * h: 98.05 +1.22 -1.20 Mpc
Ωm: 0.310 +0.009 -0.009
w0: -0.802 +0.066 -0.067
Chi squared: 30.3680 (Δ chi2 8.45)
Degs of freedom: 31
Correlation matrix:
[[ 1.      -0.06035  0.02289  0.0696 ]
 [-0.06035  1.      -0.66885 -0.85462]
 [ 0.02289 -0.66885  1.       0.25292]
 [ 0.0696  -0.85462  0.25292  1.     ]]

******************************
SDSS BAO DR16 compilation 2020
******************************

Flat ΛCDM
ΔM: -9.301 +0.088 -0.088 mag
r_d * h: 100.15 +0.96 -0.95 Mpc
Ωm: 0.313 +0.015 -0.014
w0: -1
Chi squared: 39.8737
Degs of freedom: 36

==============================

Flat wCDM
ΔM: -9.285 +0.089 -0.089 mag
r_d * h: 97.83 +1.24 -1.21 Mpc
Ωm: 0.285 +0.018 -0.018
w0: -0.806 +0.066 -0.069 (2.81 - 2.94 sigma from -1)
Chi squared: 31.9981 (Δ chi2 7.8756)
Degs of freedom: 35
Correlation matrix:
[[ 1.      -0.05676 -0.02696  0.06515]
 [-0.05676  1.       0.01977 -0.71908]
 [-0.02696  0.01977  1.      -0.63207]
 [ 0.06515 -0.71908 -0.63207  1.     ]]

==============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
ΔM: -9.281 +0.089 -0.088 mag
r_d * h: 97.60 +1.30 -1.26 Mpc
Ωm: 0.303 +0.015 -0.014
w0: -0.772 +0.077 -0.080 (2.85 - 2.96 sigma from -1)
Chi squared: 31.8358 (Δ chi2 8.0379)
Degs of freedom: 35
Correlation matrix:
[[ 1.      -0.07     0.00277  0.08178]
 [-0.07     1.      -0.40121 -0.74783]
 [ 0.00277 -0.40121  1.      -0.17619]
 [ 0.08178 -0.74783 -0.17619  1.     ]]
"""
