from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

c = 299792.458  # Speed of light in km/s
rd = 147.09  # Mpc, fixed

grid = np.linspace(0, np.max(z_sn_vals), num=1000)


@njit
def Ez(z, params):
    O_m, exp_w0 = params[2], params[3]
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (2 * cubic / (1 + cubic)) ** (2 * (1 + np.log(exp_w0)))
    return np.sqrt(O_m * cubic + (1 - O_m) * rho_de)


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(grid, params), grid, initial=0)
    return np.interp(z_sn_vals, grid, integral_values)


def distance_modulus(params):
    dL = (1 + z_sn_vals) * c * integral_Ez(params) / params[1]
    return params[0] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[1] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    result = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        zp = z[i]
        x = np.linspace(0, zp, num=max(250, int(250 * zp)))
        y = DH_z(x, params)
        result[i] = np.trapz(y=y, x=x)
    return result


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def theory_predictions(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def chi_squared(params):
    delta_sn = mu_vals - distance_modulus(params)
    chi_sn = delta_sn.dot(cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - theory_predictions(
        bao_data["z"], quantities, params
    )
    chi_bao = delta_bao.dot(cho_solve(cho_bao, delta_bao, check_finite=False))
    return chi_sn + chi_bao


bounds = np.array(
    [
        (-0.7, 0.7),  # ΔM
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.15, 0.65),  # e^w0
    ],
    dtype=np.float64,
)


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.log(params[3])
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
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.30),
                (emcee.moves.DEMove(), 0.56),
                (emcee.moves.DESnookerMove(), 0.14),
            ],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    [
        [dM_16, dM_50, dM_84],
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [exp_w0_16, exp_w0_50, exp_w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([dM_50, H0_50, Om_50, exp_w0_50], dtype=np.float64)

    w0_samples = np.log(samples[:, 3])
    w0_16, w0_50, w0_84 = np.percentile(w0_samples, [15.9, 50, 84.1])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
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

    labels = ["$Δ_M$", "$H_0$", "$Ω_m$", "$e^{w_0}$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()

    plt.figure(figsize=(16, 1.5 * ndim))
    for n in range(ndim):
        plt.subplot2grid((ndim, 1), (n, 0))
        plt.plot(chains_samples[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""
*******************************
DESI BAO DR2 2025
*******************************

Flat ΛCDM
r_d: 147.09 Mpc (fixed)
ΔM: -0.119 +0.089 -0.089 mag
H0: 68.69 +0.47 -0.48 km/s/Mpc
Ωm: 0.304 +0.008 -0.008
w0: -1
Chi squared: 38.82
Degs of freedom: 32
Correlation matrix:
[[ 1.       0.15248 -0.13541]
 [ 0.15248  1.      -0.91781]
 [-0.13541 -0.91781  1.     ]]

===============================

Flat wCDM
r_d: 147.09 Mpc (fixed)
ΔM: -0.156 +0.089 -0.088 mag
H0: 67.12 +0.74 -0.73 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.866 +0.051 -0.050
Chi squared: 32.17 (Δ chi2 6.65)
Degs of freedom: 31
Correlation matrix:
[[ 1.       0.2271  -0.06492 -0.17294]
 [ 0.2271   1.      -0.19514 -0.81673]
 [-0.06492 -0.19514  1.      -0.35745]
 [-0.17294 -0.81673 -0.35745  1.     ]]

===============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
r_d: 147.09 Mpc (fixed)
ΔM: -0.163 +0.089 -0.090 mag
H0: 66.66 +0.82 -0.80 km/s/Mpc
Ωm: 0.310 +0.009 -0.008
w0: -0.803 +0.065 -0.066
Chi squared: 30.37 (Δ chi2 8.45)
Degs of freedom: 31
Correlation matrix:
[[ 1.       0.22998 -0.17029 -0.18202]
 [ 0.22998  1.      -0.67101 -0.8569 ]
 [-0.17029 -0.67101  1.       0.26096]
 [-0.18202 -0.8569   0.26096  1.     ]]

===============================

Flat w0waCDM
r_d: 147.09 Mpc (fixed)
ΔM: -0.166 +0.091 -0.091 mag
H0: 66.23 +0.92 -0.91 km/s/Mpc
Ωm: 0.330 +0.016 -0.018
w0: -0.700 +0.117 -0.111
wa: -0.996 +0.569 -0.567
Chi squared: 28.79 (Δ chi2 10.03)
Degs of freedom: 30
Correlation matrix:
[[ 1.       0.23428 -0.14014 -0.15026  0.08527]
 [ 0.23428  1.      -0.68941 -0.81009  0.57255]
 [-0.14014 -0.68941  1.       0.7572  -0.85998]
 [-0.15026 -0.81009  0.7572   1.      -0.89353]
 [ 0.08527  0.57255 -0.85998 -0.89353  1.     ]]

*******************************
SDSS BAO DR16 compilation 2020
*******************************

Flat ΛCDM
r_d: 147.09 Mpc (fixed)
ΔM: -0.137 +0.091 -0.090 mag
H0: 68.09 +0.65 -0.65 km/s/Mpc
Ωm: 0.313 +0.015 -0.014
w0: -1
Chi squared: 39.873
Degs of freedom: 36
Correlation matrix:
[[ 1.       0.21127 -0.17231]
 [ 0.21127  1.      -0.85116]
 [-0.17231 -0.85116  1.     ]]

===============================

Flat wCDM
r_d: 147.09 Mpc (fixed)
ΔM: -0.171 +0.091 -0.091 mag
H0: 66.52 +0.84 -0.83 km/s/Mpc
Ωm: 0.285 +0.018 -0.019
w0: -0.807 +0.067 -0.068
Chi squared: 31.999 (Δ chi2 7.874)
Degs of freedom: 35
Correlation matrix:
[[ 1.       0.24296 -0.01448 -0.15654]
 [ 0.24296  1.       0.01761 -0.71864]
 [-0.01448  0.01761  1.      -0.63004]
 [-0.15654 -0.71864 -0.63004  1.     ]]

===============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
r_d: 147.09 Mpc (fixed)
ΔM: -0.171 +0.091 -0.092 mag
H0: 66.35 +0.88 -0.86 km/s/Mpc
Ωm: 0.303 +0.015 -0.014
w0: -0.771 +0.077 -0.079
Chi squared: 31.834 (Δ chi2 8.039)
Degs of freedom: 35
Correlation matrix:
[[ 1.       0.24304 -0.12174 -0.15046]
 [ 0.24304  1.      -0.38784 -0.75141]
 [-0.12174 -0.38784  1.      -0.18599]
 [-0.15046 -0.75141 -0.18599  1.     ]]

===============================

Flat w0waCDM
r_d: 147.09 Mpc (fixed)
ΔM: -0.171 +0.091 -0.090 mag
H0: 66.35 +0.93 -0.92 km/s/Mpc
Ωm: 0.300 +0.032 -0.067
w0: -0.753 +0.109 -0.093
wa: -0.303 +0.790 -0.925
Chi squared: 32.6992
Degs of freedom: 34
Correlation matrix:
[[ 1.       0.23258 -0.04115 -0.11096  0.02962]
 [ 0.23258  1.      -0.37163 -0.70946  0.43873]
 [-0.04115 -0.37163  1.       0.30448 -0.80707]
 [-0.11096 -0.70946  0.30448  1.      -0.7151 ]
 [ 0.02962  0.43873 -0.80707 -0.7151   1.     ]]
"""
