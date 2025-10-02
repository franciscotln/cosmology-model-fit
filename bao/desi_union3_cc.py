from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, sn_mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(cov_matrix_bao)
cho_cc = cho_factor(cov_matrix_cc)
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = 299792.458  # Speed of light in km/s

z_grid = np.linspace(0, np.max(z_sn_vals), num=1000)
one_plus_z_sn = 1 + z_sn_vals


@njit
def Ez(z, params):
    O_m, w0 = params[4], params[5]
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubed + (1 - O_m) * rho_de)


def mu_theory(params):
    dM, h0 = params[1], params[2]
    y = 1 / Ez(z_grid, params)
    integral_values = cumulative_trapezoid(y=y, x=z_grid, initial=0)
    I = np.interp(z_sn_vals, z_grid, integral_values)
    return dM + 25 + 5 * np.log10(one_plus_z_sn * (c / h0) * I)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


@njit
def DM_z(z, params):
    x = np.linspace(0, z, num=max(250, int(250 * z)))
    y = DH_z(x, params)
    return np.trapz(y=y, x=x)


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
def bao_theory(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        q = qty[i]
        if q == 0:
            results[i] = DV_z(z[i], params) / params[3]
        elif q == 1:
            results[i] = DM_z(z[i], params) / params[3]
        elif q == 2:
            results[i] = DH_z(z[i], params) / params[3]
    return results


bounds = np.array(
    [
        (0.4, 2.5),  # f_cc
        (-0.7, 0.7),  # ΔM
        (55, 80),  # H0
        (125, 170),  # r_d
        (0.2, 0.7),  # Ωm
        (-1.6, -0.4),  # w0
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f_cc = params[0]
    delta_sn = sn_mu_vals - mu_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = f_cc**2 * np.dot(delta_cc, cho_solve(cho_cc, delta_cc, check_finite=False))
    return chi_sn + chi_bao + chi_cc


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 500
    burn_in = 100
    nsteps = 1400 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(4) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.5),
                (emcee.moves.DEMove(), 0.4),
                (emcee.moves.DESnookerMove(), 0.1),
            ],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    print("Correlation matrix:")
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [dM_16, dM_50, dM_84],
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([f_cc_50, dM_50, h0_50, rd_50, Om_50, w0_50], dtype=np.float64)

    deg_of_freedom = (
        z_sn_vals.size + bao_data["value"].size + z_cc_vals.size - len(best_fit)
    )

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=sn_mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f}, $\Omega_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$f_{CCH}$", "ΔM", "$H_0$", "$r_d$", "Ωm", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    chains_samples = sampler.get_chain(discard=0, flat=False)
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    axes[ndim - 1].set_xlabel("chain step")
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
f_cc: 1.47 +0.18 -0.18
ΔM: -0.121 +0.113 -0.113
H0: 68.6 +2.3 -2.3
r_d: 147.1 +5.0 -4.6
Ωm: 0.305 +0.008 -0.008
w0: -1
Chi squared: 71.07
Degrees of freedom: 63
Correlation matrix:
[[ 1.       0.00839  0.00528 -0.01803  0.02173]
 [ 0.00839  1.       0.63723 -0.62528 -0.14372]
 [ 0.00528  0.63723  1.      -0.97817 -0.24218]
 [-0.01803 -0.62528 -0.97817  1.       0.05877]
 [ 0.02173 -0.14372 -0.24218  0.05877  1.     ]]

==============================

Flat wCDM: w(z) = w0
f_cc: 1.47 +0.19 -0.18
ΔM: -0.158 +0.116 -0.117 mag
H0: 67.1 +2.4 -2.3 km/s/Mpc
r_d: 147.3 +5.0 -4.7 Mpc
Ωm: 0.299 +0.009 -0.009
w0: -0.871 +0.051 -0.051
Chi squared: 64.53
Degrees of freedom: 62
Correlation matrix:
[[ 1.       0.01169  0.01152 -0.0159   0.02073 -0.02177]
 [ 0.01169  1.       0.6528  -0.63128 -0.07639 -0.1442 ]
 [ 0.01152  0.6528   1.      -0.94895 -0.10269 -0.27759]
 [-0.0159  -0.63128 -0.94895  1.       0.04269  0.0253 ]
 [ 0.02073 -0.07639 -0.10269  0.04269  1.      -0.3544 ]
 [-0.02177 -0.1442  -0.27759  0.0253  -0.3544   1.     ]]

==============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 1.46 +0.18 -0.18
ΔM: -0.165 +0.115 -0.117 mag
H0: 66.7 +2.4 -2.3 km/s/Mpc
r_d: 147.2 +5.0 -4.7 Mpc
Ωm: 0.310 +0.009 -0.008
w0: -0.811 +0.066 -0.067
Chi squared: 62.59
Degrees of freedom: 62
Correlation matrix:
[[ 1.       0.00697  0.01083 -0.01303  0.00814 -0.02863]
 [ 0.00697  1.       0.6435  -0.61921 -0.16838 -0.14937]
 [ 0.01083  0.6435   1.      -0.93879 -0.2726  -0.31085]
 [-0.01303 -0.61921 -0.93879  1.       0.04871  0.02084]
 [ 0.00814 -0.16838 -0.2726   0.04871  1.       0.24825]
 [-0.02863 -0.14937 -0.31085  0.02084  0.24825  1.     ]]

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
f_cc: 1.45 +0.18 -0.18
ΔM: -0.167 +0.116 -0.118 mag
H0: 66.3 +2.4 -2.4 km/s/Mpc
r_d: 147.1 +5.1 -4.7 Mpc
Ωm: 0.329 +0.016 -0.019
w0: -0.725 +0.113 -0.107
wa: -0.887 +0.564 -0.555
Chi squared: 61.22
Degrees of freedom: 61
Correlation matrix:
[[ 1.       0.00752  0.01645 -0.01413 -0.03293 -0.05057  0.04854]
 [ 0.00752  1.       0.64539 -0.6261  -0.09129 -0.10317  0.04608]
 [ 0.01645  0.64539  1.      -0.92733 -0.2588  -0.28832  0.1947 ]
 [-0.01413 -0.6261  -0.92733  1.       0.00794 -0.01275  0.01455]
 [-0.03293 -0.09129 -0.2588   0.00794  1.       0.75963 -0.87139]
 [-0.05057 -0.10317 -0.28832 -0.01275  0.75963  1.      -0.88604]
 [ 0.04854  0.04608  0.1947   0.01455 -0.87139 -0.88604  1.     ]]
"""
