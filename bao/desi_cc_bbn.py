from numba import njit
import numpy as np
import emcee
import corner
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from cmb.data_desi_compression import r_drag, c
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from .plot_predictions import plot_bao_predictions
from cosmic_chronometers.plot_predictions import plot_cc_predictions

cc_legend, z_cc_vals, H_cc_vals, cc_cov_matrix = get_cc_data()
bao_legend, data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)
inv_cov_cc = np.linalg.inv(cc_cov_matrix)
logdet_cc = np.linalg.slogdet(cc_cov_matrix)[1]
N_cc = len(z_cc_vals)


@njit
def rd(H0, Om, Obh2):  # Mpc
    Omh2 = Om * (H0 / 100) ** 2
    return r_drag(wb=Obh2, wm=Omh2)


@njit
def Ez(z, params):
    Om, w0 = params[1], params[3]
    cubic = (1 + z) ** 3
    rho_DE = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(Om * cubic + (1 - Om) * rho_DE)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    x = np.linspace(0, z, num=200)
    y = DH_z(x, params)
    return np.trapz(y=y, x=x)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


def bao_theory(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    rd_val = rd(H0, Om, Obh2)
    results = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        q = qty[i]
        if q == "DV_over_rs":
            results[i] = DV_z(z[i], params) / rd_val
        elif q == "DM_over_rs":
            results[i] = DM_z(z[i], params) / rd_val
        elif q == "DH_over_rs":
            results[i] = DH_z(z[i], params) / rd_val
    return results


bounds = np.array(
    [
        (50, 80),  # H0
        (0.15, 0.45),  # Ωm
        (0.019, 0.025),  # Ωb h^2
        (-2, 0),  # w0
        (0.5, 2.5),  # f_cc
    ]
)

# Prior from BBN: https://arxiv.org/abs/2401.15054
omega_b_h2_prior = 0.02196
omega_b_h2_prior_sigma = 0.00063


def chi_squared(params):
    f_cc = params[-1]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, np.dot(inv_cov_cc * f_cc**2, delta_cc))

    delta_bao = data["value"] - bao_theory(data["z"], data["quantity"], params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))

    bbn_delta = (omega_b_h2_prior - params[2]) / omega_b_h2_prior_sigma
    return chi_cc + chi_bao + bbn_delta**2


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[-1]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 10 * ndim
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
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [w0_16, w0_50, w0_84],
        [f_cc_16, f_cc_50, f_cc_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50, f_cc_50])
    rd_samples = rd(samples[:, 0], samples[:, 1], samples[:, 2])
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1], axis=0)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Degs of freedom: {1 + data['value'].size + z_cc_vals.size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=f"{bao_legend}: $H_0$={H0_50:.2f} km/s/Mpc",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cc_cov_matrix)) / f_cc_50,
        label=f"{cc_legend}: $H_0$={H0_50:.1f} km/s/Mpc",
    )

    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "$f_{CCH}$"]
    corner.corner(
        samples,
        labels=labels,
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
*******************************
Dataset: DESI DR2 2025
*******************************

Flat ΛCDM
H0: 68.51 ± 0.57 km/s/Mpc
Ωb h^2: 0.02200 +0.00061 -0.00062
Ωm: 0.299 ± 0.008
w0: -1
f_cc: 1.49 +0.19 -0.18
rd: 148.00 +1.38 -1.35 Mpc
Chi squared: 43.56
log likelihood: -135.77
Degs of freedom: 43

===============================

Flat wCDM
H0: 67.04 +1.86 -1.84 km/s/Mpc
Ωb h^2: 0.02203 +0.00062 -0.00061
Ωm: 0.299 +0.009 -0.008
w0: -0.939 +0.070 -0.074
f_cc: 1.48 ± 0.18
rd: 149.39 +2.34 -2.13 Mpc
Chi squared: 42.24
log likelihood: -135.36
Degs of freedom: 42

===============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.4 +2.0 -1.9 km/s/Mpc
Ωb h^2: 0.02203 ± 0.00062
Ωm: 0.308 ± 0.011
w0: -0.867 +0.117 -0.125
f_cc: 1.47 ± 0.18
rd: 149.15 +1.79 -1.74 Mpc
Chi squared: 41.59
log likelihood: -135.14
Degs of freedom: 42

===============================

Flat w0waCDM
H0: 65.2 +2.0 -1.9 km/s/Mpc
Ωb h^2: 0.02193 +0.00062 -0.00062
Ωm: 0.348 +0.028 -0.028
w0: -0.547 +0.256 -0.243
wa: -1.465 +0.871 -0.904
f_cc: 1.45 +0.18 -0.17
rd: 145.97 +2.49 -2.19 Mpc
Chi squared: 39.41
log likelihood: -134.64
Degs of freedom: 41
"""
