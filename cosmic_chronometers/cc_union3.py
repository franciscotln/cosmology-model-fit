from matplotlib.pylab import f
from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool

from torch import exp_
from y2023union3.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_cc_predictions

legend_sn, z_sn_vals, mu_vals, cov_matrix_sn = get_sn_data()
legend_cc, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)
cho_sn = cho_factor(cov_matrix_sn)
cho_cc = cho_factor(cov_matrix_cc)

c = 299792.458  # Speed of light in km/s

z_grid = np.linspace(0, np.max(z_sn_vals), num=1000)


@njit
def Ez(z, O_m, exp_w0):
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + np.log(exp_w0)))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * rho_de)


def mu_theory(params):
    dM, h0 = params[1], params[2]
    y = 1 / Ez(z_grid, *params[3:])
    integral_values = cumulative_trapezoid(y=y, x=z_grid, initial=0)
    I = np.interp(z_sn_vals, z_grid, integral_values)
    return dM + 25 + 5 * np.log10((1 + z_sn_vals) * (c / h0) * I)


@njit
def H_z(z, params):
    H0, Om, w0 = params[2], params[3], params[4]
    return H0 * Ez(z, Om, w0)


bounds = np.array(
    [
        (0.4, 2.5),  # f_cc
        (-0.7, 0.5),  # ΔM
        (55, 80),  # H0
        (0.1, 0.7),  # Ωm
        (0.01, 0.8),  # exp(w0)
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f_cc = params[0]
    delta_sn = mu_vals - mu_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = f_cc**2 * np.dot(cc_delta, cho_solve(cho_cc, cc_delta, check_finite=False))

    return chi_sn + chi_cc


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
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

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
        print("effective samples", nwalkers * (nsteps - burn_in) * ndim / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    print("correlation matrix:")
    print(np.corrcoef(samples, rowvar=False))

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [dM_16, dM_50, dM_84],
        [h0_16, h0_50, h0_84],
        [Om_16, Om_50, Om_84],
        [exp_w0_16, exp_w0_50, exp_w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [f_cc_50, dM_50, h0_50, Om_50, exp_w0_50]
    deg_of_freedom = z_sn_vals.size + z_cc_vals.size - len(best_fit)

    w0_samples = np.log(samples[:, 4])
    w0_16, w0_50, w0_84 = np.percentile(w0_samples, [15.9, 50, 84.1])

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.2f} +{(w0_84 - w0_50):.2f} -{(w0_50 - w0_16):.2f}")
    print(
        f"exp(w0): {exp_w0_50:.2f} +{(exp_w0_84 - exp_w0_50):.2f} -{(exp_w0_50 - exp_w0_16):.2f}"
    )
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_50,
        label=f"{legend_cc}: $H_0$={h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=legend_sn,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f} km/s/Mpc, $\Omega_m$={Om_50:.4f}",
        x_scale="log",
    )
    labels = ["$f_{CCH}$", "ΔM", "$H_0$", "$Ωm$", "$e^{w_0}$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=2.0,
        smooth1d=2.0,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
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
Flat ΛCDM: w(z) = -1
f_cc: 1.47 +0.18 -0.18
ΔM: -0.203 +0.122 -0.122 mag
H0: 65.9 +2.6 -2.6 km/s/Mpc
Ωm: 0.349 +0.024 -0.023
w0: -1
Chi squared: 56.20
Log likelihood: -142.67
Degrees of freedom: 51

==============================

Flat wCDM: w(z) = w0
f_cc: 1.45 +0.18 -0.18
ΔM: -0.176 +0.123 -0.123 mag
H0: 66.4 +2.7 -2.6 km/s/Mpc
Ωm: 0.300 +0.047 -0.055
w0: -0.84 +0.12 -0.13
exp(w0): 0.43 +0.06 -0.05
Chi squared: 54.09
Log likelihood: -142.07
Degrees of freedom: 50

==============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / ((1 + z)**3 + 1)
f_cc: 1.45 +0.18 -0.18
ΔM: -0.177 +0.123 -0.123 mag
H0: 66.3 +2.6 -2.6 km/s/Mpc
Ωm: 0.318 +0.033 -0.033
w0: -0.82 +0.12 -0.13
exp(w0): 0.44 +0.06 -0.05
Chi squared: 53.84
Log likelihood: -141.87
Degrees of freedom: 50
"""
