from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2022pantheonSHOES.data import get_data
from y2005cc.data import get_data as get_cc_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_cc_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
legend, z_vals, z_hel_vals, apparent_mag_values, cov_matrix_sn = get_data()
cov_sn_cho = cho_factor(cov_matrix_sn)
cho_cc = cho_factor(cov_matrix_cc)
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

z_grid_sn = np.linspace(0, np.max(z_vals), num=2000)

c = 299792.458  # Speed of light in km/s


@njit
def Ez(z, O_m, w0):
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * rho_de)


def integral_Ez(params):
    y = 1 / Ez(z_grid_sn, *params[3:])
    integral_values = cumulative_trapezoid(y=y, x=z_grid_sn, initial=0)
    return np.interp(z_vals, z_grid_sn, integral_values)


def apparent_mag_theory(params):
    H0, M = params[1], params[2]
    comoving_distance = (c / H0) * integral_Ez(params)
    return M + 25 + 5 * np.log10((1 + z_hel_vals) * comoving_distance)


@njit
def H_z(z, params):
    H0, Om, w0 = params[1], params[3], params[4]
    return H0 * Ez(z, Om, w0)


bounds = np.array(
    [
        (0.4, 2.5),  # f_cc
        (55, 80),  # H0
        (-20, -19),  # M
        (0.15, 0.70),  # Ωm
        (-1.5, 0.0),  # w0
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f_cc = params[0]
    delta_sn = apparent_mag_values - apparent_mag_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cov_sn_cho, delta_sn, check_finite=False))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, cho_solve(cho_cc, delta_cc, check_finite=False)) * f_cc**2

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
    nwalkers = 100 * ndim
    burn_in = 100
    nsteps = 1000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(10) as pool:
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
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [h0_16, h0_50, h0_84],
        [M_16, M_50, M_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [f_cc_50, h0_50, M_50, Om_50, w0_50]

    deg_of_freedom = z_vals.size + z_cc_vals.size - len(best_fit)

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_50,
        label=f"{cc_legend}: $H_0$={h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag_theory(best_fit),
        label=f"Best fit: $H_0$={h0_50:.1f} km/s/Mpc, $\Omega_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$f_{CCH}$", "$H_0$", "M", "$\Omega_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
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
f_cc: 1.47 +0.19 -0.18
H0: 67.0 +2.4 -2.4 km/s/Mpc
M: -19.446 +0.075 -0.077 mag
Ωm: 0.331 +0.017 -0.017
w0: -1
Chi squared: 1435.09
Degrees of freedom: 1619

==============================

Flat wCDM: w(z) = w0
f_cc: 1.46 +0.19 -0.18
H0: 67.3 +2.6 -2.6 km/s/Mpc
M: -19.436 +0.084 -0.086 mag
Ωm: 0.318 +0.039 -0.043
w0: -0.962 +0.101 -0.111
Chi squared: 1434.34
Degrees of freedom: 1618

==============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / ((1 + z)**3 + 1)
f_cc: 1.46 +0.18 -0.18
H0: 67.2 +2.6 -2.5 km/s/Mpc
M: -19.437 +0.081 -0.084 mag
Ωm: 0.323 +0.030 -0.030
w0: -0.964 +0.097 -0.109
Chi squared: 1434.50
Degrees of freedom: 1618
"""
