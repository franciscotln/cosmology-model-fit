import numba
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data, effective_sample_size
from y2005cc.data import get_data as get_cc_data
from hubble.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_cc_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, observed_mu_vals, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)
inv_cov_cc = np.linalg.inv(cov_matrix_cc)
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

grid = np.linspace(0, np.max(z_cmb), num=2000)

c = 299792.458  # Speed of light in km/s


@numba.njit
def Ez(z, params):
    O_m, w0 = params[3], params[4]
    cubed = (1 + z) ** 3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubed + (1 - O_m) * rho_de)


def theory_mu(params):
    offset_mag, H0 = params[1], params[2]
    integral = cumulative_trapezoid(1 / Ez(grid, params), grid, initial=0)
    I = np.interp(z_cmb, grid, integral)
    return offset_mag + 25 + 5 * np.log10((1 + z_hel) * (c / H0) * I)


@numba.njit
def H_z(z, params):
    return params[2] * Ez(z, params)


bounds = np.array(
    [
        (0.4, 2.5),  # f_cc
        (-0.7, 0.7),  # ΔM
        (50, 80),  # H0
        (0.1, 0.5),  # Ωm
        (-2, 0),  # w0
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f_cc = params[0]
    delta_sn = observed_mu_vals - theory_mu(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, np.dot(inv_cov_cc * f_cc**2, delta_cc))

    return chi_sn + chi_cc


@numba.njit
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
    nwalkers = 5 * ndim
    burn_in = 500
    nsteps = 20000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

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
        [f_cc_16, f_cc_50, f_cc_84],
        [dM_16, dM_50, dM_84],
        [h0_16, h0_50, h0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [f_cc_50, dM_50, h0_50, Om_50, w0_50]
    deg_of_freedom = effective_sample_size + z_cc_vals.size - len(best_fit)

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
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
        legend=sn_legend,
        x=z_cmb,
        y=observed_mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"Best fit: $\Omega_m$={Om_50:.3f}, $H_0$={h0_50:.1f} km/s/Mpc",
        x_scale="log",
    )

    corner.corner(
        samples,
        labels=["$f_{CCH}$", "ΔM", "$H_0$", "Ωm", "$w_0$"],
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
f_cc: 1.47 +0.19 -0.18
ΔM: -0.113 +0.073 -0.074
H0: 65.9 +2.4 -2.3
Ωm: 0.349 +0.016 -0.016
w0: -1
Chi squared: 1672.14
Degrees of freedom: 1764

==============================

Flat wCDM: w(z) = w0
f_cc: 1.46 +0.18 -0.18
ΔM: -0.076 +0.083 -0.082
H0: 66.7 +2.6 -2.5
Ωm: 0.308 +0.043 -0.048
w0: -0.883 +0.101 -0.110
Chi squared: 1670.77
Degrees of freedom: 1763

==============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / ((1 + z)**3 + 1)
f_cc: 1.45 +0.18 -0.18
ΔM: -0.074 +0.081 -0.083
H0: 66.7 +2.5 -2.5
Ωm: 0.318 +0.030 -0.030
w0: -0.870 +0.094 -0.104
Chi squared: 1669.96
Degrees of freedom: 1763
"""
