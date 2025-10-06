from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
import cmb.data_union3_compression as cmb
from .plotting import plot_predictions


c = cmb.c  # km/s
O_r_h2 = cmb.Omega_r_h2()

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)

sn_grid = np.linspace(0, np.max(z_sn_vals), num=1000)


@njit
def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = O_r_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


def mu_theory(params):
    H0, mag_offset = params[0], params[-1]
    integral_vals = cumulative_trapezoid(1 / Ez(sn_grid, params), sn_grid, initial=0)
    I = np.interp(z_sn_vals, sn_grid, integral_vals)
    return mag_offset + 25 + 5 * np.log10((1 + z_sn_vals) * I * c / H0)


def chi_squared(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = np.dot(delta, np.dot(cmb.inv_cov_mat, delta))

    delta_sn = mu_vals - mu_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.45),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-2.0, 0.0),  # w0
        (-0.7, 0.7),  # ΔM
    ],
    dtype=np.float64,
)


@njit
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

    one_sigma_percentiles = [15.9, 50, 84.1]

    pct = np.percentile(samples, one_sigma_percentiles, axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]
    dM_16, dM_50, dM_84 = pct[4]

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50, dM_50], dtype=np.float64)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_percentiles)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_percentiles)
    z_d_16, z_d_50, z_d_84 = np.percentile(z_drag_samples, one_sigma_percentiles)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"r_d: {cmb.rs_z(Ez, z_d_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    labels = ["$H_0$", "$Ω_m$", "$ω_m$", "$w_0$", "$Δ_M$"]
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
        levels=(0.393, 0.864),
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
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*******************************

Flat ΛCDM w(z) = -1
H0: 67.11 +0.57 -0.56 km/s/Mpc
Ωm: 0.319 +0.008 -0.008
ωm: 0.14358 +0.00120 -0.00120
ωb: 0.02235 +0.00014 -0.00014
w0: -1
ΔM: -0.167 +0.089 -0.088
z*: 1091.99 +0.27 -0.27
z_drag: 1059.88 +0.28 -0.29
r*: 144.00 Mpc
r_d: 146.84 Mpc
Chi squared: 26.2
Degrees of freedom: 21

===============================

Flat wCDM w(z) = w0
H0: 65.19 +1.22 -1.20 km/s/Mpc
Ωm: 0.336 +0.013 -0.013
ωm: 0.14294 +0.00126 -0.00127
ωb: 0.02240 +0.00014 -0.00015
w0: -0.924 +0.042 -0.043
ΔM: -0.220 +0.093 -0.095
z*: 1091.86 +0.28 -0.28
z_drag: 1059.95 +0.29 -0.29
r*: 144.14 Mpc
r_d: 146.97 Mpc
Chi squared: 23.2 (Δ chi2 3.0)
Degrees of freedom: 20

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)^3)
H0: 65.30 +1.08 -1.06 km/s/Mpc
Ωm: 0.335 +0.012 -0.012
ωm: 0.14288 +0.00125 -0.00124
ωb: 0.02240 +0.00014 -0.00015
w0: -0.872 +0.067 -0.066
ΔM: -0.212 +0.092 -0.091
z*: 1091.85 +0.28 -0.27
z_drag: 1059.95 +0.29 -0.29
r*: 144.15 Mpc
r_d: 146.98 Mpc
Chi squared: 22.5 (Δ chi2 3.7)
Degrees of freedom: 20

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 66.51 +1.30 -1.40 km/s/Mpc
Ωm: 0.324 +0.014 -0.013
ωm: 0.14306 +0.00127 -0.00128
ωb: 0.02239 +0.00015 -0.00014
w0: -0.689 +0.155 -0.160
wa: -1.106 +0.734 -0.753
ΔM: -0.160 +0.098 -0.101
z*: 1091.89 +0.27 -0.28
z_drag: 1059.94 +0.29 -0.29
r*: 144.09 Mpc
r_d: 146.92 Mpc
Chi squared: 21.4 (Δ chi2 4.8)
Degrees of freedom: 19
"""
