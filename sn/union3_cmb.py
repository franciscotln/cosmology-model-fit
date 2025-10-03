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
    H0, Om, exp_w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = O_r_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + np.log(exp_w0)))

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
        (0.01, 1.0),  # w0
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

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    exp_w0_16, exp_w0_50, exp_w0_84 = pct[3]
    dM_16, dM_50, dM_84 = pct[4]

    best_fit = np.array([H0_50, Om_50, Obh2_50, exp_w0_50, dM_50], dtype=np.float64)

    w0_samples = np.log(samples[:, 3])
    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)

    w0_16, w0_50, w0_84 = np.percentile(w0_samples, [15.9, 50, 84.1])
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, [15.9, 50, 84.1])
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(z_drag_samples, [15.9, 50, 84.1])

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(
        f"z_drag: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}"
    )
    print(f"r_s(z*) = {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"r_s(z_drag) = {cmb.rs_z(Ez, z_dr_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
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
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$e^{w_0}$", "$Δ_M$"]
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
H0: 67.11 +0.57 -0.57 km/s/Mpc
Ωm: 0.319 +0.008 -0.008
Ωm h^2: 0.14359 +0.00122 -0.00121
Ωb h^2: 0.02234 +0.00014 -0.00014
w0: -1
ΔM: -0.167 +0.089 -0.090
z*: 1091.99 +0.27 -0.27
z_drag: 1063.40 +0.29 -0.30
r_s(z*) = 144.00 Mpc
r_s(z_drag) = 146.52 Mpc
Chi squared: 26.2
Degrees of freedom: 21

===============================

Flat wCDM w(z) = w0
H0: 65.15 +1.20 -1.21 km/s/Mpc
Ωm: 0.337 +0.014 -0.013
Ωm h^2: 0.14293 +0.00125 -0.00125
Ωb h^2: 0.02240 +0.00015 -0.00015
w0: -0.922 +0.043 -0.043
ΔM: -0.222 +0.094 -0.095
z*: 1091.86 +0.28 -0.28
z_drag: 1063.46 +0.30 -0.30
r_s(z*) = 144.14 Mpc
r_s(z_drag) = 146.65 Mpc
Chi squared: 23.2 (Δ chi2 3.0)
Degrees of freedom: 20

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)^3)
H0: 65.24 +1.08 -1.03 km/s/Mpc
Ωm: 0.336 +0.012 -0.012
Ωm h^2: 0.14287 +0.00125 -0.00124
Ωb h^2: 0.02240 +0.00014 -0.00014
w0: -0.868 +0.065 -0.066
ΔM: -0.215 +0.092 -0.090
z*: 1091.85 +0.27 -0.28
z_drag: 1063.47 +0.29 -0.29
r_s(z*) = 144.16 Mpc
r_s(z_drag) = 146.66 Mpc
Chi squared: 22.53 (Δ chi2 3.9)
Degrees of freedom: 20

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 66.49 +1.32 -1.34 km/s/Mpc
Ωm: 0.324 +0.014 -0.013
Ωm h^2: 0.14307 +0.00127 -0.00124
Ωb h^2: 0.02238 +0.00014 -0.00014
w0: -0.685 +0.154 -0.158
wa: -1.123 +0.731 -0.741
ΔM: -0.158 +0.098 -0.100
z*: 1091.89 +0.28 -0.27
z_drag: 1063.45 +0.29 -0.29
r_s(z*) = 144.11 Mpc
r_s(z_drag) = 146.62 Mpc
Chi squared: 21.80 (Δ chi2 4.4)
Degrees of freedom: 19
"""
