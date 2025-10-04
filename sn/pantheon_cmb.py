from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2022pantheonSHOES.data import get_data
import cmb.data_chen_compression as cmb
from .plotting import plot_predictions as plot_sn_predictions, gelman_rubin

c = cmb.c  # Speed of light in km/s
O_r_h2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, mb_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)

sn_grid = np.linspace(0, np.max(z_cmb), num=1000)
one_plus_z_hel = 1 + z_hel


@njit
def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = O_r_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    fz_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * fz_de)


def sn_apparent_mag(params):
    H0, M = params[0], params[-1]
    integral_vals = cumulative_trapezoid(1 / Ez(sn_grid, params), sn_grid, initial=0)
    I = np.interp(z_cmb, sn_grid, integral_vals)
    dL = one_plus_z_hel * (c / H0) * I
    return M + 25 + 5 * np.log10(dL)


def chi_squared(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]

    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = np.dot(delta, np.dot(cmb.inv_cov_mat, delta))

    delta_sn = mb_values - sn_apparent_mag(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.15, 0.40),  # Ωm
        (0.020, 0.025),  # Ωb * h^2
        (-2.0, 0.0),  # w0
        (-20, -19),  # M
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
    nsteps = 1500 + burn_in
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
        print("effective samples", nwalkers * ndim * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains = sampler.get_chain(discard=burn_in, flat=False)

    print("Gelman-Rubin statistic:", gelman_rubin(np.transpose(chains, (1, 0, 2))))

    one_sigma_conf_int = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_conf_int, axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]
    M_16, M_50, M_84 = pct[4]

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50, M_50], dtype=np.float64)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    r_drag_samples = cmb.r_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_conf_int)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_conf_int)
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(z_drag_samples, one_sigma_conf_int)
    r_dr_16, r_dr_50, r_dr_84 = np.percentile(r_drag_samples, one_sigma_conf_int)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(
        f"z_drag: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}"
    )
    print(f"r_s(z*) = {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(
        f"r_s(z_drag) = {r_dr_50:.2f} +{(r_dr_84 - r_dr_50):.2f} -{(r_dr_50 - r_dr_16):.2f} Mpc"
    )
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mb_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=sn_apparent_mag(best_fit),
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "M"]
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
        plt.plot(chains[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
H0: 67.21 +0.55 -0.53 km/s/Mpc
Ωm: 0.319 +0.008 -0.007
Ωm h^2: 0.14429 +0.00114 -0.00113
Ωb h^2: 0.02233 +0.00014 -0.00014
w0: -1
M: -19.444 +0.015 -0.015
z*: 1088.98 +0.20 -0.20
z_drag: 1059.89 +0.28 -0.28
r_s(z*) = 144.07 Mpc
r_s(z_drag) = 146.64 +0.26 -0.27 Mpc
Chi squared: 1403.48
Degrees of freedom: 1587

===============================

Flat wCDM w(z) = w0
H0: 66.67 +0.81 -0.81 km/s/Mpc
Ωm: 0.324 +0.009 -0.009
Ωm h^2: 0.14385 +0.00126 -0.00124
Ωb h^2: 0.02236 +0.00014 -0.00014
w0: -0.975 +0.029 -0.030
M: -19.456 +0.021 -0.021
z*: 1088.91 +0.22 -0.21
z_drag: 1059.94 +0.29 -0.29
r_s(z*) = 144.17 Mpc
r_s(z_drag) = 146.72 +0.28 -0.28 Mpc
Chi squared: 1402.76
Degrees of freedom: 1586

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.77 +0.72 -0.71 km/s/Mpc
Ωm: 0.323 +0.008 -0.008
Ωm h^2: 0.14384 +0.00123 -0.00125
Ωb h^2: 0.02236 +0.00014 -0.00014
w0: -0.961 +0.042 -0.043
M: -19.452 +0.018 -0.018
z*: 1088.91 +0.21 -0.21
z_drag: 1059.94 +0.29 -0.29
r_s(z*) = 144.16 Mpc
r_s(z_drag) = 146.73 +0.29 -0.28 Mpc
Chi squared: 1402.70
Degrees of freedom: 1586

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.24 +1.27 -1.40 km/s/Mpc
Ωm: 0.318 +0.014 -0.012
Ωm h^2: 0.14390 +0.00124 -0.00125
Ωb h^2: 0.02236 +0.00014 -0.00015
w0: -0.919 +0.105 -0.111
wa: -0.290 +0.543 -0.544
M: -19.434 +0.042 -0.049
z*: 1088.92 +0.22 -0.21
z_drag: 1059.93 +0.29 -0.29
r_s(z*) = 144.14 Mpc
r_s(z_drag) = 146.72 +0.29 -0.28 Mpc
Chi squared: 1402.68
Degrees of freedom: 1585
"""
