from numba import njit
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
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

z_grid_sn = np.linspace(0, np.max(z_vals), num=1000)

c = 299792.458  # Speed of light in km/s


@njit
def Ez(z, Om, w0):
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))
    return np.sqrt(Om * one_plus_z**3 + (1 - Om) * rho_de)


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
        (0.1, 1.5),  # f_cc
        (55, 80),  # H0
        (-20, -19),  # M
        (0.15, 0.70),  # Ωm
        (-2.0, 0.0),  # w0
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f_cc = params[0]
    delta_sn = apparent_mag_values - apparent_mag_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cov_sn_cho, delta_sn, check_finite=False))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, cho_solve(cho_cc, delta_cc, check_finite=False)) * f_cc**-2

    return chi_sn + chi_cc


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc + 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee, corner
    import matplotlib.pyplot as plt
    from multiprocessing import Pool

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 1500 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.3),
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

    Omh2_samples = samples[:, 3] * (samples[:, 1] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) * f_cc_50,
        label=f"{cc_legend}: $H_0$={h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag_theory(best_fit),
        label=f"Best fit: $H_0$={h0_50:.1f} km/s/Mpc, $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$f_{CC}$", "$H_0$", "M", "$Ω_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
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
f_cc: 0.70 +0.10 -0.08
H0: 67.1 +2.5 -2.5 km/s/Mpc
M: -19.445 +0.077 -0.079 mag
Ωm: 0.331 +0.017 -0.017
ωm: 0.1490
w0: -1
Chi squared: 1433.17
Degrees of freedom: 1619

===============================

Flat wCDM: w(z) = w0
f_cc: 0.71 +0.10 -0.08
H0: 67.4 +2.7 -2.6 km/s/Mpc
M: -19.432 +0.085 -0.087 mag
Ωm: 0.317 +0.040 -0.043
ωm: 0.1432 +0.0159 -0.0173
w0: -0.958 +0.102 -0.113
Chi squared: 1432.42
Degrees of freedom: 1618

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / ((1 + z)**3 + 1)
f_cc: 0.70 +0.10 -0.08
H0: 67.3 +2.6 -2.6 km/s/Mpc
M: -19.436 +0.084 -0.086 mag
Ωm: 0.322 +0.030 -0.030
ωm: 0.1455 +0.0130 -0.0128
w0: -0.963 +0.097 -0.109
Chi squared: 1432.58
Degrees of freedom: 1618
"""
