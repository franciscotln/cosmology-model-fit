import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2022pantheonSHOES.data import get_data
from y2005cc.data import get_data as get_cc_data
from hubble.plotting import plot_predictions as plot_sn_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
legend, z_vals, z_hel_vals, apparent_mag_values, cov_matrix_sn = get_data()
cov_sn_cho = cho_factor(cov_matrix_sn)
cov_cc_cho = cho_factor(cov_matrix_cc)

c = 299792.458  # Speed of light in km/s


def Ez(z, O_m, w0=-1):
    sum = 1 + z
    evolving_de = ((2 * sum**2) / (1 + sum**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * sum**3 + (1 - O_m) * evolving_de)


z_grid_sn = np.linspace(0, np.max(z_vals), num=2000)


def integral_Ez(params):
    y = 1 / Ez(z_grid_sn, *params[2:])
    integral_values = cumulative_trapezoid(y=y, x=z_grid_sn, initial=0)
    return np.interp(z_vals, z_grid_sn, integral_values)


def apparent_mag_theory(params):
    h0, M = params[0], params[1]
    comoving_distance = (c / h0) * integral_Ez(params)
    return M + 25 + 5 * np.log10((1 + z_hel_vals) * comoving_distance)


def plot_cc_predictions(params):
    z_smooth = np.linspace(0, max(z_cc_vals), 100)
    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=np.sqrt(np.diag(cov_matrix_cc)),
        fmt=".",
        color="blue",
        alpha=0.4,
        label="CCH data",
        capsize=2,
        linestyle="None",
    )
    plt.plot(z_smooth, H_z(z_smooth, params), color="red", alpha=0.5, label="Model")
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z_cc_vals) + 0.2)
    plt.legend()
    plt.title(f"{cc_legend}: $H_0$={params[0]:.1f} km/s/Mpc")
    plt.show()


def H_z(z, params):
    return params[0] * Ez(z, *params[2:])


bounds = np.array(
    [
        (55, 80),  # H0
        (-20, -19),  # M
        (0.15, 0.70),  # Ωm
        (-1.5, 0.0),  # w0
    ]
)


def chi_squared(params):
    delta_sn = apparent_mag_values - apparent_mag_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cov_sn_cho, delta_sn))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, cho_solve(cov_cc_cho, delta_cc))

    return chi_sn + chi_cc


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
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
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 8000 + burn_in
    initial_pos = np.random.default_rng().uniform(
        bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim)
    )

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
        [h0_16, h0_50, h0_84],
        [M_16, M_50, M_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, M_50, Om_50, w0_50]

    deg_of_freedom = z_vals.size + z_cc_vals.size - len(best_fit)

    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag_theory(best_fit),
        label=f"Best fit: $H_0$={h0_50:.1f} km/s/Mpc, $\Omega_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$H_0$", "M", "$\Omega_m$", "$w_0$"]
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


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
H0: 66.9 +3.4 -3.3 km/s/Mpc
M: -19.449 +0.105 -0.110
Ωm: 0.331 +0.018 -0.017
w0: -1
wa: 0
Chi squared: 1417.49
Degrees of freedom: 1619

==============================

Flat wCDM: w(z) = w0
H0: 67.4 +3.6 -3.6 km/s/Mpc
M: -19.432 +0.113 -0.119 mag
Ωm: 0.312 +0.047 -0.052
w0: -0.946 +0.114 -0.129
wa: 0
Chi squared: 1417.27
Degrees of freedom: 1618

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 67.3 +3.6 -3.5 km/s/Mpc
M: -19.435 +0.113 -0.116 mag
Ωm: 0.319 +0.038 -0.039
w0: -0.956 +0.108 -0.123
wa: 0
Chi squared: 1417.27
Degrees of freedom: 1618
"""
