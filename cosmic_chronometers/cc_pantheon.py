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


def h_over_h0_model(z, params):
    O_m, w0 = params[2], params[3]
    sum = 1 + z
    return np.sqrt(
        O_m * sum**3 + (1 - O_m) * ((2 * sum**2) / (1 + sum**2)) ** (3 * (1 + w0))
    )


z_grid_sn = np.linspace(0, np.max(z_vals), num=2000)


def integral_e_z(params):
    integral_values = cumulative_trapezoid(
        1 / h_over_h0_model(z_grid_sn, params), z_grid_sn, initial=0
    )
    return np.interp(z_vals, z_grid_sn, integral_values)


def model_apparent_mag(params):
    h0, M = params[0], params[1]
    comoving_distance = (c / h0) * integral_e_z(params)
    return M + 25 + 5 * np.log10((1 + z_hel_vals) * comoving_distance)


def plot_cc_predictions(params):
    h0 = params[0]
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
    plt.title(f"{cc_legend}: $H_0$={h0:.2f} km/s/Mpc")
    plt.show()


def H_z(z, params):
    h0 = params[0]
    return h0 * h_over_h0_model(z, params)


bounds = np.array(
    [
        (55, 80),  # H0
        (-20, -19),  # M
        (0.15, 0.70),  # Ωm
        (-1.5, 0.0),  # w0
        (-3.5, 3.5),  # wa
    ]
)


def chi_squared(params):
    delta_sn = apparent_mag_values - model_apparent_mag(params)
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
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, M_50, omega_50, w0_50, wa_50]

    deg_of_freedom = z_vals.size + z_cc_vals.size - len(best_fit)

    print(f"H0: {h0_50:.2f} +{(h0_84 - h0_50):.2f} -{(h0_50 - h0_16):.2f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(
        f"Ωm: {omega_50:.3f} +{(omega_84 - omega_50):.3f} -{(omega_50 - omega_16):.3f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"wa: {wa_50:.3f} +{(wa_84 - wa_50):.3f} -{(wa_50 - wa_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=model_apparent_mag(best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f} km/s/Mpc, $\Omega_m$={omega_50:.3f}",
        x_scale="log",
    )

    labels = [r"$H_0$", "M", f"$\Omega_m$", r"$w_0$", r"$w_a$"]
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
H0: 67.04 +3.37 -3.32 km/s/Mpc
M: -19.445 +0.105 -0.109 mag
w0: -1
wa: 0
Ωm: 0.331 +0.018 -0.017
Chi squared: 1417.45
Degrees of freedom: 1619

==============================

Flat wCDM: w(z) = w0
H0: 67.44 +3.64 -3.54 km/s/Mpc
M: -19.430 +0.114 -0.118 mag
Ωm: 0.313 +0.047 -0.053
w0: -0.950 +0.115 -0.133
wa: 0
Chi squared: 1418.03
Degrees of freedom: 1618

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 67.45 +3.65 -3.54 km/s/Mpc
M: -19.429 +0.115 -0.118 mag
Ωm: 0.316 +0.039 -0.039
w0: -0.948 +0.107 -0.125 (0.35 - 0.41 sigma)
wa: 0
Chi squared: 1417.18
Degrees of freedom: 1618

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
H0: 67.5435 +1.3550 -1.3776 km/s/Mpc
M: -19.4252 +0.0413 -0.0429
Ωm: 0.3371 +0.0529 -0.0850
w0: -0.9321 +0.0990 -0.1089 (0.62 - 0.69 sigma)
wa: -0.3567 +0.9664 -1.3853
f: 0.7145 +0.1046 -0.0844 (2.73 - 3.38 sigma)
Chi squared: 1433.5840
Degrees of freedom: 1616
"""
