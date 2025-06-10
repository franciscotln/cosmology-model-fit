import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from hubble.plotting import plot_predictions as plot_sn_predictions

legend_cc, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
legend_sn, z_sn_vals, mu_vals, cov_matrix_sn = get_sn_data()
cho_cc = cho_factor(cov_matrix_cc)
cho_sn = cho_factor(cov_matrix_sn)

c = 299792.458  # Speed of light in km/s


def Ez(z, O_m, w0):
    sum = 1 + z
    evolving_de = ((2 * sum**2) / (1 + sum**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * sum**3 + (1 - O_m) * evolving_de)


grid = np.linspace(0, np.max(z_sn_vals), num=3000)


def integral_Ez(params):
    y = 1 / Ez(grid, *params[2:])
    integral_values = cumulative_trapezoid(y=y, x=grid, initial=0)
    return np.interp(z_sn_vals, grid, integral_values)


def mu_theory(params):
    dM, h0 = params[0], params[1]
    dL = (1 + z_sn_vals) * (c / h0) * integral_Ez(params)
    return dM + 25 + 5 * np.log10(dL)


def plot_cc_predictions(params):
    z_smooth = np.linspace(0, max(z_cc_vals), 100)
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=np.sqrt(np.diag(cov_matrix_cc)),
        fmt=".",
        color="blue",
        alpha=0.4,
        label="CC data",
        capsize=2,
        linestyle="None",
    )
    plt.plot(z_smooth, H_z(z_smooth, params), color="green", alpha=0.5)
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z_cc_vals) + 0.2)
    plt.legend()
    plt.title(f"{legend_cc}: $H_0$={params[1]:.1f} km/s/Mpc")
    plt.show()


def H_z(z, params):
    return params[1] * Ez(z, *params[2:])


bounds = np.array(
    [
        (-0.7, 0.5),  # ΔM
        (55, 80),  # H0
        (0.1, 0.7),  # Ωm
        (-1.5, 0),  # w0
    ]
)


def chi_squared(params):
    delta_sn = mu_vals - mu_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(cc_delta, cho_solve(cho_cc, cc_delta))

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
    nwalkers = 8 * ndim
    burn_in = 500
    nsteps = 20000 + burn_in
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
        [dM_16, dM_50, dM_84],
        [h0_16, h0_50, h0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [dM_50, h0_50, Om_50, w0_50]
    deg_of_freedom = z_sn_vals.size + z_cc_vals.size - len(best_fit)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.2f} +{(w0_84 - w0_50):.2f} -{(w0_50 - w0_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(best_fit)
    plot_sn_predictions(
        legend=legend_sn,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f} km/s/Mpc, $\Omega_m$={Om_50:.4f}",
        x_scale="log",
    )
    corner.corner(
        samples,
        labels=["ΔM", "$H_0$", "$Ωm$", "$w_0$"],
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
ΔM: -0.214 +0.142 -0.147 mag
H0: 65.5 +3.5 -3.4 km/s/Mpc
Ωm: 0.353 +0.025 -0.025
w0: -1
wa: 0
Chi squared: 38.67
Degrees of freedom: 51

==============================

Flat wCDM: w(z) = w0
ΔM: -0.177 +0.146 -0.147 mag
H0: 66.4 +3.6 -3.5 km/s/Mpc
Ωm: 0.294 +0.059 -0.069
w0: -0.83 +0.14 -0.15
wa: 0
Chi squared: 37.35
Degrees of freedom: 50

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.179 +0.146 -0.148 mag
H0: 66.3 +3.6 -3.5 km/s/Mpc
Ωm: 0.312 +0.045 -0.046
w0: -0.83 +0.13 -0.15
wa: 0
Chi squared: 37.11
Degrees of freedom: 50

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
ΔM: -0.243 +0.150 -0.149 mag
H0: 63.6 +3.8 -3.6 km/s/Mpc
Ωm: 0.427 +0.046 -0.068
w0: -0.59 +0.25 -0.22
wa: -3.69 +2.44 -2.48
Chi squared: 35.18
Degrees of freedom: 49
"""
