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

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, observed_mu_vals, cov_matrix_sn = get_data()
cho_cc = cho_factor(cov_matrix_cc)
cho_sn = cho_factor(cov_matrix_sn)

c = 299792.458  # Speed of light in km/s

grid = np.linspace(0, np.max(z_cmb), num=2000)


def Ez(z, O_m, w0=-1):
    sum = 1 + z
    evolving_de = ((2 * sum**2) / (1 + sum**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * sum**3 + (1 - O_m) * evolving_de)


def integral_Ez(params):
    integral = cumulative_trapezoid(1 / Ez(grid, *params[2:]), grid, initial=0)
    return np.interp(z_cmb, grid, integral)


def theory_mu(params):
    dM, H0 = params[0], params[1]
    return dM + 25 + 5 * np.log10((1 + z_hel) * (c / H0) * integral_Ez(params))


def H_z(z, params):
    return params[1] * Ez(z, *params[2:])


def plot_cc_predictions(params):
    h0 = params[1]
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
    plt.plot(z_smooth, H_z(z_smooth, params), color="green", alpha=0.5, label="Model")
    plt.xlabel("Redshift (z)")
    plt.ylabel("$H(z)$")
    plt.xlim(0, np.max(z_cc_vals) + 0.2)
    plt.legend()
    plt.title(f"{cc_legend}: $H_0$={h0:.1f} km/s/Mpc")
    plt.show()


bounds = np.array(
    [
        (-0.7, 0.7),  # ΔM
        (50, 80),  # H0
        (0.1, 0.5),  # Ωm
        (-3, 1),  # w0
    ]
)


def chi_squared(params):
    delta_sn = observed_mu_vals - theory_mu(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, cho_solve(cho_cc, delta_cc))

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
    nwalkers = 10 * ndim
    burn_in = 500
    nsteps = 10000 + burn_in
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
    deg_of_freedom = effective_sample_size + z_cc_vals.size - len(best_fit)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(best_fit)
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
        labels=["ΔM", "$H_0$", "Ωm", "$w_0$"],
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
ΔM: -0.117 +0.103 -0.106 mag
H0: 65.7 +3.3 -3.2 km/s/Mpc
Ωm: 0.350 +0.017 -0.016
w0: -1
wa: 0
Chi squared: 1654.77
Degrees of freedom: 1764

==============================

Flat wCDM: w(z) = w0
ΔM: -0.070 +0.112 -0.115 mag
H0: 66.9 +3.5 -3.5 km/s/Mpc
Ωm: 0.299 +0.050 -0.059
w0: -0.862 +0.113 -0.123
wa: 0
Chi squared: 1653.57
Degrees of freedom: 1763

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.068 +0.109 -0.116 mag
H0: 66.9 +3.5 -3.5 km/s/Mpc
Ωm: 0.310 +0.039 -0.040
w0: -0.860 +0.103 -0.117
wa: 0
Chi squared: 1653.26

==============================

ΔM: -0.117 +0.114 -0.118 mag
H0: 65.20 +3.54 -3.45 km/s/Mpc
Ωm: 0.400 +0.033 -0.054
w0: -0.803 +0.112 -0.124
wa: -2.358 +1.443 -0.827 (unconstrained)
Chi squared: 1650.40
Degrees of freedom: 1762
"""
