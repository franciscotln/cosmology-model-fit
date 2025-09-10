import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2024DES.data import get_data, effective_sample_size

legend, z_cmb_vals, z_hel_vals, observed_mu_vals, covmat = get_data()
cho = cho_factor(covmat)

C = 299792.458  # Speed of light (km/s)
H0 = 70  # Hubble constant km/s/Mpc

grid = np.linspace(0, np.max(z_cmb_vals), num=2500)
one_plus_z = 1 + grid


def integral_Ez(Om, w0=-1):
    Ode = 1 - Om
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))
    Ez = np.sqrt(Om * one_plus_z**3 + Ode * rho_de)
    integral_vals = cumulative_trapezoid(1 / Ez, grid, initial=0)
    return np.interp(z_cmb_vals, grid, integral_vals)


def dL(params):
    return (1 + z_hel_vals) * (C / H0) * integral_Ez(*params[1:])


def theory_mu(params):
    return params[0] + 25 + 5 * np.log10(dL(params))


def chi_squared(params):
    delta = observed_mu_vals - theory_mu(params)
    return np.dot(delta, cho_solve(cho, delta))


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([(-0.6, 0.6), (0, 0.8), (-1.5, 0)])  # ΔM  # Ωm  # w0


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    else:
        return -np.inf


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    steps_to_discard = 500
    ndim = len(bounds)
    nwalkers = 10 * ndim
    nsteps = steps_to_discard + 12000
    initial_state = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_state, nsteps, progress=True)

    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Effective number of independent samples", len(samples) / tau)
    except Exception as e:
        print("Autocorrelation time could not be computed")

    [
        [dM_16, dM_50, dM_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit_params = [dM_50, Om_50, w0_50]

    theory_mu_vals = theory_mu(best_fit_params)
    residuals = observed_mu_vals - theory_mu_vals

    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed_mu_vals - np.mean(observed_mu_vals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    # Print the values in the console
    dM_label = f"{dM_50:.3f} +{dM_84-dM_50:.3f} -{dM_50-dM_16:.3f}"
    Om_label = f"{Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}"
    w0_label = f"{w0_50:.2f} +{w0_84-w0_50:.2f} -{w0_50-w0_16:.2f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_cmb_vals[0]:.3f} - {z_cmb_vals[-1]:.3f}")
    print_color("Sample size", len(z_cmb_vals))
    print_color("ΔM", dM_label)
    print_color("Ωm", Om_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{stats.skew(residuals):.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit_params):.2f}")
    print_color("Effective deg of freedom", effective_sample_size - ndim)

    # plot posterior distribution from samples
    labels = ["ΔM", "$Ω_m$", "$w_0$"]
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

    y_err = np.sqrt(covmat.diagonal())
    plot_predictions(
        legend=legend,
        x=z_cmb_vals,
        y=observed_mu_vals,
        y_err=y_err,
        y_model=theory_mu_vals,
        label=f"Model: $\Omega_m$={Om_label}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb_vals, residuals=residuals, y_err=y_err, bins=40)


if __name__ == "__main__":
    main()

"""
********************************
Dataset: DES-SN5YR
z range: 0.025 - 1.121
Sample size: 1829
********************************

Flat ΛCDM
ΔM: 0.022 +0.011 -0.011
Ωm: 0.352 +0.017 -0.017
w0: -1
wa: 0
R-squared (%): 98.41
RMSD (mag): 0.263
Skewness of residuals: 3.407
Chi squared: 1640.08

==============================

Flat wCDM
ΔM: 0.031 +0.013 -0.013
Ωm: 0.267 +0.072 -0.095
w0: -0.80 +0.14 -0.16
wa: 0
R-squared (%): 98.40
RMSD (mag): 0.264
Skewness of residuals: 3.415
Chi squared: 1638.52

==============================

Flat w0 - (1 + w0) * ((1 + z)**3 - 1) / ((1 + z)**3 + 1)
ΔM: 0.033 +0.013 -0.014 mag
Ωm: 0.298 +0.044 -0.046
w0: -0.81 +0.12 -0.14
wa: 0
R-squared (%): 98.40
RMSD (mag): 0.264
Skewness of residuals: 3.417
Chi squared: 1637.98

==============================

Flat w0waCDM
ΔM: 0.0580 +0.0173 -0.0170
Ωm: 0.4933 +0.0313 -0.0446
w0: -0.4021 +0.3529 -0.2997 (1.83 sigma)
wa: -8.5853 +3.8321 -4.3735 (2.09 sigma)
R-squared (%): 98.40
RMSD (mag): 0.264
Skewness of residuals: 3.453
Chi squared: 1632.74
"""
