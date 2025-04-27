import emcee
from getdist import MCSamples, plots
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.integrate import cumulative_trapezoid
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2024DES.data import get_data

legend, z_values, distance_modulus_values, cov_matrix = get_data()

# Speed of light (km/s)
C = 299792.458

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Hubble constant km/s/Mpc
H0 = 70


# Grid
z = np.linspace(0, np.max(z_values), num=3000)
sum = 1 + z

# Flat
def integral_e_z(params):
    omega_m, w0, wa = params[1], params[2], params[3]
    Ez = np.sqrt(omega_m * sum**3 + (1 - omega_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))
    integral_values = cumulative_trapezoid(1/Ez, z, initial=0)
    return np.interp(z_values, z, integral_values)


def model_distance_modulus(params):
    deltaM = params[0]
    comoving_distance = (C / H0) * integral_e_z(params)
    return deltaM + 25 + 5 * np.log10((1 + z_values) * comoving_distance)


def chi_squared(params):
    delta = distance_modulus_values - model_distance_modulus(params)
    return np.dot(delta, np.dot(inv_cov_matrix, delta))


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (-0.5, 0.5), # ΔM
    (0, 0.9), # Ωm
    (-3, 1), # w0
    (-18, 5) # wa
])


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
    nwalkers = 6 * ndim
    n_steps = steps_to_discard + 15000
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Effective number of independent samples", len(samples) / tau)
    except Exception as e:
        print("Autocorrelation time could not be computed")

    [
        [deltaM_16, deltaM_50, deltaM_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84]
    ] = np.percentile(samples, [16, 50, 84], axis=0).T

    best_fit_params = [deltaM_50, omega_50, w0_50, wa_50]

    predicted_distance_modulus_values = model_distance_modulus(best_fit_params)
    residuals = distance_modulus_values - predicted_distance_modulus_values

    skewness = stats.skew(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(distance_modulus_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((distance_modulus_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    # Print the values in the console
    deltaM_label = f"{deltaM_50:.4f} +{deltaM_84-deltaM_50:.4f} -{deltaM_50-deltaM_16:.4f}"
    omega_label = f"{omega_50:.4f} +{omega_84-omega_50:.4f} -{omega_50-omega_16:.4f}"
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f} -{w0_50-w0_16:.4f}"
    wa_label = f"{wa_50:.4f} +{wa_84-wa_50:.4f} -{wa_50-wa_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("ΔM", deltaM_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("wa", wa_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("Chi squared", chi_squared(best_fit_params))

    # plot posterior distribution from samples
    labels = ["ΔM", "Ωm", "w_0", "w_a"]
    gdsamples = MCSamples(
        samples=samples,
        names=labels,
        labels=labels,
        settings={"fine_bins_2D": 128, "smooth_scale_2D": 0.9}
    )
    g = plots.get_subplot_plotter()
    g.triangle_plot(
        gdsamples,
        Filled=False,
        contour_levels=[0.68, 0.95],
        title_limit=True,
        diag1d_kwargs={"density": True},
    )
    plt.show()

    y_err = np.sqrt(cov_matrix.diagonal())

    # Plot the predictions
    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=y_err,
        y_model=predicted_distance_modulus_values,
        label=f"$\Omega_m$={omega_label}",
        x_scale="log"
    )

    # Plot the residual analysis
    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=y_err,
        bins=40
    )

if __name__ == '__main__':
    main()

"""
********************************
Dataset: DES-SN5YR
z range: 0.025 - 1.121
Sample size: 1829
********************************

Flat ΛCDM
ΔM: 0.0216 +0.0110 -0.0111
Ωm: 0.3507 +0.0171 -0.0169
w0: -1
wa: 0
R-squared (%): 98.41
RMSD (mag): 0.264
Skewness of residuals: 3.409
Chi squared: 1640.3080970418623

==============================

Flat wCDM
ΔM: 0.0304 +0.0126 -0.0128
Ωm: 0.2729 +0.0688 -0.0935
w0: -0.8154 +0.1435 -0.1536
wa: 0
R-squared (%): 98.40
RMSD (mag): 0.264
Skewness of residuals: 3.416
Chi squared: 1639.00

==============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: 0.0309 +0.0134 -0.0134
Ωm: 0.2969 +0.0486 -0.0541
w0: -0.8334 +0.1239 -0.1382
wa: 0
R-squared (%): 98.40
RMSD (mag): 0.264
Skewness of residuals: 3.418
Chi squared: 1638.70

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
