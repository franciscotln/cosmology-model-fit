import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
# from y2018pantheon.data import get_data
from y2022pantheonSHOES.data import get_data

legend, z_values, apparent_mag_values, cov_matrix = get_data()

# Speed of light (km/s)
C = 299792.458

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Flat ΛCDM
def integral_of_e_z(zs, w0, wa):
    def integrand(z):
        radiation_limit = 1 / 3
        w_z = radiation_limit + (w0 - radiation_limit) * np.exp(-wa * z)
        return 1 / np.sqrt((1 + z) ** (3 * (1 + w_z)))

    return np.array([quad(integrand, 0, z_item)[0] for z_item in zs])


def lcdm_apparent_mag(z, params):
    [M, omega_m] = params
    a0_over_ae = 1 + z
    luminosity_distance = a0_over_ae * C * integral_of_e_z(z, omega_m, -1)
    return M + 25 + 5 * np.log10(luminosity_distance)


def wcdm_apparent_mag(z, params):
    [M, w0, wa] = params
    a0_over_ae = 1 + z
    luminosity_distance = a0_over_ae * C * integral_of_e_z(z, w0, wa)
    return M + 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params):
    delta = apparent_mag_values - wcdm_apparent_mag(z_values, params)
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (-30, -27), # M
    (-2, 0.5), # w0
    (-0.5, 1) # wa
])


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    steps_to_discard = 100
    n_dim = len(bounds)
    n_walkers = 100
    n_steps = steps_to_discard + 1000
    initial_pos = np.zeros((n_walkers, n_dim))

    for dim, (lower, upper) in enumerate(bounds):
      initial_pos[:, dim] = np.random.uniform(lower, upper, n_walkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        print_color("Autocorrelation time", tau)
        print_color("Average autocorrelation time", np.mean(tau))
    except:
        print_color("Autocorrelation time", "Not available")

    M0_samples = samples[:, 0]
    w0_samples = samples[:, 1]
    wa_samples = samples[:, 2]

    [M0_16, M0_50, M0_84] = np.percentile(M0_samples, [16, 50, 84])
    [w0_16, w0_50, w0_84] = np.percentile(w0_samples, [16, 50, 84])
    [wa_16, wa_50, wa_84] = np.percentile(wa_samples, [16, 50, 84])

    best_fit_params = [M0_50, w0_50, wa_50]

    # Calculate residuals
    predicted_apparent_mag = wcdm_apparent_mag(z_values, best_fit_params)
    residuals = apparent_mag_values - predicted_apparent_mag

    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(apparent_mag_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((apparent_mag_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    # Print the values in the console
    M0_label = f"{M0_50:.4f} +{M0_84-M0_50:.4f}/-{M0_50-M0_16:.4f}"
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f}/-{w0_50-w0_16:.4f}"
    wa_label = f"{wa_50:.4f} +{wa_84-wa_50:.4f}/-{wa_50-wa_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.4f} - {z_values[-1]:.4f}")
    print_color("Sample size", len(z_values))
    print_color("w_0", w0_label)
    print_color("w_a", wa_label)
    print_color("M0", M0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Reduced chi squared", chi_squared(best_fit_params)/ (len(z_values) - len(best_fit_params)))

    # Plot the data and the fit
    labels = [r"$M_0$", r"$w_0$", r"$w_a$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 12},
        smooth=1.5,
        smooth1d=1.5,
    )
    plt.show()

    # Plot chains for each parameter
    fig, axes = plt.subplots(n_dim, figsize=(10, 7))
    if n_dim == 1:
        axes = [axes]
    for i in range(n_dim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit_params[i], color='white', linestyle='--', alpha=0.5)
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_values,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=predicted_apparent_mag,
        label=f"Best fit: $w_0$={w0_50:.4f}, $M_0$={M0_50:.4f}",
        x_scale="log"
    )

    # Plot the residual analysis
    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=np.sqrt(np.diag(cov_matrix)),
        bins=40
    )

if __name__ == '__main__':
    main()

"""
*****************************
Dataset: Pantheon (2018)
z range: 0.010 - 2.260
Sample size: 1048
M0 contains Hubble constant and absolute magnitude
*****************************

ΛCDM
M0: -28.5749 +0.0102/-0.0108 (M(0.7)=-19.3494)
Ωm: 0.2993 +0.0207/-0.0219
R-squared: 99.71 %
RMSD (mag): 0.143
Skewness of residuals: 0.191
kurtosis of residuals: 0.698
Reduce chi squared: 0.9817

*****************************
Dataset: Pantheon+ (2022)
z range: 0.0102 - 2.2614
Sample size: 1590
M0 contains Hubble constant and absolute magnitude
*****************************

ΛCDM
M0: -28.5763 +0.0070/-0.0068 (M(0.7)=-19.3508)
Ωm: 0.3323 +0.0070/-0.0068
R-squared: 99.74 %
RMSD (mag): 0.154
Skewness of residuals: 0.091
kurtosis of residuals: 1.585
Reduce chi squared: 0.8840

=============================

wCDM
M0: -28.5731 +0.0076/-0.0085 (M(0.7)=-19.3476)
Ωm: 0.2698 +0.0566/-0.0701
w: -0.8580 +0.1104/-0.1336
R-squared: 99.74 %
RMSD (mag): 0.154
Skewness of residuals: 0.075
kurtosis of residuals: 1.593
Reduced chi squared: 0.8845

=============================

Fluid model
M0: -28.5725 +0.0084/-0.0084 (M(0.7)=-19.3470)
w0: -0.6293 +0.0393/-0.0420
wa: 0.2207 +0.0784/-0.0751
R-squared: 99.74 %
RMSD (mag): 0.154
Skewness of residuals: 0.079
kurtosis of residuals: 1.591
Reduced chi squared: 0.8844
"""
