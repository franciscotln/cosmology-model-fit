import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.integrate import cumulative_trapezoid
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
# from y2018pantheon.data import get_data
from y2022pantheonSHOES.data import get_data

legend, z_values, apparent_mag_values, cov_matrix = get_data()

# Speed of light (km/s)
C = 299792.458

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

def w_de(z, w0, wa):
    return w0 + (wa - w0) * np.tanh(0.5*((1 + z)**2 - 1))


def rho_de(zs, w0, wa):
    z = np.linspace(0, np.max(zs), num=1500)
    w =  w_de(z, w0, wa)
    integral_values = cumulative_trapezoid(3*(1 + w)/(1 + z), z, initial=0)
    return np.exp(np.interp(zs, z, integral_values))


# Flat
def integral_of_e_z(z, Omega_m, w0, wa):
    z_grid = np.linspace(0, np.max(z), num=1500)
    sum = 1 + z_grid
    H_over_H0 = np.sqrt(Omega_m * sum**3 + (1 - Omega_m) * rho_de(z_grid, w0, wa))
    integral_values = cumulative_trapezoid(1/H_over_H0, z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)


def lcdm_apparent_mag(z, params):
    [M, Omega_m] = params
    a0_over_ae = 1 + z
    luminosity_distance = a0_over_ae * C * integral_of_e_z(z=z, Omega_m=Omega_m, w0=-1, wa=0)
    return M + 25 + 5 * np.log10(luminosity_distance)


def wcdm_apparent_mag(z, params):
    [M, Omega_m, w0, wa] = params
    a0_over_ae = 1 + z
    luminosity_distance = a0_over_ae * C * integral_of_e_z(z=z, Omega_m=Omega_m, w0=w0, wa=wa)
    return M + 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params):
    delta = apparent_mag_values - wcdm_apparent_mag(z_values, params)
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (-30, -27), # M
    (0, 1), # Ωm
    (-2, 0), # w0
    (-4, 1), # wa
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
    n_steps = steps_to_discard + 2000
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))

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

    [
        [M0_16, M0_50, M0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84]
    ] = np.percentile(samples, [16, 50, 84], axis=0).T

    best_fit_params = [M0_50, omega_50, w0_50, wa_50]

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
    omega_label = f"{omega_50:.4f} +{omega_84-omega_50:.4f}/-{omega_50-omega_16:.4f}"
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f}/-{w0_50-w0_16:.4f}"
    wa_label = f"{wa_50:.4f} +{wa_84-wa_50:.4f}/-{wa_50-wa_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.4f} - {z_values[-1]:.4f}")
    print_color("Sample size", len(z_values))
    print_color("M0", M0_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("wa", wa_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Reduced chi squared", chi_squared(best_fit_params)/ (len(z_values) - len(best_fit_params)))

    # Plot the data and the fit
    labels = [r"$M_0$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 12},
        smooth=1.5,
        smooth1d=1.5,
        bins=50,
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
        y=apparent_mag_values - M0_50,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=predicted_apparent_mag - M0_50,
        label=f"Best fit: $\Omega_m$={omega_50:.4f}, $M_0$={M0_50:.4f}",
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
M0: -28.5748 +0.0108/-0.0106 (M(0.7)=-19.3495)
Ωm: 0.2993 +0.0225/-0.0214
R-squared: 99.71 %
RMSD (mag): 0.143
Skewness of residuals: 0.191
kurtosis of residuals: 0.698
Reduced chi squared: 0.9817

=============================

wCDM
M0: -28.5778 +0.0144/-0.0146 (M(0.7)=-19.3509)
Ωm: 0.3227 +0.0650/-0.0802
w0: -1.0687 +0.1983/-0.2320
R-squared: 99.71 %
RMSD (mag): 0.143
Skewness of residuals: 0.196
kurtosis of residuals: 0.698
Reduced chi squared: 0.9826

=============================

Flat w0 + (wa - w0) * tanh(0.5*(1 + z - 1 / (1 + z)))
M0: -28.5764 +0.0163/-0.0170 (M(0.7)=-19.3501)
Ωm: 0.3587 +0.0587/-0.1094
w0: -1.0583 +0.1843/-0.2275
wa: -1.6679 +1.3440/-1.4695
R-squared (%): 99.71
RMSD (mag): 0.144
Skewness of residuals: 0.198
kurtosis of residuals: 0.706
Reduced chi squared: 0.9854

==============================

Flat w0 + (wa - w0) * tanh(0.5*((1 + z)**2 - 1))
M0: -28.5756 +0.0172/-0.0168 (M(0.7)=-19.3503)
Ωm: 0.3722 +0.0562/-0.1115
w0: -1.0619 +0.1874/-0.2247
wa: -1.7662 +1.1670/-1.3553
R-squared (%): 99.70
RMSD (mag): 0.144
Skewness of residuals: 0.197
kurtosis of residuals: 0.708
Reduced chi squared: 0.9860

*****************************
Dataset: Pantheon+ (2022)
z range: 0.0102 - 2.2614
Sample size: 1590
M0 contains Hubble constant and absolute magnitude
*****************************

ΛCDM
M0: -28.5767 +0.0070/-0.0068 (M(0.7)=-19.3502)
Ωm: 0.3310 +0.0181/-0.0178
w0: -1
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.091
kurtosis of residuals: 1.583
Reduced chi squared: 0.8840

=============================

wCDM
M0: -28.5734 +0.0088/-0.0089 (M(0.7)=-19.3489)
Ωm: 0.2986 +0.0618/-0.0745
w0: -0.9180 +0.1430/-0.1575
R-squared: 99.74 %
RMSD (mag): 0.154
Skewness of residuals: 0.082
kurtosis of residuals: 1.590
Reduced chi squared: 0.8843

=============================

Flat w0 + (wa - w0) * tanh(z)
M0: -28.5728 +0.0095/-0.0097 (M(0.7)=-19.3484)
Ωm: 0.3414 +0.0799/-0.1457
w0: -0.9326 +0.1580/-0.1713
wa: -1.2474 +0.8452/-1.4036
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.077
kurtosis of residuals: 1.600
Reduced chi squared: 0.8863

Flat w0 + (wa - w0) * tanh(0.5*((1 + z)**2 - 1))
M0: -28.5729 +0.0095/-0.0094 (M(0.7)=-19.3485)
Ωm: 0.3448 +0.0791/-0.1372
w0: -0.9443 +0.1573/-0.1582
wa: -1.2489 +0.7560/-1.2065
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.078
kurtosis of residuals: 1.598
Reduced chi squared: 0.8858

=============================

Flat w0 + (wa - w0) * tanh(0.5*(1 + z - 1 / (1 + z)))
M0: -28.5734 +0.0097/-0.0096 (M(0.7)=-19.3489)
Ωm: 0.3331 +0.0817/-0.1431
w0: -0.9274 +0.1540/-0.1644
wa: -1.1959 +0.9122/-1.5201
R-squared (%): 99.74
RMSD (mag): 0.154
Skewness of residuals: 0.078
kurtosis of residuals: 1.598
Reduced chi squared: 0.8861
"""
