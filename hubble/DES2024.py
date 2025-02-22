import emcee
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2024DES.data import get_data

legend, z_values, distance_modulus_values, cov_matrix = get_data()

# Speed of light (km/s)
C = 299792.458

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Hubble constant km/s/Mpc as assumed in the dataset
h0 = 70

# ΛCDM - flat
def integral_of_e_z(zs, omega_m):
    i = 0
    res = np.empty((len(zs),))
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([(1 / np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m))) for z in z_axis], x=z_axis)
        res[i] = integ
        i = i + 1
    return res


# wCDM - flat
def integral_of_e_z_w(zs, omega_m, w):
    i = 0
    res = np.empty((len(zs),))
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([(1 / np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m) * (1 + z) ** (3 * (1 + w)))) for z in z_axis], x=z_axis)
        res[i] = integ
        i = i + 1
    return res


def lcdm_distance_modulus(z, params):
    [omega_m] = params
    a0_over_ae = 1 + z
    comoving_distance = (C / h0) * integral_of_e_z(z, omega_m)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def wcdm_distance_modulus(z, params):
    [omega_m, w] = params
    a0_over_ae = 1 + z
    comoving_distance = (C / h0) * integral_of_e_z_w(z, omega_m, w)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


# Distance modulus for modified matter-dominated, flat universe:
def model_distance_modulus(z, params):
    [p] = params
    a0_over_ae = (1 + z)**(1 / (1 - p))
    luminosity_distance = (2 * C * (1 - p) / h0) * (a0_over_ae - np.sqrt(a0_over_ae))
    return 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params):
    """ 
    Computes modified likelihood to marginalize over M 
    (Wood-Vasey et al. 2001, Appendix A9-A12)
    """
    delta = distance_modulus_values - model_distance_modulus(z_values, params)
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov_matrix @ deltaT)      # First term: (Δ^T C^-1 Δ)
    B = np.sum(delta @ inv_cov_matrix)                   # Second term: B
    C = np.sum(inv_cov_matrix)                           # Third term: C
    chi2 = chit2 - (B**2 / C) + np.log(C / (2 * np.pi))  # Full modified chi2
    return chi2


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (0.1, 0.6), # p
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
    steps_to_discard = 100
    n_dim = len(bounds)
    n_walkers = 60
    n_steps = steps_to_discard + 500
    initial_pos = np.zeros((n_walkers, n_dim))

    for dim, (lower, upper) in enumerate(bounds):
      initial_pos[:, dim] = np.random.uniform(lower, upper, n_walkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            log_probability,
            pool=pool
        )
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=steps_to_discard, flat=True)

    p_samples = samples[:, 0]

    one_sigma_quantile = np.array([16, 50, 84])
    [p_16, p_50, p_84] = np.percentile(p_samples, one_sigma_quantile)

    best_fit_params = [p_50]

    predicted_distance_modulus_values = model_distance_modulus(z_values, best_fit_params)
    residuals = distance_modulus_values - predicted_distance_modulus_values

    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(distance_modulus_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((distance_modulus_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    # Print the values in the console
    p_label = f"{p_50:.4f} +{p_84-p_50:.4f}/-{p_50-p_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("p", p_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Chi squared", chi_squared(best_fit_params))

    # Posterior distribution
    sns.histplot(p_samples, kde=True, color="skyblue", stat="density", bins=50)
    plt.axvline(p_16, color='red', linestyle='--', label=r"-1$\sigma$")
    plt.axvline(p_50, color='green', linestyle='-', label=r"$p50$")
    plt.axvline(p_84, color='red', linestyle='--', label=r"+1$\sigma$")
    plt.xlabel("p")
    plt.ylabel("Density")
    plt.title("Posterior Distribution")
    plt.legend()
    plt.show()

    # Plot chains for each parameter
    fig, axes = plt.subplots(n_dim, figsize=(10, 7))
    axes.plot(chains_samples[:, :, 0], color='black', alpha=0.3)
    axes.set_ylabel(r"$p$")
    axes.set_xlabel("chain step")
    axes.axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
    plt.show()

    y_err = np.sqrt(cov_matrix.diagonal())

    # Plot the predictions
    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=y_err,
        y_model=predicted_distance_modulus_values,
        label=f"p={p_label}",
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
Dataset: Union2.1
z range: 0.015 - 1.414
Sample size: 580
Chi squared: 553.2905
p: 0.3410 +0.0193/-0.0210
R-squared (%): 99.26
RMSD (mag): 0.276
Skewness of residuals: 1.423
kurtosis of residuals: 8.377

==============================
Dataset: DES-SN5YR
z range: 0.025 - 1.121
Sample size: 1829
Chi squared: 1654.7499
p: 0.3267 +0.0077/-0.0078
R-squared (%): 98.24
RMSD (mag): 0.277
Skewness of residuals: 3.416
kurtosis of residuals: 25.859

==============================
Flat ΛCDM

Dataset: DES-SN5YR
z range: 0.025 - 1.121
Sample size: 1829
Chi squared: 1649.4596
Ωm: 0.3507 +0.0170/-0.0167
R-squared (%): 98.35
RMSD (mag): 0.268
Skewness of residuals: 3.409
kurtosis of residuals: 25.926
"""
