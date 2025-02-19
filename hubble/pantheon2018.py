import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2018pantheon.data import get_data

legend, z_values, apparent_mag_values, _, cov_matrix = get_data()

# Speed of light (km/s)
C = 299792.458

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# ΛCDM
def integral_of_e_z(zs, omega_m):
    i = 0
    res = np.empty((len(zs),), dtype=np.float64)
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([(1 / np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m))) for z in z_axis], x=z_axis)
        res[i] = integ
        i = i + 1
    return res

def model_lcdm_apparent_mag(z, omega_m, M):
    a0_over_ae = 1 + z
    comoving_distance = C * integral_of_e_z(zs = z, omega_m=omega_m)
    return M + 25 + 5 * np.log10(a0_over_ae * comoving_distance)

# Apparent magnitude for alternative matter-dominated, flat universe:
def model_apparent_mag(z, p, M):
    a0_over_ae = (1 + z)**(1 / (1 - p))
    comoving_distance = 2 * C * (1 - p) * (1 - 1 / np.sqrt(a0_over_ae))
    return M + 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def chi_squared(params, z, observed_mag):
    [M0, p0] = params
    delta = observed_mag - model_apparent_mag(z=z, p=p0, M=M0)
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params, z, observed_mag):
    return -0.5 * chi_squared(params, z, observed_mag)


def log_prior(params):
    [M0, p0] = params
    if -30 < M0 < -27 and 0.2 < p0 < 0.6:
        return 0.0
    return -np.inf


def log_probability(params, z, observed_mag):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params, z, observed_mag)


def main():
    n_dim = 2
    n_walkers = 20
    n_steps = 2100
    initial_pos = np.zeros((n_walkers, n_dim))
    initial_pos[:, 0] = np.random.uniform(-30, -27, n_walkers)
    initial_pos[:, 1] = np.random.uniform(0.2, 0.6, n_walkers)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            log_probability,
            args=(z_values, apparent_mag_values),
            pool=pool
        )
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    # Extract samples
    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=100, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        effective_samples = n_steps * n_walkers / np.max(tau)
        print(f"Estimated autocorrelation time: {tau}")
        print(f"Effective samples: {effective_samples:.2f}")
    except Exception as e:
        print("Could not calculate the autocorrelation time")

    # Calculate the posterior means and uncertainties
    M0_samples = samples[:, 0]
    p_samples = samples[:, 1]

    [M0_16, M0_50, M0_84] = np.percentile(M0_samples, [16, 50, 84])
    [p_16, p_50, p_84] = np.percentile(p_samples, [16, 50, 84])

    corner.corner(
        samples,
        labels=[r"$M_0$", r"$p$"],
        truths=[M0_50, p_50],
        show_titles=True,
        title_fmt=".5f",
        title_kwargs={"fontsize": 12},
    )
    plt.show()

    # Plot results: chains for each parameter
    fig, axes = plt.subplots(2, figsize=(10, 7))
    axes[0].plot(chains_samples[:, :, 0], color='black', alpha=0.3)
    axes[0].set_ylabel(r"$M_0$")
    axes[1].plot(chains_samples[:, :, 1], color='black', alpha=0.3)
    axes[1].set_ylabel(r"$p$")
    plt.show()

    # Calculate residuals
    predicted_apparent_mag = model_apparent_mag(z=z_values, p=p_50, M=M0_50)
    residuals = apparent_mag_values - predicted_apparent_mag

    # Compute skewness
    skewness = stats.skew(residuals)

    # Compute kurtosis
    kurtosis = stats.kurtosis(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(apparent_mag_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((apparent_mag_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    # Print the values in the console
    M0_label = f"{M0_50:.5f} +{M0_84-M0_50:.5f}/-{M0_50-M0_16:.5f}"
    p_label = f"{p_50:.5f} +{p_84-p_50:.5f}/-{p_50-p_16:.5f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("p", p_label)
    print_color("M0", M0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Reduced chi squared", chi_squared([M0_50, p_50], z_values, apparent_mag_values)/ (len(z_values) - 2))

    # Plot the data and the fit
    plot_predictions(
        legend=legend,
        x=z_values,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=model_apparent_mag(z=z_values, p=p_50, M=M0_50),
        label=f"Apparent mag: $p$={p_50:.4f} and $M_0$={M0_50:.4f}",
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
Dataset: Pantheon2018
z range: 0.010 - 2.260
Sample size: 1048
M0 contains Hubble constant and absolute magnitude

=============================

Alternative
M0: -28.549 +/-0.011
p: 0.349 +0.010/-0.011
R-squared (%): 99.69
RMSD (mag): 0.147
Skewness of residuals: 0.082
kurtosis of residuals: 0.786
Reduced chi squared:  0.991

=============================

ΛCDM
M0: -28.5749 +0.0116/-0.0096
Ωm: 0.299 +0.022/-0.020
R-squared (%): 99.71
RMSD (mag): 0.143
Skewness of residuals: 0.186
kurtosis of residuals: 0.687
Reduce chi squared: 0.9815
"""
