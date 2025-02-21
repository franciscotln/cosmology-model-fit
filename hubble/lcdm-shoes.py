import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data import get_data

legend, z_values, distance_modulus_values, cov_matrix = get_data()
sigma_distance_moduli = np.sqrt(cov_matrix.diagonal())

# inverse covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Speed of light (km/s)
C = 299792.458


# Flat ΛCDM
def integral_of_e_z(zs, omega_m):
    i = 0
    res = np.empty((len(zs),), dtype=np.float64)
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([(1 / np.sqrt(omega_m * (1 + z) ** 3 + 1 - omega_m)) for z in z_axis], x=z_axis)
        res[i] = integ
        i = i + 1
    return res


def model_distance_modulus(z, h0, omega_m):
    normalized_h0 = 100 * h0
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(zs = z, omega_m=omega_m)
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)


def chi_squared(params, z, observed_mu):
    h0, omega_m = params
    delta = observed_mu - model_distance_modulus(z, h0, omega_m)
    return delta.T @ inv_cov_matrix @ delta


def log_likelihood(params, z, observed_mu):
    return -0.5 * chi_squared(params, z, observed_mu)

h0_bounds = (0.40, 0.9)
omega_m_bounds = (0.1, 0.6)

def log_prior(params):
    h0, omega_m = params
    if h0_bounds[0] < h0 < h0_bounds[1] and omega_m_bounds[0] < omega_m < omega_m_bounds[1]:
        return 0.0
    return -np.inf


def log_probability(params, z, observed_mu):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params, z, observed_mu)


def main():
    steps_to_discard = 50
    n_dim = 2
    n_walkers = 10
    n_steps = steps_to_discard + 500
    initial_pos = np.zeros((n_walkers, n_dim))
    initial_pos[:, 0] = np.random.uniform(*h0_bounds, n_walkers)
    initial_pos[:, 1] = np.random.uniform(*omega_m_bounds, n_walkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability,
            args=(z_values, distance_modulus_values),
            pool=pool
        )
        sampler.run_mcmc(initial_pos, n_steps, progress=True)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=50, flat=True)

    h0_samples = samples[:, 0]
    omega_m_samples = samples[:, 1]

    [h0_16, h0_50, h0_84] = np.percentile(h0_samples, [16, 50, 84])
    [omega_m_16, omega_m_50, omega_m_84] = np.percentile(omega_m_samples, [16, 50, 84])

    # Calculate residuals
    predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0_50, omega_m=omega_m_50)
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

    # Compute correlations
    spearman_corr, _ = stats.spearmanr(h0_samples, omega_m_samples)

    # Print the values in the console
    h0_label = f"{h0_50:.5f} +{h0_84-h0_50:.5f}/-{h0_50-h0_16:.5f}"
    omega_m_label = f"{omega_m_50:.5f} +{omega_m_84-omega_m_50:.5f}/-{omega_m_50-omega_m_16:.5f}"
    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("Estimated h = H0 / 100 (km/s/Mpc)", h0_label)
    print_color("Ωm", f"{omega_m_label}")
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurtosis:.3f}")
    print_color("Spearman correlation", f"{spearman_corr:.3f}")
    print_color("Chi squared", chi_squared([h0_50, omega_m_50], z_values, distance_modulus_values))
    print_color("Degrees of freedom", len(z_values) - 2)

    # Plot the data and the fit
    corner.corner(
        samples,
        labels=[r"$h_0$", r"$Ω_M$"],
        truths=[h0_50, omega_m_50],
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
    )
    plt.show()

    # Plot results: chains for each parameter
    fig, axes = plt.subplots(2, figsize=(10, 7))
    axes[0].plot(chains_samples[:, :, 0], color='black', alpha=0.3)
    axes[0].set_ylabel(r"$h_0$")
    axes[0].set_xlabel("chain step")
    axes[0].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
    axes[1].plot(chains_samples[:, :, 1], color='black', alpha=0.3)
    axes[1].set_ylabel(r"$Ω_M$")
    axes[1].set_xlabel("chain step")
    axes[1].axvline(x=steps_to_discard, color='red', linestyle='--', alpha=0.5)
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_modulus_values,
        y_err=sigma_distance_moduli,
        y_model=predicted_distance_modulus_values,
        label=f"$h_0$={h0_label} & $Ω_M$={omega_m_label}",
        x_scale="log"
    )

    # Plot the residual analysis
    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=sigma_distance_moduli,
        bins=40
    )

if __name__ == '__main__':
    main()

"""
-- RESULTS WITH SHOES - LCDM --
Estimated autocorrelation time: [28.29 27.12]
Effective samples: 724
Pearson correlation: -0.86033
Chi squared:  1752.51

Dataset:  Pantheon+SHOES
z range:  0.001 - 2.261
Sample size:  1701
Estimated h = H0 / 100 (km/s/Mpc):  0.72841 ± 0.00230
Ωm:  0.36127 ± 0.01863
R-squared (%):  99.74
RMSD (mag):  0.172
Skewness of residuals:  0.108
kurtosis of residuals:  4.308

-- RESULTS WITHOUT SHOES - LCDM --
Estimated autocorrelation time: [27.19 22.39]
Effective samples: 735.60
Pearson correlation: -0.84528

Dataset:  Pantheon+
z range:  0.010 - 2.261
Sample size:  1590
Estimated h = H0 / 100 (km/s/Mpc):  0.73235 ± 0.00246
Ωm:  0.33167 ± 0.01842
R-squared (%):  99.74
RMSD (mag):  0.154
Skewness of residuals:  -0.091
kurtosis of residuals:  1.584
"""
