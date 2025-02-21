import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2023union3.data import get_data

legend, z_values, distance_moduli_values, cov_matrix = get_data()

inverse_cov = np.linalg.inv(cov_matrix)

# Speed of light (km/s)
C = 299792.458

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

def model_lcdm_distance(z, omega_m, h0):
    normalized_h0 = h0 * 100
    a0_over_ae = 1 + z
    comoving_distance = (C / normalized_h0) * integral_of_e_z(zs = z, omega_m=omega_m)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)

# Distance modulus for alternative matter-dominated, flat universe:
def model_distance_modulus(z, p, h0):
    normalized_h0 = h0 * 100
    a0_over_ae = (1 + z)**(1 / (1 - p))
    comoving_distance = 2 * (C / normalized_h0) * (1 - p) * (1 - 1 / np.sqrt(a0_over_ae))
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def chi_squared(params, z, observed_mag):
    [h0, p0] = params
    delta = observed_mag - model_distance_modulus(z=z, p=p0, h0=h0)
    return delta.T @ inverse_cov @ delta


def log_likelihood(params, z, observed_mag):
    return -0.5 * chi_squared(params, z, observed_mag)


h0_bounds = (0.5, 1)
p_bounds = (0.1, 0.6)


def log_prior(params):
    [h0, p0] = params
    if h0_bounds[0] < h0 < h0_bounds[1] and p_bounds[0] < p0 < p_bounds[1]:
        return 0.0
    return -np.inf


def log_probability(params, z, observed_mag):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params, z, observed_mag)


def main():
    n_dim = 2
    n_walkers = 40
    n_steps = 4100
    initial_pos = np.zeros((n_walkers, n_dim))
    initial_pos[:, 0] = np.random.uniform(*h0_bounds, n_walkers)
    initial_pos[:, 1] = np.random.uniform(*p_bounds, n_walkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=n_dim,
            log_prob_fn=log_probability,
            args=(z_values, distance_moduli_values),
            pool=pool
        )
        sampler.run_mcmc(initial_pos, n_steps, progress=True)


    # Extract samples
    discarded_steps = 100
    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=discarded_steps, flat=True)

    try:
        tau = sampler.get_autocorr_time()
        effective_samples = n_steps * n_walkers / np.max(tau)
        print(f"Estimated autocorrelation time: {tau}")
        print(f"Effective samples: {effective_samples:.2f}")
    except Exception as e:
        print("Could not calculate the autocorrelation time")

    h0_samples = samples[:, 0]
    p_samples = samples[:, 1]

    [h0_16, h0_50, h0_84] = np.percentile(h0_samples, [16, 50, 84])
    [p_16, p_50, p_84] = np.percentile(p_samples, [16, 50, 84])

    predicted_mag = model_distance_modulus(z=z_values, p=p_50, h0=h0_50)
    residuals = distance_moduli_values - predicted_mag

    skewness = skew(residuals)
    kurt = kurtosis(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(distance_moduli_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((distance_moduli_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    # Print the values in the console
    h0_label = f"{h0_50:.4f} +{h0_84-h0_50:.4f}/-{h0_50-h0_16:.4f}"
    p_label = f"{p_50:.4f} +{p_84-p_50:.4f}/-{p_50-p_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("p", p_label)
    print_color("h0", h0_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("kurtosis of residuals", f"{kurt:.3f}")
    print_color("Reduced chi squared", chi_squared([h0_50, p_50], z_values, distance_moduli_values)/ (len(z_values) - 2))

    # Plot the data and the fit
    corner.corner(
        samples,
        labels=[r"$h_0$", r"$p$"],
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
    axes[0].axvline(x=discarded_steps, color='red', linestyle='--', alpha=0.5)
    axes[1].plot(chains_samples[:, :, 1], color='black', alpha=0.3)
    axes[1].set_ylabel(r"$p$")
    axes[1].set_xlabel("chain step")
    axes[1].axvline(x=discarded_steps, color='red', linestyle='--', alpha=0.5)
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=predicted_mag,
        label=f"Distance modulus (mag): $p$={p_50:.4f} & $h_0$={h0_50:.4f}",
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
Dataset: Union 3 Bins
z range: 0.050 - 2.262

=============================

Alternative
Effective chain samples: 22580

p: 0.3104 +0.0126/-0.0132
h0: 0.7148 +0.0297/-0.0284
R-squared (%):  99.87
RMSD (mag): 0.080
Skewness of residuals: -3.300
kurtosis of residuals: 11.512
Reduced chi squared: 1.300

=============================

ΛCDM
Effective chain samples: 1537

Ωm: 0.35602 +0.02765/-0.02555
h0: 0.72471 +0.03005/-0.02864
R-squared (%): 99.95
RMSD (mag): 0.048
Skewness of residuals: 0.554
kurtosis of residuals: 0.689
Reduced chi squared: 1.198
"""
