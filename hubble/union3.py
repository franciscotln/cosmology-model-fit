import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from scipy.integrate import cumulative_trapezoid
from multiprocessing import Pool
from .plotting import plot_predictions, print_color, plot_residuals
from y2023union3.data import get_data

legend, z_values, distance_moduli_values, cov_matrix = get_data()

inverse_cov = np.linalg.inv(cov_matrix)

# Speed of light (km/s)
C = 299792.458

# Hubble constant (km/s/Mpc)
H0 = 70


# Flat model
def integral_of_e_z(zs, params):
    _, omega_m, w0, _ = params
    z = np.linspace(0, np.max(zs), num=4000)
    sum = 1 + z
    h_over_h0 = np.sqrt(omega_m * sum**3 + (1 - omega_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))
    integral_values = cumulative_trapezoid(1/h_over_h0, z, initial=0)
    return np.interp(zs, z, integral_values)


def model_distance_modulus(z, params):
    deltaM = params[0]
    a0_over_ae = 1 + z
    comoving_distance = (C / H0) * integral_of_e_z(z, params)
    return deltaM + 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def chi_squared(params):
    delta = distance_moduli_values - model_distance_modulus(z_values, params)
    return delta.T @ inverse_cov @ delta


def log_likelihood(params):
    return -0.5 * chi_squared(params)


bounds = np.array([
    (-0.5, 0.5), # ΔM
    (0, 1), # Ωm
    (-2, 0.5), # w0
    (-10, 0) # wa
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
    n_dim = len(bounds)
    n_walkers = 200
    discarded_steps = 200
    n_steps = discarded_steps + 5000
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, n_steps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=discarded_steps, flat=True)

    [
        [deltaM_16, deltaM_50, deltaM_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84]
    ] = np.percentile(samples, [16, 50, 84], axis=0).T

    best_fit_params = [deltaM_50, omega_50, w0_50, wa_50]

    predicted_distances = model_distance_modulus(z_values, best_fit_params)
    residuals = distance_moduli_values - predicted_distances

    skewness = skew(residuals)

    # Calculate R-squared
    average_distance_modulus = np.mean(distance_moduli_values)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((distance_moduli_values - average_distance_modulus) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    delta_M_label = f"{deltaM_50:.4f} +{deltaM_84-deltaM_50:.4f}/-{deltaM_50-deltaM_16:.4f}"
    omega_label = f"{omega_50:.4f} +{omega_84-omega_50:.4f}/-{omega_50-omega_16:.4f}"
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f}/-{w0_50-w0_16:.4f}"
    wa_label = f"{wa_50:.4f} +{wa_84-wa_50:.4f}/-{wa_50-wa_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("ΔM", delta_M_label)
    print_color("Ωm", omega_label)
    print_color("w0", w0_label)
    print_color("wa", wa_label)
    print_color("R-squared (%)", f"{100 * r_squared:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skewness:.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit_params):.4f}")
    print_color("Reduced chi squared", chi_squared(best_fit_params)/ (len(z_values) - len(best_fit_params)))

    labels = [f"$\Delta_M$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]
    corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
        smooth=1.5,
        smooth1d=1.5,
        bins=50
    )
    plt.show()

    _, axes = plt.subplots(n_dim, figsize=(10, 7))
    if n_dim == 1:
        axes = [axes]
    for i in range(n_dim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3, lw=0.5)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=discarded_steps, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit_params[i], color='white', linestyle='--', alpha=0.5)
    plt.show()

    sigma_mu = np.sqrt(np.diag(cov_matrix))

    plot_predictions(
        legend=legend,
        x=z_values,
        y=distance_moduli_values,
        y_err=sigma_mu,
        y_model=predicted_distances,
        label=f"Best fit: $\Omega_m$={omega_50:.4f}, $w_0$={w0_50:.4f}, $w_a$={wa_50:.4f}",
        x_scale="log"
    )

    plot_residuals(
        z_values=z_values,
        residuals=residuals,
        y_err=sigma_mu,
        bins=40
    )

if __name__ == '__main__':
    main()

"""
*****************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*****************************

Flat ΛCDM

ΔM: -0.0698 +0.0878/-0.0874
Ωm: 0.3570 +0.0274/-0.0266
w0: -1
wa: 0
R-squared (%): 99.95
RMSD (mag): 0.050
Skewness of residuals: 0.582
Chi squared: 23.9594
degrees of freedom: 20

=============================

Flat wCDM

ΔM: -0.0601 +0.0876/-0.0876
Ωm: 0.2532 +0.0883/-0.1113
w0: -0.7483 +0.1568/-0.1893 (1.45 sigma)
wa: 0
R-squared (%): 99.94
RMSD (mag): 0.054
Skewness of residuals: -1.244
Chi squared: 22.1323
degrees of freedom: 19

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.0557 +0.0895/-0.0889
Ωm: 0.2873 +0.0602/-0.0643
w0: -0.7606 +0.1449/-0.1733 (1.50 sigma)
wa: 0
R-squared (%): 99.94
RMSD (mag): 0.054
Skewness of residuals: -1.103
Chi squared: 21.8772
degrees of freedom: 19

=============================

Flat w0waCDM
ΔM: -0.0326 +0.0913/-0.0907
Ωm: 0.4449 +0.0523/-0.0800
w0: -0.5456 +0.2772/-0.2322 (1.78 sigma)
wa: -4.3188 +2.6906/-3.1919 (1.47 sigma)
R-squared (%): 99.96
RMSD (mag): 0.042
Skewness of residuals: 0.724
Chi squared: 20.5888
degrees of freedom: 18
"""
