import numpy as np
import emcee
import corner
from scipy.integrate import quad
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'

# Speed of light in km/s
c = 299792.458

# Planck rs = 147.18 ± 0.29 Mpc, h0 = 67.37 ± 0.54

# Source: https://github.com/CobayaSampler/bao_data/blob/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt
data = np.genfromtxt(
    path_to_data + "data.txt",
    dtype=[("z", float), ("value", float), ("quantity", "U10")],
    delimiter=" ",
    names=True,
)
cov_matrix = np.loadtxt(path_to_data + "covariance.txt", delimiter=" ", dtype=float)
inv_cov_matrix = np.linalg.inv(cov_matrix)

def plot_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(cov_matrix))

    # Compute R squared
    residuals = observed_values - model_predictions(params)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((observed_values - np.mean(observed_values))**2)
    R_squared = 1 - SS_res/SS_tot

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals ** 2))

    print(f"\033[92mR^2: {R_squared:.4f}\033[0m")
    print(f"\033[92mRMSD: {rmsd:.4f}\033[0m")

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    r_d, omega_m, w0 = params
    z_smooth = np.linspace(min(z_values), max(z_values), 100)
    plt.figure(figsize=(8, 6))
    for q in unique_quantities:
        mask = quantity_types == q
        plt.errorbar(
            x=z_values[mask],
            y=observed_values[mask],
            yerr=errors[mask],
            fmt='.',
            color=colors[q],
            label=f"Data: {q}",
            capsize=2,
            linestyle="None",
        )
        model_smooth = []
        for z in z_smooth:
            if q == "DV_over_rs":
                model_smooth.append(DV_z(z, params)/(100*r_d))
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, params)/(100*r_d))
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, params))/(100*r_d))
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO Data vs Model ($r_d * h$={r_d:.2f}, $\Omega_M$={omega_m:.4f}, $w_0$={w0:.4f} with errors")
    plt.show()


w_inf = -1/3

def H_z(z, params):
    _, Omega_m, w0 = params
    alpha = (w_inf + 1)/(w_inf - w0)
    return np.sqrt(Omega_m*(1 + z)**3 + (1 - Omega_m)*((1 + z)/(1 + z/alpha))**(3*(1 + w_inf)))


def DM_z(z, params):
    integral, _ = quad(lambda zp: c / H_z(zp, params), 0, z)
    return integral


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)


def model_predictions(params):
    r_d_x_h0 = params[0] * 100
    predictions = []
    for z, _, quantity in data:
        if quantity == "DV_over_rs":
            predictions.append(DV_z(z, params) / r_d_x_h0)
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params) / r_d_x_h0)
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params)) / r_d_x_h0)
    return np.array(predictions)


bounds = np.array([
    (85, 115), # r_d x h
    (0, 1), # Ωm
    (-2, -0.5), # w0
])


def chi_squared(params):
    delta = data['value'] - model_predictions(params)
    return delta.T @ inv_cov_matrix @ delta


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
    nwalkers = 300
    burn_in = 100
    nsteps = 2500 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
      initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    rd_percentiles = np.percentile(samples[:, 0], [16, 50, 84], axis=0)
    rd_50, rd_err = rd_percentiles[1], 0.5 * (rd_percentiles[2] - rd_percentiles[0])
    omega_percentiles = np.percentile(samples[:, 1], [16, 50, 84], axis=0)
    omega_50, omega_err = omega_percentiles[1], 0.5 * (omega_percentiles[2] - omega_percentiles[0])
    w0_percentiles = np.percentile(samples[:, 2], [16, 50, 84], axis=0)
    w0_50, w0_err = w0_percentiles[1], 0.5 * (w0_percentiles[2] - w0_percentiles[0])

    best_fit = [rd_50, omega_50, w0_50]

    print(f"\033[92mr_d*h: {rd_50:.4f} ± {rd_err:.4f}\033[0m")
    print(f"\033[92mΩm: {omega_50:.4f} ± {omega_err:.4f}\033[0m")
    print(f"\033[92mw0: {w0_50:.4f} ± {w0_err:.4f}\033[0m")
    print(f"\033[92mChi squared: {chi_squared(best_fit):.4f}\033[0m")
    print(f"\033[92mDegrees of freedom: {data['value'].size - len(best_fit)}\033[0m")
    plot_predictions(best_fit)

    labels = [r"$r_d$", f"$\Omega_m$", r"$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4f",
        smooth=2,
        smooth1d=2,
        bins=50,
    )
    plt.show()

    fig, axes = plt.subplots(ndim, figsize=(10, 7))
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit[i], color='white', linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM model
r_d*h: 101.5185 ± 0.7336
Ωm: 0.2978 ± 0.0086
Chi squared: 10.2724
Degrees of freedom: 11
R^2: 0.9987
RMSD: 0.3048

==============================

Flat wCDM model
r_d*h: 99.8055 ± 1.7060
Ωm: 0.2970 ± 0.0088
w0: -0.9155 ± 0.0775
Chi squared: 9.1190
Degrees of freedom: 10
R^2: 0.9989
RMSD: 0.2787

==============================

Modified Flat waw0CDM
r_d*h: 99.4816 ± 1.8971
Ωm: 0.3012 ± 0.0093
w0: -0.8936 ± 0.0915
Chi squared: 8.8604
Degrees of freedom: 10
R^2: 0.9990
RMSD: 0.2749
"""
