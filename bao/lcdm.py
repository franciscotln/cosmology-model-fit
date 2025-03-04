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

r_d = 147.18  # Fiducial value from Planck rs = 147.18 ± 0.29 Mpc, h0 = 0.6737 ± 0.0054

# Source: https://github.com/CobayaSampler/bao_data/blob/master/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt
data = np.genfromtxt(
    path_to_data + "data.txt",
    dtype=[("z", float), ("value", float), ("quantity", "U10")],
    delimiter=" ",
    names=True,
)
cov_matrix = np.loadtxt(path_to_data + "covariance.txt", delimiter=" ", dtype=float)
inv_cov_matrix = np.linalg.inv(cov_matrix)

def plot_predictions(params):
    O_data = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(cov_matrix))

    # Compute R squared
    SS_res = np.sum((O_data - model_predictions(params))**2)
    SS_tot = np.sum((O_data - np.mean(O_data))**2)
    R_squared = 1 - SS_res/SS_tot
    print("R^2: ", R_squared)

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    z_smooth = np.linspace(min(z_values), max(z_values), 100)
    plt.figure(figsize=(8, 6))
    for q in unique_quantities:
        mask = quantity_types == q
        plt.errorbar(
            x=z_values[mask],
            y=O_data[mask],
            yerr=errors[mask],
            fmt='.',
            color=colors[q],
            label=f"Data: {q}",
            capsize=2,
            linestyle="None",
        )
        O_model_smooth = []
        for z in z_smooth:
            if q == "DV_over_rs":
                O_model_smooth.append(DV_z(z, params) / r_d)
            elif q == "DM_over_rs":
                O_model_smooth.append(DM_z(z, params) / r_d)
            elif q == "DH_over_rs":
                O_model_smooth.append((c / H_z(z, params)) / r_d)
        plt.plot(z_smooth, O_model_smooth, color=colors[q])

    H0, w0, wm = params
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO Data vs Model (H₀={H0:.4f}, w0={w0:.4f}, wm={wm:.4f}) with Errors")
    plt.show()


def H_z(z, params):
    H0, w0, wm = params
    normalized_h0 = 100 * H0
    w_inf = 1/3
    w_z = w_inf - w_inf * (1 - (w0/w_inf))**(1 - wm * z)
    return normalized_h0 * np.sqrt((1 + z) ** (3 * (1 + w_z)))


def H_z_lcdm(z, params):
    H0, Omega_m = params
    normalized_h0 = 100 * H0
    return normalized_h0 * np.sqrt(Omega_m * (1 + z) ** 3 + (1 - Omega_m))


def DM_z(z, params):
    integral, _ = quad(lambda zp: c / H_z(zp, params), 0, z)
    return integral


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)


def model_predictions(params):
    predictions = []
    for z, _, quantity in data:
        if quantity == "DV_over_rs":
            predictions.append(DV_z(z, params) / r_d)
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params) / r_d)
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params)) / r_d)
    return np.array(predictions)


bounds = np.array([
    (0.55, 0.8), # h0
    (-1, -0.2), # w0
    (0, 0.8) # wm
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
    nwalkers = 400
    burn_in = 100
    nsteps = 2000 + burn_in
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

    H0_percentiles = np.percentile(samples[:, 0], [16, 50, 84], axis=0)
    H0_50, H0_err = H0_percentiles[1], 0.5 * (H0_percentiles[2] - H0_percentiles[0])
    w0_percentiles = np.percentile(samples[:, 1], [16, 50, 84], axis=0)
    w0_50, w0_err = w0_percentiles[1], 0.5 * (w0_percentiles[2] - w0_percentiles[0])
    wm_percentiles = np.percentile(samples[:, 2], [16, 50, 84], axis=0)
    wm_50, wm_err = wm_percentiles[1], 0.5 * (wm_percentiles[2] - wm_percentiles[0])

    best_fit = [H0_50, w0_50, wm_50]

    print(f"H0={H0_50:.4f} ± {H0_err:.4f}, w0={w0_50:.4f} ± {w0_err:.4f}, wm={wm_50:.4f} ± {wm_err:.4f}")
    print("chi squared: ", chi_squared(best_fit))

    labels = [r"$H_0$", r"w_0", r"$w_m$"]
    corner.corner(
        samples,
        labels=labels,
        percentiles=[0.16, 0.5, 0.84],
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

    plot_predictions(best_fit)

if __name__ == "__main__":
    main()


"""
ΛCDM model
H0 = 69.22 ± 0.86 km/s/Mpc
Omega_m = 0.2945 ± 0.0146
chi squared: 12.74
degree of freedom: 10
Reduced chi squared: 1.274
R squared: 99.57 %

==============================

Fluid model
H0 = 67.02 ± 1.50 km/s/Mpc
w0 = -0.5549 ± 0.0530
wm = 0.1482 ± 0.0133
chi squared: 10.98
degree of freedom: 9
Reduced chi squared: 1.220
R squared: 99.67 %
"""
