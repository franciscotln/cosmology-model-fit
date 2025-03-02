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

r_d = 147.05  # Fiducial value from Planck

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

    H0, w0, wa = params
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO Data vs Model (H₀={H0:.3f}, w0={w0:.3f}, wa={wa:.3f}) with Errors")
    plt.show()


def H_z(z, params):
    H0, w0, wa = params
    radiation_limit = 1 / 3
    w_z = radiation_limit + (w0 - radiation_limit) * np.exp(-wa * z)
    return H0 * np.sqrt((1 + z) ** (3 * (1 + w_z)))


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
    (50, 80), # H0
    (-2, 0.5), # w0
    (-0.5, 1.5) # wa
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
    nwalkers = 100
    nsteps = 3100
    burn_in = 100
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

    samples = sampler.get_chain(discard=burn_in, flat=True)

    H0_percentiles = np.percentile(samples[:, 0], [16, 50, 84], axis=0)
    H0_50, H0_err = H0_percentiles[1], 0.5 * (H0_percentiles[2] - H0_percentiles[0])
    w0_percentiles = np.percentile(samples[:, 1], [16, 50, 84], axis=0)
    w0_50, w0_err = w0_percentiles[1], 0.5 * (w0_percentiles[2] - w0_percentiles[0])
    wa_percentiles = np.percentile(samples[:, 2], [16, 50, 84], axis=0)
    wa_50, wa_err = wa_percentiles[1], 0.5 * (wa_percentiles[2] - wa_percentiles[0])

    best_fit = [H0_50, w0_50, wa_50]

    print(f"H0 = {H0_50:.2f} ± {H0_err:.2f}, w0 = {w0_50:.3f} ± {w0_err:.3f}, wa = {wa_50:.3f} ± {wa_err:.3f}")
    print("chi squared: ", chi_squared(best_fit))

    corner.corner(
        samples,
        labels=[r"$H_0$", r"w_0", r"$w_a$"],
        percentiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        smooth=2,
        smooth1d=2,
        bins=50,
    )
    plt.show()

    plot_predictions(best_fit)

if __name__ == "__main__":
    main()


"""
ΛCDM model
H0 = 69.29 ± 0.85 km/s/Mpc
Omega_m = 0.294 ± 0.014
chi squared: 12.74
degree of freedom: 10

==============================

Fluid model
H0 = 67.17 ± 1.49 km/s/Mpc
w0 = -0.559 ± 0.053
wa = 0.147 ± 0.021
chi squared: 10.98
degree of freedom: 9
"""
