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

# Source: https://arxiv.org/pdf/2503.14738
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
    z_smooth = np.linspace(0, max(z_values), 100)
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
    plt.title(f"BAO Data vs Model ($r_d * h$={r_d:.2f}, $\Omega_M$={omega_m:.4f}, $w_0$={w0:.4f}) with errors")
    plt.show()

def H_z(z, params):
    _, omega_m, w0 = params
    sum = 1 + z
    return np.sqrt(omega_m * sum**3 + (1 - omega_m) * sum**3 * np.exp(3 * w0 * z))


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
    (80, 110), # r_d x h
    (0.1, 0.7), # omega_m
    (-1.2, -0.3), # w0
])


def chi_squared(params):
    delta = data['value'] - model_predictions(params)
    return np.dot(delta, np.dot(inv_cov_matrix, delta))


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
    burn_in = 500
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

    [
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, omega_50, w0_50]

    print(f"\033[92mr_d*h: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}\033[0m")
    print(f"\033[92mΩm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}\033[0m")
    print(f"\033[92mw0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}\033[0m")
    print(f"\033[92mChi squared: {chi_squared(best_fit):.4f}\033[0m")
    print(f"\033[92mDegrees of freedom: {data['value'].size - len(best_fit)}\033[0m")
    plot_predictions(best_fit)

    labels = [r"${r_d}\times{h}$", f"$\Omega_m$", r"$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
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
r_d*h: 101.54 ± 0.72
Ωm: 0.2974 +0.0086 -0.0084
Chi squared: 10.5169
Degrees of freedom: 11
R^2: 0.9987
RMSD: 0.3054

==============================

Flat wCDM model
r_d*h: 99.86 +1.74 -1.64
Ωm: 0.2968 +0.0089 -0.0087
w0: -0.9179 +0.0743 -0.0778
Chi squared: 9.3920
Degrees of freedom: 10
R^2: 0.9989
RMSD: 0.2800

==============================

Flat w0waCDM
r_d*h: 91.3913 +5.0396 -4.3708
Ωm: 0.3859 +0.0465 -0.0487
w0: -0.1861 +0.4432 -0.4417
wa: -2.7215 +1.5416 -1.5251
Chi squared: 5.7844
Degrees of freedom: 9
R^2: 0.9994
RMSD: 0.2029

===============================

Flat modified wCDM
r_d*h: 96.4053 +1.3539 -1.2771
Ωm: 0.3442 +0.0091 -0.0088
w0: -0.6884 +0.0463 -0.0494
Chi squared: 6.4777
Degrees of freedom: 10
R^2: 0.9993
RMSD: 0.2314
"""
