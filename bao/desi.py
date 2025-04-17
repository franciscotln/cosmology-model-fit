import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Speed of light in km/s
c = 299792.458

# Planck rs = 147.18 ± 0.29 Mpc, h0 = 67.37 ± 0.54

# Source: https://arxiv.org/pdf/2503.14738
# https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2
data = np.genfromtxt(
    "bao/raw-data/data.txt",
    dtype=[("z", float), ("value", float), ("quantity", "U10")],
    delimiter=" ",
    names=True,
)
cov_matrix = np.loadtxt("bao/raw-data/covariance.txt", delimiter=" ", dtype=float)
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

    print(f"R^2: {R_squared:.4f}")
    print(f"RMSD: {rmsd:.4f}")

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    r_d, omega_m, w0, wa = params
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
    plt.title(f"BAO model: $r_d x h$={r_d:.2f}, $\Omega_M$={omega_m:.4f}, $w_0$={w0:.4f}, $w_a$={wa:.4f}")
    plt.show()


def w_de(z, params):
    _, _, w0, wa = params
    return w0 + wa * (1 - np.exp(0.5 - 0.5 * (1 + z)**2))


def rho_de(z_input, params):
    z = np.linspace(0, np.max(z_input), num=2000)
    integral_values = cumulative_trapezoid(3*(1 + w_de(z, params))/(1 + z), z, initial=0)
    return np.exp(np.interp(z_input, z, integral_values))


def H_z(z, params):
    _, omega_m, _, _ = params
    sum = 1 + z
    return np.sqrt(omega_m * sum**3 + (1 - omega_m) * rho_de(z, params))


def DM_z(zs, params):
    z = np.linspace(0, np.max(zs), num=2000)
    return cumulative_trapezoid(c / H_z(z, params), z, initial=0)[-1]


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
    (70, 110), # r_d x h
    (0.2, 0.7), # Ωm
    (-3, 1), # w0
    (-4.5, 1.5), # wa
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
    nsteps = 4000 + burn_in
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
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, omega_50, w0_50, wa_50]

    print(f"r_d*h: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degs of freedom: {data['value'].size  - len(best_fit)}")

    plot_predictions(best_fit)

    labels = [r"${r_d}\times{h}$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]

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

    _, axes = plt.subplots(ndim, figsize=(10, 7))
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
w0: -1
wa: 0
Chi squared: 10.5169
Degrees of freedom: 11
R^2: 0.9987
RMSD: 0.3054

==============================

Flat wCDM model
r_d*h: 99.86 +1.74 -1.64
Ωm: 0.2968 +0.0089 -0.0087
w0: -0.9179 +0.0743 -0.0778 (1.08 sigma)
wa: 0
Chi squared: 9.3920
Degrees of freedom: 10
R^2: 0.9989
RMSD: 0.2800

==============================

Flat w0waCDM
r_d*h: 91.4074 +5.0672 -4.4132
Ωm: 0.3856 +0.0470 -0.0494
w0: -0.1896 +0.4515 -0.4462 (1.80 sigma)
wa: -2.7090 +1.5726 -1.5582 (1.73 sigma)
Chi squared: 5.6478
Degs of freedom: 9
R^2: 0.9994
RMSD: 0.2016

===============================

Flat w0 + wa * z
r_d*h: 93.4082 +4.1359 -3.7184
Ωm: 0.3729 +0.0367 -0.0395
w0: -0.4606 +0.3014 -0.2980 (1.80 sigma)
wa: -1.2049 +0.6832 -0.6762 (1.77 sigma)
Chi squared: 5.5936
Degs of freedom: 9
R^2: 0.9994
RMSD: 0.2035

===============================

Flat w0 + wa * np.tanh(z)
r_d*h: 92.3188 +4.2977 -4.0220
Ωm: 0.3770 +0.0428 -0.0413
w0: -0.3391 +0.3648 -0.3384 (1.88 sigma)
wa: -1.5943 +0.8722 -0.9275 (1.77 sigma)
Chi squared: 5.6480
Degs of freedom: 9
R^2: 0.9994
RMSD: 0.2010

Flat w0 + wa * np.tanh(0.5 * ((1 + z)**2 - 1))
r_d*h: 92.3266 +4.3865 -3.9885
Ωm: 0.3758 +0.0425 -0.0423
w0: -0.3647 +0.3431 -0.3267 (1.90 sigma)
wa: -1.2694 +0.7022 -0.7381 (1.76 sigma)
Chi squared: 5.6009
Degs of freedom: 9
R^2: 0.9994
RMSD: 0.2005

Flat w0 + wa * (1 - np.exp(0.5 - 0.5 * (1 + z)**2))
r_d*h: 92.3829 +4.2911 -3.9730
Ωm: 0.3771 +0.0416 -0.0420
w0: -0.3470 +0.3535 -0.3350 (1.90 sigma)
wa: -1.5626 +0.8617 -0.8841 (1.79 sigma)
Chi squared: 5.6023
Degs of freedom: 9
R^2: 0.9994
RMSD: 0.2013

===============================

Flat w0 + wa * np.tanh(0.5*(1 + z - 1/(1 + z)))
r_d*h: 92.0418 +4.7081 -4.0311
Ωm: 0.3807 +0.0422 -0.0460
w0: -0.2911 +0.3814 -0.3871 (1.84 sigma)
wa: -1.9994 +1.1534 -1.1248 (1.76 sigma)
Chi squared: 5.6068
Degs of freedom: 9
R^2: 0.9994
RMSD: 0.2017
"""
