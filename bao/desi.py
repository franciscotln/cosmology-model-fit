import numpy as np
import emcee
import corner
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data

c = 299792.458  # Speed of light in km/s
H0 = 70.0  # Hubble constant in km/s/Mpc

legend, data, cov_matrix = get_data()
cho = cho_factor(cov_matrix)


def H_z(z, Om, w0=-1):
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return H0 * np.sqrt(Om * one_plus_z**3 + (1 - Om) * evolving_de)


def DH_z(z, params):
    return c / H_z(z=z, Om=params[1], w0=params[2])


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": DV_z,
    "DM_over_rs": DM_z,
    "DH_over_rs": DH_z,
}


def model_predictions(params):
    return np.array(
        [(quantity_funcs[qty](z, params) / params[0]) for z, _, qty in data]
    )


bounds = np.array(
    [
        (120, 160),  # r_d
        (0.1, 0.7),  # Ωm
        (-2, 0),  # w0
    ]
)


def chi_squared(params):
    delta = data["value"] - model_predictions(params)
    return np.dot(delta, cho_solve(cho, delta))


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


def plot_bao_predictions(params):
    errors = np.sqrt(np.diag(cov_matrix))
    colors = {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"}

    residuals = data["value"] - model_predictions(params)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    r_d, Om = params[0], params[1]
    z_smooth = np.linspace(0, max(data["z"]), 100)
    plt.figure(figsize=(8, 6))
    for q in set(data["quantity"]):
        quantity_mask = data["quantity"] == q
        plt.errorbar(
            x=data["z"][quantity_mask],
            y=data["value"][quantity_mask],
            yerr=errors[quantity_mask],
            fmt=".",
            color=colors[q],
            label=q,
            capsize=2,
            linestyle="None",
        )
        model_smooth = [quantity_funcs[q](z, params) / r_d for z in z_smooth]
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"{legend}: $r_d$={r_d:.2f}, $\Omega_M$={Om:.3f}")
    plt.show()


def main():
    ndim = len(bounds)
    nwalkers = 10 * ndim
    burn_in = 500
    nsteps = 20000 + burn_in
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

    [
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, Om_50, w0_50]

    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {data['value'].size  - len(best_fit)}")

    plot_bao_predictions(best_fit)

    labels = ["$r_d$", "$Ω_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()


if __name__ == "__main__":
    main()


"""
******************************
Dataset: DESI 2025
******************************

Flat ΛCDM model
r_d: 145.04 +1.04 -1.06 Mpc
Ωm: 0.298 +0.009 -0.008
w0: -1
wa: 0
Chi squared: 10.27
Degs of freedom: 11
R^2: 0.9987
RMSD: 0.305

==============================

Flat wCDM model
r_d: 142.55 +2.53 -2.37 Mpc
Ωm: 0.297 +0.009 -0.009
w0: -0.915 +0.077 -0.079
wa: 0
Chi squared: 9.14
Degs of freedom: 10
R^2: 0.9989
RMSD: 0.279

===============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
r_d: 141.61 +3.01 -2.82 Mpc
Ωm: 0.304 +0.010 -0.010
w0: -0.873 +0.098 -0.105
wa: 0
Chi squared: 8.68
Degs of freedom: 10
R^2: 0.9990
RMSD: 0.271

==============================

Flat w0waCDM
r_d*h: 92.02 +4.60 -3.53 km/s
Ωm: 0.380 +0.036 -0.045
w0: -0.246 +0.344 -0.400
wa: -2.5086 +1.4138 -1.1921
Chi squared: 5.65
Degs of freedom: 9
R^2: 0.9994
RMSD: 0.205

******************************
Dataset: SDSS 2020
******************************

Flat ΛCDM model
r_d: 143.65 +1.83 -1.82 Mpc
Ωm: 0.298 +0.017 -0.016
w0: -1
wa: 0
Chi squared: 10.81
Degs of freedom: 12
R^2: 0.9947
RMSD: 0.739

==============================

Flat wCDM model
r_d: 136.51 +4.04 -3.74 Mpc
Ωm: 0.282 +0.023 -0.031
w0: -0.720 +0.147 -0.148
wa: 0
Chi squared: 7.48
Degs of freedom: 11
R^2: 0.9953
RMSD: 0.694

=============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
r_d: 135.97 +4.65 -4.21 Mpc
Ωm: 0.309 +0.019 -0.018
w0: -0.689 +0.163 -0.174 (1.79 - 1.91 sigma)
wa: 0
Chi squared: 7.55
Degs of freedom: 11
R^2: 0.9952
RMSD: 0.701
"""
