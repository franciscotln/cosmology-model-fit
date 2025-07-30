import numpy as np
import emcee
import corner
from scipy.integrate import quad
from scipy.constants import c as c0
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data
from .plot_predictions import plot_bao_predictions

c = c0 / 1000  # Speed of light in km/s
legend, data, cov_matrix = get_data()
cho = cho_factor(cov_matrix)
rd = 147.09  # Mpc, fixed


def H_z(z, H0, Om, w0):
    OL = 1 - Om
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))
    return H0 * np.sqrt(Om * one_plus_z**3 + OL * rho_de)


def DH_z(z, params):
    return c / H_z(z, *params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": lambda z, params: DV_z(z, params) / rd,
    "DM_over_rs": lambda z, params: DM_z(z, params) / rd,
    "DH_over_rs": lambda z, params: DH_z(z, params) / rd,
}


def theory_predictions(z, qty, params):
    return np.array([(quantity_funcs[qty](z, params)) for z, qty in zip(z, qty)])


bounds = np.array(
    [
        (50, 80),  # H0
        (0.1, 0.7),  # Ωm
        (-1.5, 0),  # w0
    ]
)


def chi_squared(params):
    delta = data["value"] - theory_predictions(data["z"], data["quantity"], params)
    return np.dot(delta, cho_solve(cho, delta))


def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return 0.0


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 15000 + burn_in
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
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [H0_50, Om_50, w0_50]
    residuals = data["value"] - theory_predictions(
        data["z"], data["quantity"], best_fit
    )
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{Om_84-Om_50:.4f} -{Om_50-Om_16:.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {data['value'].size  - len(best_fit)}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_predictions(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend}: $H_0$={H0_50:.2f} km/s/Mpc, $\\Omega_m$={Om_50:.4f}",
    )
    corner.corner(
        samples,
        labels=["$H_0$", "$Ω_m$", "$w_0$"],
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
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

Flat ΛCDM:
rd: 147.09 Mpc (fixed)
H0: 69.02 +0.50 -0.50 km/s/Mpc
Ωm: 0.2977 +0.0087 -0.0085
w0: -1
Chi squared: 10.27
Degs of freedom: 11
R^2: 0.9987
RMSD: 0.305

===============================

Flat wCDM:
rd: 147.09 Mpc (fixed)
H0: 67.85 +1.20 -1.14 km/s/Mpc
Ωm: 0.2970 +0.0091 -0.0088
w0: -0.915 +0.076 -0.079 (1.08 - 1.12 sigma from -1)
Chi squared: 9.11
Degs of freedom: 10
R^2: 0.9989
RMSD: 0.279

==============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
rd: 147.09 Mpc (fixed)
H0: 67.02 +1.61 -1.52 km/s/Mpc
Ωm: 0.3079 +0.0119 -0.0117
w0: -0.834 +0.122 -0.128 (1.28 - 1.38 sigma from -1)
Chi squared: 8.44
Degs of freedom: 10
R^2: 0.9990
RMSD: 0.266
"""
