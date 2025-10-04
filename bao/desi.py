from numba import njit
import numpy as np
import emcee
import corner
from scipy.constants import c as c0
import matplotlib.pyplot as plt
from y2025BAO.data import get_data
from .plot_predictions import plot_bao_predictions, plot_bao_residuals

c = c0 / 1000  # Speed of light in km/s
rd = 147.09  # Mpc, fixed
legend, data, cov_matrix = get_data()
cho = np.linalg.cholesky(cov_matrix)
cho_T = cho.T


@njit
def H_z(z, params):
    h, Om, w0 = params
    OL = 1 - Om
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return 100 * h * np.sqrt(Om * cubed + OL * rho_de)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    result = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        zp = z[i]
        x = np.linspace(0, zp, num=max(250, int(250 * zp)))
        y = DH_z(x, params)
        result[i] = np.trapz(y=y, x=x)
    return result


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


@njit
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


bounds = np.array(
    [
        (0.500, 0.800),  # h
        (0.1, 0.7),  # Ωm
        (-2.0, 0.0),  # w0
    ],
    dtype=np.float64,
)


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

quantities = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def chi_squared(params):
    delta = data["value"] - bao_theory(data["z"], quantities, params)
    y = np.linalg.solve(cho, delta)
    x = np.linalg.solve(cho_T, y)
    return np.dot(delta, x)


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return 0.0


@njit
def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    n_dim = len(bounds)
    n_walkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    initial_pos = np.zeros((n_walkers, n_dim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, n_walkers)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        log_probability,
        moves=[
            (emcee.moves.KDEMove(), 0.30),
            (emcee.moves.DEMove(), 0.56),
            (emcee.moves.DESnookerMove(), 0.14),
        ],
    )
    sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("Auto-correlation time:", tau)
        print(
            "Effective samples:", n_dim * n_walkers * (nsteps - burn_in) / np.max(tau)
        )
        print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [h_16, h_50, h_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([h_50, Om_50, w0_50], dtype=np.float64)

    residuals = data["value"] - bao_theory(data["z"], quantities, best_fit)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"h: {h_50:.3f} +{(h_84 - h_50):.3f} -{(h_50 - h_16):.3f}")
    print(f"Ωm: {Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {data['value'].size  - len(best_fit)}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend}: $H_0$={100 * h_50:.1f} km/s/Mpc, $Ω_m$={Om_50:.3f}",
    )
    plot_bao_residuals(data, residuals, np.sqrt(np.diag(cov_matrix)))

    labels = ["$h$", "$Ω_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()

    plt.figure(figsize=(16, 1.5 * n_dim))
    for n in range(n_dim):
        plt.subplot2grid((n_dim, 1), (n, 0))
        plt.plot(sampler.get_chain(discard=burn_in)[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


"""
*******************************
Dataset: DESI DR2 2024
*******************************

Flat ΛCDM:
rd: 147.09 Mpc (fixed)
h: 0.690 +0.005 -0.005
Ωm: 0.298 +0.009 -0.008
w0: -1
Chi squared: 10.27
Degs of freedom: 11
R^2: 0.9987
RMSD: 0.305

================================

Flat wCDM:
rd: 147.09 Mpc (fixed)
h: 0.678 +0.012 -0.011
Ωm: 0.297 +0.009 -0.009
w0: -0.915 +0.075 -0.079
Chi squared: 9.12
Degs of freedom: 10
R^2: 0.9989
RMSD: 0.279

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
rd: 147.09 Mpc (fixed)
h: 0.670 +0.016 -0.015
Ωm: 0.308 +0.012 -0.011
w0: -0.832 +0.118 -0.125
Chi squared: 8.44
Degs of freedom: 10
R^2: 0.9990
RMSD: 0.265

*******************************
Dataset: SDSS 2020 compilation
*******************************

Flat ΛCDM:
h: 0.688 +0.007 -0.007
Ωm: 0.294 +0.016 -0.015
w0: -1
Chi squared: 11.81
Degs of freedom: 15
R^2: 0.9955
RMSD: 0.684

===============================

Flat wCDM:
h: 0.663 +0.017 -0.016
Ωm: 0.283 +0.019 -0.022
w0: -0.793 +0.128 -0.132
Chi squared: 9.81
Degs of freedom: 14
R^2: 0.9956
RMSD: 0.673

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
h: 0.659 +0.021 -0.019
Ωm: 0.305 +0.018 -0.017
w0: -0.737 +0.166 -0.178
Chi squared: 10.04
Degs of freedom: 14
R^2: 0.9956
RMSD: 0.678
"""
