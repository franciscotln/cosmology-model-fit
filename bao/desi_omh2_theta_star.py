from numba import njit
import numpy as np
import emcee
import corner
from scipy.constants import c as c0
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
import cmb.data_union3_compression as cmb
from .plot_predictions import plot_bao_predictions

c = c0 / 1000  # speed of light in km/s
Or_h2 = cmb.Omega_r_h2()

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)

# arXiv:2503.14738v2 Planck Plik [1]
z_star = 1089.92

theta_100 = 1.04110
theta_100_err = 0.00031

Omh2 = 0.1430
Omh2_sigma = 0.0011


@njit
def Ez(z, params):
    H0, Om, w0 = params[1], params[2], params[4]
    h = H0 / 100
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or

    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


@njit
def H_z(z, params):
    return params[1] * Ez(z, params)


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


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_predictions(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[0]


def theta_100_theory(params):
    H0, wb = params[1], params[3]
    rs_star = cmb.rs_z(Ez, z_star, params, H0, wb)
    DA_star = cmb.DA_z(Ez, z_star, params, H0)
    return 100 * rs_star / ((1 + z_star) * DA_star)


def chi_squared(params):
    h, Om = params[1] / 100, params[2]
    chi2_Omh2 = ((Omh2 - Om * h**2) / Omh2_sigma) ** 2

    delta_theta_100 = theta_100 - theta_100_theory(params)
    chi2_theta_100 = (delta_theta_100 / theta_100_err) ** 2

    delta_bao = bao_data["value"] - bao_predictions(bao_data["z"], quantities, params)
    chi2_bao = delta_bao.dot(cho_solve(cho_bao, delta_bao, check_finite=False))

    return chi2_Omh2 + chi2_theta_100 + chi2_bao


bounds = np.array(
    [
        (120, 160),  # rd
        (55, 75),  # H0
        (0.20, 0.50),  # Ωm
        (0.015, 0.040),  # Ωb * h^2
        (-1.4, 0.0),  # w0
    ],
    dtype=np.float64,
)


@njit
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
    nwalkers = 180
    burn_in = 200
    nsteps = 2200 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.30),
                (emcee.moves.DEMove(), 0.56),
                (emcee.moves.DESnookerMove(), 0.14),
            ],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    rd_16, rd_50, rd_84 = pct[0]
    H0_16, H0_50, H0_84 = pct[1]
    Om_16, Om_50, Om_84 = pct[2]
    Obh2_16, Obh2_50, Obh2_84 = pct[3]
    w0_16, w0_50, w0_84 = pct[4]

    best_fit = np.array([rd_50, H0_50, Om_50, Obh2_50, w0_50], dtype=np.float64)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"r*: {cmb.rs_z(Ez, z_star, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_predictions(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    labels = ["$r_d$", "$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
    )
    plt.show()

    plt.figure(figsize=(16, 1.5 * ndim))
    for n in range(ndim):
        plt.subplot2grid((ndim, 1), (n, 0))
        plt.plot(chains_samples[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: DESI DR2 2024 + θ∗ + Ωm x h2
*******************************

Flat ΛCDM w(z) = -1
H0: 69.35 +1.03 -1.02 km/s/Mpc
Ωm: 0.297 +0.009 -0.008
Ωb h^2: 0.02355 +0.00080 -0.00079
w0: -1
rd: 146.42 +1.32 -1.31 Mpc
r*: 143.56 Mpc
Chi squared: 10.28
degs of freedom: 11

===============================

Flat wCDM w(z) = w0
H0: 69.44 +1.07 -1.05 km/s/Mpc
Ωm: 0.297 +0.009 -0.009
Ωb h^2: 0.02558 +0.00243 -0.00201
w0: -0.913 +0.075 -0.079
rd: 143.90 +2.63 -2.97 Mpc
r*: 142.56 Mpc
Chi squared: 10.14
degs of freedom: 10

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 68.17 +1.35 -1.28 km/s/Mpc
Ωm: 0.308 +0.012 -0.012
Ωb h^2: 0.02490 +0.00137 -0.00131
w0: -0.829 +0.119 -0.128
rd: 144.59 +1.90 -1.94 Mpc
r*: 142.89 Mpc
Chi squared: 8.63
degs of freedom: 10
"""
