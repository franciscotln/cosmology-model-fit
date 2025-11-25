from numba import njit
import numpy as np
from scipy.constants import c as c0
from y2025BAO.data import get_data as get_bao_data

c = c0 / 1000  # Speed of light in km/s

bao_legend, bao_data, bao_cov_matrix = get_bao_data()

inv_cov_mat = np.linalg.inv(bao_cov_matrix)

z_grid = np.linspace(0, np.max(bao_data["z"]) + 0.1, num=2000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    h, Omh2, w0 = params[1] / 100, params[2], params[3]
    Om = Omh2 / h**2
    zp1 = 1 + z
    cubic = zp1**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Om * cubic + (1 - Om) * rho_de)


@njit
def H_z(z, params):
    return params[1] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[0]


@njit
def chi_squared(params):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    return delta_bao @ inv_cov_mat @ delta_bao


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("rd", dist=(120, 160))
    prior.add_parameter("H0", dist=(50.0, 85.0))
    prior.add_parameter("ωm", dist=norm(loc=0.1430, scale=0.0011))  # Planck prior
    prior.add_parameter("w0", dist=(-1.6, 0.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=8_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    one_sigma_ci = [0.159, 0.5, 0.841]

    Om_samples = samples[:, 2] / (samples[:, 1] / 100) ** 2

    rd_16, rd_50, rd_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    H0_16, H0_50, H0_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    Omh2_16, Omh2_50, Omh2_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)

    best_fit = [rd_50, H0_50, Omh2_50, w0_50]

    corner(
        samples,
        weights=w,
        labels=prior.keys,
        quantiles=one_sigma_ci,
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
        range=np.repeat(0.9999, len(prior.keys)),
    )
    plt.show()

    degs_of_freedom = len(bao_data["z"]) - len(best_fit)

    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"Degs of freedom: {degs_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )


if __name__ == "__main__":
    main()

"""
*******************************
DESI BAO DR2 2025
Union3 SNe
Omh2 prior from Planck 2018

Priors:
rd U(120, 160)
H0 U(50.0, 85.0)
ωm N(0.1430, 0.0011)
w0 U(-1.6, 0.0)
wa U(-5.0, +3.0)

*******************************

Flat ΛCDM: w(z) = -1
rd: 146.44 +1.34 -1.31 Mpc
H0: 69.34 +1.04 -1.03 km/s/Mpc
Ωm: 0.297 +0.009 -0.008
ωm: 0.1430 +0.0011 -0.0011
w0: -1
wa: 0
Chi squared: 10.3
Log evidence: -11.4
Degs of freedom: 10

===============================

Flat wCDM: w(z) = w0
rd: 144.02 +2.67 -2.96 Mpc
H0: 69.42 +1.08 -1.07 km/s/Mpc
Ωm: 0.297 +0.009 -0.009
ωm: 0.1430 +0.0011 -0.0011
w0: -0.915 +0.076 -0.081
wa: 0
Chi squared: 9.2
Log evidence: -12.9
Degs of freedom: 9

===============================

Flat wzCDM: w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)
rd: 144.71 +1.91 -1.93 Mpc
H0: 67.93 +1.48 -1.41 km/s/Mpc
Ωm: 0.310 +0.013 -0.013
ωm: 0.1430 +0.0011 -0.0011
w0: -0.797 +0.146 -0.156
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
Chi squared: 8.3
Log evidence: -12.0
Degs of freedom: 9

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
TODO
"""
