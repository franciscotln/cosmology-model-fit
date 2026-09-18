from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2025BAO.data_fs_lya import get_data as get_bao_data

c = c0 / 1000  # Speed of light in km/s

bao_legend, bao, bao_cov_matrix = get_bao_data()

L_cov = np.linalg.cholesky(bao_cov_matrix)

z_grid = np.linspace(0, np.max(bao["z"]) + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]

z_piv_w0_wa = 0.406


@njit
def H_z(z, params):
    H0, omh2, wp, wa = params[1], params[2], params[3], params[4]
    h = H0 / 100
    om = omh2 / h**2
    zp1 = 1. + z
    cubic = zp1**3
    rho_om = om * cubic
    rho_de = (1. - om) * cubic**(1. + wp + (wa / (1. + z_piv_w0_wa))) * np.exp(-3 * wa * z / zp1)
    return H0 * np.sqrt(rho_om + rho_de)


@njit
def DM_DH_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dy)
    return (cum_dm, dh_grid)


@njit
def DH_z(z, dm_dh_grid):
    return interp_pchip(z, z_grid, dm_dh_grid[1])


@njit
def DM_z(z, dm_dh_grid):
    return interp_hermite(z, z_grid, dm_dh_grid[0], dm_dh_grid[1])


@njit
def DV_z(z, DM, DH):
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    dm_dh_grid = DM_DH_grid(params)
    DM = DM_z(z, dm_dh_grid)
    DH = DH_z(z, dm_dh_grid)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    F_AP_mask = qty == 3

    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH[DH_mask] / params[0]
    results[DM_mask] = DM[DM_mask] / params[0]
    results[DV_mask] = DV_z(z[DV_mask], DM[DV_mask], DH[DV_mask]) / params[0]
    results[F_AP_mask] = DM[F_AP_mask] / DH[F_AP_mask]
    return results


@njit
def chi_squared(params):
    delta_bao = bao["value"] - bao_theory(bao["z"], qty, params)
    y = solve_triangular(L_cov, delta_bao)
    return y @ y


def log_likelihood(params):
    if params[3] + (params[4] / (1. + z_piv_w0_wa)) >= -1 / 3:
        return -np.inf
    return -0.5 * chi_squared(params)


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("rd", dist=(120, 160))
    prior.add_parameter("H0", dist=(50, 85))
    prior.add_parameter("ωm", dist=norm(loc=0.14331, scale=0.00092))  # SPA prior
    prior.add_parameter("wp", dist=(-1.5, -0.5))
    prior.add_parameter("wa", dist=(-10, +1.5))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=8_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    one_sigma_ci = [0.159, 0.5, 0.841]

    om_samples = samples[:, 2] / (samples[:, 1] / 100) ** 2

    rd_16, rd_50, rd_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    H0_16, H0_50, H0_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    omh2_16, omh2_50, omh2_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    wp_16, wp_50, wp_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    wa_16, wa_50, wa_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    om_16, om_50, om_84 = quantile(om_samples, one_sigma_ci, weights=w)

    def weighted_corrcoef(x, y, weights):
        cov = np.cov(x, y, aweights=weights)
        return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    print(f"Correlation between wp and wa: {weighted_corrcoef(samples[:, 3], samples[:, 4], w):.3f}")

    best_fit = samples[np.argmax(log_l)]
    DOF = len(bao["z"]) - len(best_fit)

    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {om_50:.3f} +{(om_84 - om_50):.3f} -{(om_50 - om_16):.3f}")
    print(f"ωm: {omh2_50:.4f} +{(omh2_84 - omh2_50):.4f} -{(omh2_50 - omh2_16):.4f}")
    print(f"wp: {wp_50:.3f} +{(wp_84 - wp_50):.3f} -{(wp_50 - wp_16):.3f}")
    print(f"wa: {wa_50:.3f} +{(wa_84 - wa_50):.3f} -{(wa_50 - wa_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {DOF}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
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


if __name__ == "__main__":
    main()


# *******************************
# DESI BAO DR2 2025
# Union3 SNe
# Omh2 gaussian prior from SPA 2026
# -------------------------------
#
# Priors:
# rd ~ U(120, 160)
# H0 ~ U(50.0, 85.0)
# ωm ~ N(0.14331, 0.00092)
#
# wCDM:
# w ~ U(-1.5, -0.5)
#
# w0waCDM:
# wp ~ U(-1.5, -0.5)
# wa ~ U(-10, +1.5)
# wp + wa / (1 + z_piv) > -1/3 enforced in the likelihood
# *******************************


# Flat ΛCDM: w(z) = -1
# rd: 146.73 +1.17 -1.15 Mpc
# H0: 68.98 +0.90 -0.91 km/s/Mpc
# Ωm: 0.301 +0.008 -0.008
# ωm: 0.1433 +0.0009 -0.0009
# Chi2 (MAP): 12.8
# Log evidence: -12.8
# Degs of freedom: 11
# -------------------------------


# Flat wCDM: w(z) = w
# rd: 145.94 +2.01 -2.19 Mpc
# H0: 68.90 +0.97 -0.95 km/s/Mpc
# Ωm: 0.302 +0.008 -0.008
# ωm: 0.1433 +0.0009 -0.0009
# w: -0.967 +0.072 -0.074
# Chi2 (MAP): 12.6
# Log evidence: -14.4
# Degs of freedom: 10
# -------------------------------


# Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
# at z_pivot = 0.406 correlation between wp and wa: -0.014
#
# rd: 150.8 +1.6 -1.8 Mpc
# H0: 60.0 +3.6 -3.1 km/s/Mpc
# Ωm: 0.398 +0.044 -0.043
# ωm: 0.1433 +0.0009 -0.0009
# wp: -0.99 +0.07 -0.07
# wa: -3.15 +1.45 -1.46
# Chi2 (MAP): 7.2
# Log evidence: -13.3
# DOF: 9
# -------------------------------
