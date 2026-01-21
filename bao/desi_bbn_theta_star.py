from numba import njit
import numpy as np
from y2025BAO.data import get_data as get_bao_data
import cmb.data_planck_compression as cmb
import y2024BBN.prior_lcdm_schoneberg as bbn

c = cmb.c  # speed of light in km/s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
inv_cov_sn = np.linalg.inv(bao_cov_matrix)

z_max = np.max(bao_data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0, wa):
    zp1 = 1 + z
    return (2 * zp1**3 / (1 + w0 + (1 - w0) * zp1**3)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0, wa)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0 = params
    return H0 * Ez(z, H0, Obh2, Och2, w0)


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
    Obh2, Och2 = params[1], params[2]
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnu_h2)
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def chi_squared(params):
    delta_thetastar = (
        cmb.DISTANCE_PRIORS[1] - cmb.cmb_distances(H_z, params[1], params[2], params)[1]
    )
    chi2_thetastar = delta_thetastar**2 / cmb.covariance[1, 1]

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao @ inv_cov_sn @ delta_bao

    return chi2_thetastar + chi_bao


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(50.0, 90.0))
    prior.add_parameter("ωb", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("ωc", dist=(0.05, 0.30))
    prior.add_parameter("w0", dist=(-1.0, 0.0))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    one_sigma_ci = [0.159, 0.5, 0.841]

    H0_16, H0_50, H0_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    Obh2_16, Obh2_50, Obh2_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    Och2_16, Och2_50, Och2_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)

    best_fit = [H0_50, Obh2_50, Och2_50, w0_50]

    Omh2_samples = samples[:, 1] + samples[:, 2] + Omnu_h2
    Om_samples = Omh2_samples / (samples[:, 0] / 100) ** 2
    rd_samples = cmb.r_drag(wb=samples[:, 1], wm=Omh2_samples)
    zstar_samples = cmb.z_star(wb=samples[:, 1], wm=Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)
    zst_16, zst_50, zst_84 = quantile(zstar_samples, one_sigma_ci, weights=w)

    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r*: {cmb.rs_z(H_z, zst_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"z*: {zst_50:.2f} +{(zst_84 - zst_50):.2f} -{(zst_50 - zst_16):.2f}")
    print(
        f"100 θ*: {100 * np.pi / (cmb.cmb_distances(H_z, Obh2_50, Och2_50, best_fit)[1]):.5f}"
    )
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")

    labels = ["$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    corner(
        samples,
        weights=w,
        labels=labels,
        quantiles=one_sigma_ci,
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
        range=np.repeat(0.9999, len(labels)),
    )
    plt.show()

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
Dataset: DESI DR2 2024 + θ∗ Planck + BBN
*******************************

Flat ΛCDM w(z) = -1
rd: 148.30 +0.71 -0.70 Mpc
H0: 68.51 +0.48 -0.47 km/s/Mpc
ωb: 0.02217 +0.00054 -0.00054
ωc: 0.1163 +0.0009 -0.0009
ωm: 0.1391 +0.0011 -0.0011
Ωm: 0.2964 +0.0045 -0.0045
w0: -1
wa: 0
r*: 145.55 Mpc
z*: 1089.84 +0.68 -0.65
100 θ*: 1.04109
Chi squared: 10.29
Log evidence: -15.0

===============================

Flat wCDM w(z) = w0
rd: 148.48 +0.76 -0.75 Mpc
H0: 67.81 +1.13 -1.09 km/s/Mpc
ωb: 0.02223 +0.00054 -0.00055
ωc: 0.1154 +0.0016 -0.0016
ωm: 0.1383 +0.0017 -0.0017
Ωm: 0.3007 +0.0076 -0.0077
w0: -0.967 +0.048 -0.049
wa: 0
r*: 145.74 Mpc
z*: 1089.69 +0.72 -0.69
100 θ*: 1.04107
Chi squared: 9.71
Log evidence: -17.3

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
rd: 148.46 +0.71 -0.71 Mpc
H0: 66.56 +1.23 -1.41 km/s/Mpc
ωb: 0.02227 +0.00054 -0.00054
ωc: 0.1153 +0.0010 -0.0011
ωm: 0.1382 +0.0013 -0.0013
Ωm: 0.3122 +0.0124 -0.0104
w0: -0.841 +0.109 -0.098
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
r*: 145.73 Mpc
z*: 1089.63 +0.68 -0.66
100 θ*: 1.04099
Chi squared: 9.04
Log evidence: -15.8

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
w0 - prior width 4.0: -2.5 to 1.5
wa - prior width 12.0: -8.0 to 4.0
"""
