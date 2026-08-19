from numba import njit
import numpy as np
from scipy.linalg import block_diag
import cmb.data_planck_act_compression as cmb
from interpolator import interp_hermite
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2026union3_1.data import get_data as get_sn_data
from y2025BAO.data_fs_lya import get_data as get_bao_data
from y2024DESBAO.data import get_data as get_bao_des_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_desi_legend, bao_desi_data, cov_mat_bao_desi = get_bao_data()
bao_des_legend, bao_des_data, cov_mat_bao_des = get_bao_des_data()

bao = np.concatenate((bao_desi_data, bao_des_data))
cov_mat_bao = block_diag(cov_mat_bao_desi, cov_mat_bao_des)

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(cov_mat_bao)

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2


@njit
def Ez(z, h, Obh2, Och2):
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return np.sqrt(radiation_term + matter_term + neutrino_term + dark_energy_term)


@njit
def H_z(z, theta):
    H0 = theta[1]
    return H0 * Ez(z, h=H0 / 100, Obh2=theta[2], Och2=theta[3])


cmb.set_HZ(H_z)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, theta):
    Obh2, Och2 = theta[2], theta[3]
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnuh2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    FAP_mask = qty == 3
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta) / rd
    results[DM_mask] = DM_z(z[DM_mask], theta) / rd
    results[DV_mask] = DV_z(z[DV_mask], theta) / rd
    results[FAP_mask] = (DM_z(z[FAP_mask], theta) / DH_z(z[FAP_mask], theta))
    return results


@njit
def mu_corr(params, DM_ref):
    v_km_s = 100 * params[4] * np.where(z_cmb <= 0.2, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_ref)


@njit
def theory_mu(theta, DM):
    return theta[0] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(theta):
    DM = DM_z(z_cmb, theta)
    delta_sn = mu_values - theory_mu(theta, DM) - mu_corr(theta, DM)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_bao(theta):
    delta_bao = bao["value"] - bao_theory(bao["z"], bao_qty, theta)
    return delta_bao @ inv_cov_bao @ delta_bao


@njit
def chi2_thetastar(theta):
    delta_lA = cmb.DISTANCE_PRIORS[1] - cmb.cmb_distances(theta[2], theta[3], theta)[1]
    return delta_lA**2 / cmb.covariance[1, 1]


@njit
def chi_squared(theta):
    return chi2_sn(theta) + chi2_bao(theta) + chi2_thetastar(theta)


def log_likelihood(theta):
    return -0.5 * chi_squared(theta)


def q0(Om, w0=-1):
    """Calculate the deceleration parameter at z=0."""
    return Om / 2 + (1 + 3 * w0) * (1 - Om) / 2


def j0(Om, w0=-1, wa=0):
    """Calculate the jerk parameter at z=0."""
    return 1 + (3 / 2) * (1 - Om) * (3 * w0 * (1 + w0) + wa)


def main():
    from scipy.stats import norm
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("ΔM", dist=(-1.0, 1.0))
    prior.add_parameter("H0", dist=(50.0, 90.0))
    prior.add_parameter("ωb", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("ωc", dist=(0.05, 0.30))
    prior.add_parameter("v100", dist=(-8.5, 8.5))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=8_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    one_sigma_ci = [0.159, 0.5, 0.841]

    dM_16, dM_50, dM_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    H0_16, H0_50, H0_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    Obh2_16, Obh2_50, Obh2_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    Och2_16, Och2_50, Och2_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    v100_16, v100_50, v100_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    rd_samples = cmb.r_drag(samples[:, 2], Omh2_samples)
    q0_samples = q0(Om_samples)
    j0_samples = j0(Om_samples)

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)
    q0_16, q0_50, q0_84 = quantile(q0_samples, one_sigma_ci, weights=w)
    j0_16, j0_50, j0_84 = quantile(j0_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, H0_50, Obh2_50, Och2_50, v100_50]
    DOF = 1 + len(bao) + len(z_cmb) - len(best_fit)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"v: {v100_50:.3f} +{(v100_84 - v100_50):.3f} -{(v100_50 - v100_16):.3f} x 100 km/s")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"q0: {q0_50:.3f} +{(q0_84 - q0_50):.3f} -{(q0_50 - q0_16):.3f}")
    print(f"j0: {j0_50:.3f} +{(j0_84 - j0_50):.3f} -{(j0_50 - j0_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(samples[np.argmax(log_l)]):.2f}")
    print(f"Log Evidence: {sampler.log_z:.2f}")
    print(f"DOF: {DOF}")

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

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao,
        errors=np.sqrt(np.diag(cov_mat_bao)),
        title=f"{bao_desi_legend} + {bao_des_legend}",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(best_fit, DM_z(z_cmb, best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit, DM_z(z_cmb, best_fit)),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# *********************************
# BAO: DESI DR2 + FS Lya + DES BAO 2025
# Prior on ωb from BBN (Y2024)
# SN1a: Union3.1 compilation (Y2026)
# *********************************
# 
# Priors:
#
# all models:
# ΔM: U[-1.0, 1.0]
# H0: U[50, 90]
# ωb: N(loc=0.02218, scale=0.00055)
# ωc: U[0.05, 0.30]
#
# flow correction:
# v_100 ~U[-8.5, 8.5] x 100 km/s
#
# wCDM:
# w0: U[-1.5, -0.5]
#
# wzCDM
# w0: U[-1.0, -1/3]
#
# w0waCDM:
# w0: U[-1.5, 0.0]
# wa: U[-3.0, 1.0]
# w0 + wa < 0 enforced
# ---------------------------------


# ----------- Flat ΛCDM -----------
# ΔM: -0.057 +0.012 -0.012 mag
# H0: 68.25 +0.46 -0.46 km/s/Mpc
# ωb: 0.02207 +0.00053 -0.00053
# ωc: 0.1165 +0.0008 -0.0008
# ωm: 0.1392 +0.0011 -0.0011
# Ωm: 0.299 +0.004 -0.004
# r_d: 148.37 +0.70 -0.70 Mpc
# q0: -0.552 +0.007 -0.006
# j0: 1
# Chi2 (MAP): 43.91
# Log Evidence: -37.73
# DOF: 34
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.2 inflow z > 0.2 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)
#
# v: -3.1 +1.0 -1.0 x 100 km/s
# ΔM: -0.056 +0.012 -0.012 mag
# H0: 68.36 +0.45 -0.46 km/s/Mpc
# ωb: 0.02212 +0.00053 -0.00053
# ωc: 0.1163 +0.0008 -0.0008
# ωm: 0.1391 +0.0011 -0.0011
# Ωm: 0.298 +0.004 -0.004
# r_d: 148.36 +0.70 -0.70 Mpc
# q0: -0.553 +0.007 -0.006
# j0: 1
# Chi2 (MAP): 35.57 (2.89 sigma significance)
# Log Evidence: -35.19 (Δ logZ = 2.54 in favour of ΛCDM with v step correction)
# DOF: 33
# ---------------------------------


# ----------- Flat wCDM -----------
# ΔM: -0.069 +0.014 -0.014 mag
# H0: 67.13 +0.79 -0.78 km/s/Mpc
# ωb: 0.02223 +0.00054 -0.00054
# ωc: 0.1148 +0.0013 -0.0014
# ωm: 0.1377 +0.0014 -0.0015
# Ωm: 0.305 +0.006 -0.006
# w0: -0.942 +0.034 -0.034
# r_d: 148.66 +0.72 -0.70 Mpc
# q0: -0.481 +0.041 -0.042
# j0: 0.828 +0.098 -0.088
# Chi2 (MAP): 41.42 (1.58 sigma away from ΛCDM)
# Log Evidence: -38.75 (Δ logZ = -1.02 in favour of ΛCDM)
# DOF: 33
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
# 
# ΔM: -0.069 +0.013 -0.013 mag
# H0: 66.64 +0.83 -0.83 km/s/Mpc
# ωb: 0.02221 +0.00053 -0.00054
# ωc: 0.1155 +0.0009 -0.0009
# ωm: 0.1383 +0.0012 -0.0011
# Ωm: 0.312 +0.007 -0.007
# w0: -0.858 +0.063 -0.062
# wa: -0.396 [derived wa = -1.5 * (1 - w0^2)]
# r_d: 148.49 +0.70 -0.70 Mpc
# q0: -0.385 +0.071 -0.072
# j0: 0.213 +0.316 -0.278
# Chi2 (MAP): 39.72 (2.05 sigma away from ΛCDM)
# Log Evidence: -36.82 (Δ logZ = 0.91 in favour of wzCDM)
# DOF: 33
# ---------------------------------


# ---------- Flat w0waCDM ---------
# ΔM: -0.061 +0.015 -0.015 mag
# H0: 66.61 +0.85 -0.84 km/s/Mpc
# ωb: 0.02208 +0.00054 -0.00054
# ωc: 0.1179 +0.0018 -0.0021
# ωm: 0.1406 +0.0019 -0.0021
# Ωm: 0.317 +0.009 -0.009
# w0: -0.803 +0.094 -0.090
# wa: -0.597 +0.359 -0.382
# r_d: 148.03 +0.78 -0.77 Mpc
# q0: -0.323 +0.105 -0.103
# j0: -0.098 +0.545 -0.516
# Chi2 (MAP): 38.25 (1.89 sigma away from ΛCDM)
# Log Evidence: -39.27 + 0.09 = -39.18 (Δ logZ = -1.45 in favour of ΛCDM)
# DOF: 32
# ---------------------------------
