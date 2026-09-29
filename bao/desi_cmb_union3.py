from numba import njit
import numpy as np
from scipy.linalg import block_diag
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2026union3_1.data import get_data
from y2025BAO.data_fs_lya import get_data as get_bao_data
from y2024DESBAO.data import get_data as get_des_bao_data
from y20116dFBAO.data import get_data as get_6dF_bao_data
import cmb.data_spt_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
desi_legend, desi_bao_data, desi_bao_cov_matrix = get_bao_data()
des_legend, des_bao_data, des_bao_cov_matrix = get_des_bao_data()
sixdF_legend, sixdF_bao_data, sixdF_bao_cov_matrix = get_6dF_bao_data()

bao = np.concatenate((desi_bao_data, des_bao_data, sixdF_bao_data))
bao_cov_mat = block_diag(desi_bao_cov_matrix, des_bao_cov_matrix, sixdF_bao_cov_matrix)

L_sn = np.linalg.cholesky(cov_matrix_sn)
L_bao = np.linalg.cholesky(bao_cov_mat)
L_cmb = np.linalg.cholesky(cmb.covariance)

logdet_sn = 2 * np.sum(np.log(np.diag(L_sn)))
logdet_bao = 2 * np.sum(np.log(np.diag(L_bao)))
logdet_cmb = 2 * np.sum(np.log(np.diag(L_cmb)))

N_sn = len(z_cmb)
N_bao = len(bao["z"])
N_cmb = len(cmb.DISTANCE_PRIORS)

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, 4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0, wa):
    # w1w2CDM
    zp1 = 1. + z
    return zp1**(3 * (1. + w0 + wa)) * ((zp1**2 + 1) / (2 * zp1**2))**(3 * wa)


@njit
def H_z(z, params):
    H0, Obh2, Och2 = params[1], params[2], params[3]
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return H0 * np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


cmb.set_HZ(H_z)


@njit
def DM_DH_grid(params):
    dh_grid = c/ H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    dm_grid = np.zeros(z_grid.size, dtype=np.float64)
    dm_grid[1:] = np.cumsum(dh * dz)
    return (dm_grid, dh_grid)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params, dm_dh_grid):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnuh2
    inv_rd = 1 / cmb.r_drag(Obh2, Omh2)

    DM = interp_hermite(z, z_grid, y=dm_dh_grid[0], y_prime=dm_dh_grid[1])
    DH = interp_pchip(z, z_grid, y=dm_dh_grid[1])

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    FAP_mask = qty == 3
    results[FAP_mask] = DM[FAP_mask] / DH[FAP_mask]
    results[DM_mask] = DM[DM_mask] * inv_rd
    results[DH_mask] = DH[DH_mask] * inv_rd
    results[DV_mask] = (z[DV_mask] * DH[DV_mask] * DM[DV_mask] ** 2) ** (1 / 3) * inv_rd
    return results


@njit
def get_z_cosmo(dz_1000):
    # Heaviside step at z = 0.2
    z_offset = 1e-03 * dz_1000 * np.where(z_cmb <= 0.2, 1, -1)
    return z_cmb + z_offset


def mu_corr(dz_1000, dm_dh_grid):
    # For plotting purposes only
    z_cosmo = get_z_cosmo(dz_1000)
    DM_obs = interp_hermite(z_cmb, z_grid, *dm_dh_grid)
    DM_cosmo = interp_hermite(z_cosmo, z_grid, *dm_dh_grid)
    return 5.0 * np.log10(DM_cosmo / DM_obs)


@njit
def mu_theory(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(params, dm_dh_grid):
    z_cosmo = get_z_cosmo(params[4])
    DM = interp_hermite(z_cosmo, z_grid, *dm_dh_grid)

    delta_sn = mu_vals - mu_theory(params[0], DM)
    y = solve_triangular(L_sn, delta_sn)
    return np.dot(y, y)


@njit
def chi2_bao(params, dm_dh_grid):
    delta_bao = bao["value"] - bao_theory(bao["z"], bao_qty, params, dm_dh_grid)
    y = solve_triangular(L_bao, delta_bao)
    return np.dot(y, y)


@njit
def chi2_cmb(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    y = solve_triangular(L_cmb, delta_cmb)
    return np.dot(y, y)


@njit
def chi_squared(params):
    dm_dh_grid = DM_DH_grid(params)
    return chi2_cmb(params) + chi2_bao(params, dm_dh_grid) + chi2_sn(params, dm_dh_grid)


@njit
def log_likelihood(params):
    norm_sn = N_sn * np.log(2 * np.pi) + logdet_sn
    norm_bao = N_bao * np.log(2 * np.pi) + logdet_bao
    norm_cmb = N_cmb * np.log(2 * np.pi) + logdet_cmb
    return -0.5 * (chi_squared(params) + norm_sn + norm_bao + norm_cmb)


def main():
    from multiprocessing import Pool
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("dM", dist=(-1, +1))  # mag
    prior.add_parameter("H0", dist=(60, 75))  # km/s/Mpc
    prior.add_parameter("obh2", dist=(0.01, 0.03))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("dz_1000", dist=(-3.5, 3.5))  # 1000 x Δz

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()

    labels=["ΔM", "H_0", "ω_b", "ω_c", "1000 Δz"]
    gd_samples = MCSamples(samples=samples, weights=np.exp(log_w), names=prior.keys, labels=labels)
    gd_samples.addDerived(gd_samples["obh2"] + gd_samples["och2"] + Omnuh2, name="omh2", label="ω_m")
    gd_samples.addDerived(gd_samples["omh2"] / (gd_samples["H0"] / 100) ** 2, name="om", label="Ω_m")
    gd_samples.addDerived(cmb.z_star(gd_samples["obh2"], gd_samples["omh2"]), name="zstar", label="z_*")
    gd_samples.addDerived(cmb.z_drag(gd_samples["obh2"], gd_samples["omh2"]), name="zdrag", label="z_d")
    gd_samples.addDerived(cmb.r_drag(gd_samples["obh2"], gd_samples["omh2"]), name="rdrag", label="r_d")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    DOF = N_sn + N_bao + N_cmb - len(best_fit)

    print(f"χ2 (MAP): {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {DOF}")

    plots.get_subplot_plotter().triangle_plot(
        gd_samples,
        params=["H0", "om", "omh2", "dz_1000"],
        title_limit=1,
        contour_colors=["C0"],
    )
    plt.show()

    dm_dh_grid_best = DM_DH_grid(best_fit)

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit, dm_dh_grid_best),
        data=bao,
        errors=np.sqrt(np.diag(bao_cov_mat)),
        title="DESI + DES + 6dF BAO",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit[4], dm_dh_grid_best),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit[0], interp_hermite(z_cmb, z_grid, *dm_dh_grid_best)),
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# *********************************
# Union 3.1 SNe 2026
# Compressed Planck + ACT
# DESI BAO DR2 + FS Lyα
# DES BAO 2025
# 6dF BAO 2011
# *********************************


# ----------- Flat ΛCDM -----------
# H0: 68.06 ± 0.24 km/s/Mpc
# Ωm: 0.3050 ± 0.0033
# ωb: 0.022470 ± 0.000091
# ωc: 0.11816 ± 0.00059
# ωm: 0.14128 ± 0.00059
# z*: 1088.55 ± 0.11
# z_d: 1059.95 ± 0.20
# r_d: 147.48 ± 0.18 Mpc
# ΔM: -0.0576 ± 0.0065 mag
# χ2 (MAP): 52.96
# Log evidence: 48.7
# DOF: 37
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Flat ΛCDM w(z) = -1
# Z offset step correction SNe observed redshifts
# (turning point z <= 0.2 positive z > 0.2 negative)
# z_cosmo = z_cmb ± Δz
#
# 1000 Δz: 1.08 ± 0.39 (prior ~U[-3.5, 3.5])
# H0: 68.10 ± 0.24 km/s/Mpc
# Ωm: 0.3044 ± 0.0033
# ωb: 0.022475 ± 0.000091
# ωc: 0.11805 ± 0.00059
# ωm: 0.14117 ± 0.00059
# z*: 1088.54 ± 0.11
# z_d: 1059.95 ± 0.21
# r_d: 147.51 ± 0.18 Mpc
# ΔM: -0.0592 ± 0.0065 mag
# χ2 (MAP): 45.42 (2.75 sigma significance)
# Log evidence: 50.5 (Δ logZ = 1.8 in favour of redshift step correction)
# DOF: 36
# ---------------------------------


# ----------- Flat wCDM -----------
# w: -1.012 ± 0.026 (prior ~U[-1.5, -0.5])
# H0: 68.33 ± 0.67 km/s/Mpc
# Ωm: 0.3030 ± 0.0057
# ωb: 0.022463 ± 0.000093
# ωc: 0.11835 ± 0.00073
# ωm: 0.14146 ± 0.00072
# z*: 1088.57 ± 0.12
# z_d: 1059.95 ± 0.21
# r_d: 147.44 ± 0.20 Mpc
# ΔM: -0.054 ± 0.010 mag
# χ2 (MAP): 52.82
# Log evidence: 46.1 (Δ logZ = -2.6 in favour of ΛCDM)
# DOF: 36
# ---------------------------------


# ----------- Flat w0waCDM --------
# Enforced wa + w0 < 0 in the likelihood
# (+0.2 to evidence from excluded volume)
#
# w0: -0.757 ± 0.081 (prior ~U[-1.5, 0.0])
# wa: -0.87 +0.29 -0.26 (prior ~U[-2.5, 1.5])
# H0: 66.88 ± 0.79 km/s/Mpc
# Ωm: 0.3194 ± 0.0078
# ωb: 0.022412 ± 0.000094
# ωc: 0.11974 ± 0.00081
# ωm: 0.14279 ± 0.00079
# z*: 1088.73 ± 0.13
# z_d: 1059.94 ± 0.20
# r_d: 147.13 ± 0.21 Mpc
# ΔM: -0.048 ± 0.011 mag
# χ2 (MAP): 41.09
# Log evidence: 49.9 + 0.2 (Δ logZ = 1.4 in favour of w0waCDM)
# DOF: 35
# ---------------------------------


# ----------- Flat w0waCDM --------
# Enforced wa + w0 < 0 in the likelihood
# Derived wa = -1.5 * (1 + w0)
#
# w0 = -0.967 ± 0.047
# H0 = 67.56 ± 0.76 km/s/Mpc
# Ωm = 0.3092 ± 0.0068
# ωb = 0.022479 ± 0.000092
# ωc = 0.11796 ± 0.00066
# ωm = 0.14108 ± 0.00065
# z* = 1088.53 ± 0.12
# z_d = 1059.95 ± 0.20
# r_d = 147.53 ± 0.19 Mpc
# ΔM = -0.0625 ± 0.0095 mag
# χ2 (MAP): 52.45
# Log evidence: 46.4 (Δ logZ = -2.3 in favour of ΛCDM)
# DOF: 36
# ---------------------------------


# ----------- Flat w1w2CDM --------
# Enforced w1 + w2 < 0 in the likelihood
# (+0.2 to evidence from excluded volume)
#
# w(z) = w1 + w2 * ((1 + z)^2 - 1) / ((1 + z)^2 + 1)
#
# H0 = 66.91 ± 0.78 km/s/Mpc
# ωb = 0.022412 ± 0.000093
# ωc = 0.11975 ± 0.00081
# w1 = -0.771 ± 0.077
# w2 = -0.71 +0.24 -0.21
# ωm = 0.14281 ± 0.00079
# Ωm = 0.3191 ± 0.0077
# z* = 1088.73 ± 0.13
# z_d = 1059.94 ± 0.20
# r_d = 147.12 ± 0.21 Mpc
# ΔM = -0.048 ± 0.011 mag
# χ2 (MAP): 41.12
# Log evidence: 49.7 + 0.2 (Δ logZ = 1.2 in favour of w1w2CDM)
# DOF: 35
# ---------------------------------