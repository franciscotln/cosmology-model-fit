from numba import njit, prange
import numpy as np
from scipy.linalg import block_diag
from interpolator import interp_hermite, interp_pchip
from y2026union3_1.data import get_data
from y2025BAO.data_fs_lya import get_data as get_bao_data
from y2024DESBAO.data import get_data as get_des_bao_data
from y20116dFBAO.data import get_data as get_6dF_bao_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
desi_legend, desi_bao_data, desi_bao_cov_matrix = get_bao_data()
des_legend, des_bao_data, des_bao_cov_matrix = get_des_bao_data()
sixdF_legend, sixdF_bao_data, sixdF_bao_cov_matrix = get_6dF_bao_data()

bao = np.concatenate((desi_bao_data, des_bao_data, sixdF_bao_data))
bao_cov_mat = block_diag(desi_bao_cov_matrix, des_bao_cov_matrix, sixdF_bao_cov_matrix)

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(bao_cov_mat)

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, 4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


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

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0 = params[1]
    return H0 * Ez(z, H0 / 100, Obh2=params[2], Och2=params[3])


cmb.set_HZ(H_z)


@njit
def DM_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return (cum_dm, dh_grid)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params, DM_interp):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnuh2
    rdrag = cmb.r_drag(Obh2, Omh2)

    DM = interp_hermite(z, z_grid, *DM_interp)
    DH = interp_pchip(z, z_grid, DM_interp[1])

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    FAP_mask = qty == 3
    results[FAP_mask] = DM[FAP_mask] / DH[FAP_mask]
    results[DM_mask] = DM[DM_mask] / rdrag
    results[DH_mask] = DH[DH_mask] / rdrag
    results[DV_mask] = (z[DV_mask] * DH[DV_mask] * DM[DV_mask] ** 2) ** (1 / 3) / rdrag
    return results


@njit
def chi2_bao(params, DM_interp):
    delta_bao = bao["value"] - bao_theory(bao["z"], bao_qty, params, DM_interp)
    return delta_bao @ inv_cov_bao @ delta_bao


@njit
def mu_corr(v100, DM_interp):
    # Heaviside step at z = 0.2
    v_km_s = 100 * v100 * np.where(z_cmb <= 0.2, 1, -1)
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + v_km_s / c)

    DM_obs = interp_hermite(z_cmb, z_grid, *DM_interp)
    DM_cosmo = interp_hermite(z_cosmo, z_grid, *DM_interp)
    return 5.0 * np.log10(DM_cosmo / DM_obs)


@njit
def mu_theory(offset, DM_interp):
    DM = interp_hermite(z_cmb, z_grid, *DM_interp)
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(params, DM_interp):
    delta_sn = mu_vals - mu_theory(params[0], DM_interp) - mu_corr(params[4], DM_interp)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_cmb(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    return delta_cmb @ cmb.inv_cov_mat @ delta_cmb


@njit
def chi_squared(params):
    DM_interp = DM_grid(params)
    return chi2_cmb(params) + chi2_bao(params, DM_interp) + chi2_sn(params, DM_interp)


@njit
def log_likelihood(params):
    return -0.5 * chi_squared(params)


@njit(parallel=True)
def log_likelihood_vec(batch):
    n = batch.shape[0]
    log_likelihoods = np.empty(n, dtype=np.float32)
    for i in prange(n):
        log_likelihoods[i] = log_likelihood(batch[i])
    return log_likelihoods


def main():
    import os
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    os.environ["OMP_NUM_THREADS"] = "1"

    prior = Prior()
    prior.add_parameter("dM", dist=(-1, +1))  # mag
    prior.add_parameter("H0", dist=(60, 75))  # km/s/Mpc
    prior.add_parameter("obh2", dist=(0.01, 0.03))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("v", dist=(-10, 4))  # x 100 km/s

    sampler = Sampler(
        prior,
        log_likelihood_vec,
        n_live=6_000,
        pool=(None, 4),
        seed=42,
        pass_dict=False,
        vectorized=True,
    )
    sampler.run(verbose=True)
    samples, log_w, log_l = sampler.posterior()

    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=["ΔM", "H_0", "ω_b", "ω_c", "v_{100}"],
    )
    gd_samples.addDerived(
        gd_samples["obh2"] + gd_samples["och2"] + Omnuh2, name="omh2", label="ω_m"
    )
    gd_samples.addDerived(
        gd_samples["omh2"] / (gd_samples["H0"] / 100) ** 2, name="om", label="Ω_m"
    )
    gd_samples.addDerived(
        cmb.z_star(gd_samples["obh2"], gd_samples["omh2"]), name="zstar", label="z_*"
    )
    gd_samples.addDerived(
        cmb.z_drag(gd_samples["obh2"], gd_samples["omh2"]),
        name="zdrag",
        label="z_{drag}",
    )
    gd_samples.addDerived(
        cmb.r_drag(gd_samples["obh2"], gd_samples["omh2"]),
        name="rdrag",
        label="r_{drag}",
    )

    best_fit = gd_samples.mean(prior.keys)
    DOF = len(z_cmb) + len(bao) + len(cmb.DISTANCE_PRIORS) - len(best_fit)

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    index_MAP = np.argmax(log_l)
    print(f"χ2 (MAP): {chi_squared(samples[index_MAP]):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {DOF}")

    plots.get_subplot_plotter().triangle_plot(
        gd_samples, params=["H0", "om", "omh2", "v"], title_limit=1, contour_colors=["C0"]
    )
    plt.show()

    DM_grid_best = DM_grid(best_fit)

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit, DM_grid_best),
        data=bao,
        errors=np.sqrt(np.diag(bao_cov_mat)),
        title="DESI + DES + 6dF BAO",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit[4], DM_grid_best),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit[0], DM_grid_best),
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
# ΔM: -0.0516 ± 0.0069 mag
# H0: 68.36 ± 0.27 km/s/Mpc
# Ωm: 0.3012 ± 0.0035
# ωb: 0.02257 ± 0.00010
# ωc: 0.11753 ± 0.00064
# ωm: 0.14074 ± 0.00063
# z*: 1089.43 ± 0.15
# z_d: 1060.19 ± 0.23
# r_d: 147.54 ± 0.19 Mpc
# χ2 (MAP): 48.20
# Log evidence: -43.1
# DOF: 37
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Flat ΛCDM w(z) = -1
# Velocity step correction SNe observed redshifts
# (turning point z <= 0.2 inflow z > 0.2 outflow)
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# ΔM: -0.0519 ± 0.0069 mag
# v: -3.0 ± 1.0 (prior U(-10, 4)) x 100 km/s
# v / (z_cut=0.2): -1500 ± 500 km/s
# H0: 68.42 ± 0.26 km/s/Mpc
# Ωm: 0.30036 ± 0.00350
# ωb: 0.02257 ± 0.00010
# ωc: 0.11739 ± 0.00063
# ωm: 0.1404 ± 0.0006
# z*: 1089.41 ± 0.15
# z_d: 1060.20 ± 0.23
# r_d: 147.57 ± 0.19 Mpc
# χ2 (MAP): 39.70 (2.92 sigma significance)
# Log evidence: -40.5 (Δ logZ = 2.6 in favour of step correction)
# DOF: 36
# ---------------------------------


# ----------- Flat wCDM -----------
# ΔM: -0.0550 ± 0.0103 mag
# H0: 68.09 ± 0.67 km/s/Mpc
# Ωm: 0.3031 ± 0.0056
# ωb: 0.02258 ± 0.00011
# ωc: 0.1173 ± 0.0009
# ωm: 0.14049 ± 0.00082
# w0: -0.988 ± 0.027 (prior U(-1.5, -0.5))
# z*: 1089.39 ± 0.17
# z_d: 1060.20 ± 0.23
# r_d: 147.59 ± 0.22 Mpc
# χ2 (MAP): 47.97 (0.48 sigma away from ΛCDM)
# Log evidence: -45.7 (Δ logZ = -2.6 in favour of ΛCDM)
# DOF: 36
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
# ΔM: -0.0620 ± 0.0089 mag
# H0: 67.14 +0.78 -0.63 km/s/Mpc
# Ωm: 0.3112 +0.0059 -0.0072
# ωb: 0.02259 ± 0.00010
# ωc: 0.1170 ± 0.0007
# ωm: 0.14023 ± 0.00069
# w0: -0.900 +0.046 -0.062 (prior U(-1, -1/3)
# z*: 1089.35 ± 0.16
# z_d: 1060.21 ± 0.23
# r_d: 147.65 ± 0.20 Mpc
# χ2 (MAP): 45.88 (1.52 sigma away from ΛCDM)
# Log evidence: -43.5 (Δ logZ = -0.4 in favour of ΛCDM)
# DOF: 36
# ---------------------------------


# ----------- Flat w0waCDM --------
# Enforced wa + w0 < 0 in the likelihood (corrected evidence calculation)
# ΔM: -0.0492 ± 0.0107 mag
# H0: 66.94 ± 0.79 km/s/Mpc
# Ωm: 0.3171 ± 0.0078
# ωb: 0.02251 ± 0.00011
# ωc: 0.1189 ± 0.0010
# ωm: 0.14204 ± 0.00095
# w0: -0.781 ± 0.081 (prior U(-1.5, 0.0))
# wa: -0.73 +0.29 -0.26 (prior U(-2.5, 1.0))
# z*: 1089.63 ± 0.19
# z_d: 1060.17 ± 0.23
# r_d: 147.24 ± 0.25 Mpc
# χ2 (MAP): 40.43 (2.32 sigma away from ΛCDM)
# Log evidence: -43.8 + 0.1 (Δ logZ = -0.6 in favour of ΛCDM)
# DOF: 35
# ---------------------------------
