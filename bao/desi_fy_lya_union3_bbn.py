from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import block_diag
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
import y2024BBN.prior_lcdm_schoneberg as bbn
from cmb.data_early_lcdm_compression import r_drag
from y2026union3_1.data import get_data as get_sn_data
from y2025BAO.data_fs_lya import get_data as get_bao_data
from y2024DESBAO.data import get_data as get_des_bao_data


c = c0 / 1000  # km/s

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_desi_legend, desi_bao_data, desi_bao_cov_mat = get_bao_data()
des_bao_legend, des_bao_data, des_bao_cov_mat = get_des_bao_data()

# combine bao data and covariances
bao_data = np.concatenate((desi_bao_data, des_bao_data))
bao_cov_mat = block_diag(desi_bao_cov_mat, des_bao_cov_mat)

L_sn = np.linalg.cholesky(cov_matrix_sn)
L_bao = np.linalg.cholesky(bao_cov_mat)

log_2pi = np.log(2 * np.pi)

N_sn = z_cmb.size
N_bao = bao_data.size

logdet_sn = 2 * np.sum(np.log(np.diag(L_sn)))
logdet_bao = 2 * np.sum(np.log(np.diag(L_bao)))

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0, wa):
    # w0waCDM
    return (1. + z)**(3 * (1.0 + w0 + wa)) * np.exp(-3 * wa * z / (1. + z))


@njit
def H_z(z, params):
    H0, Om = params[0], params[1]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_DH_grid(theta):
    dh_grid = c / H_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return cum_dm, dh_grid


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
bao_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params, dm_dh_grid):
    h, Om, Obh2 = params[0] / 100, params[1], params[2]
    rd = r_drag(wb=Obh2, wm=Om * h**2)

    DM = DM_z(z, dm_dh_grid)
    DH = DH_z(z, dm_dh_grid)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    F_AP_mask = qty == 3
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH[DH_mask] / rd
    results[DM_mask] = DM[DM_mask] / rd
    results[DV_mask] = DV_z(z[DV_mask], DM[DV_mask], DH[DV_mask]) / rd
    results[F_AP_mask] = DM[F_AP_mask] / DH[F_AP_mask]
    return results


@njit
def get_zcosmo(params):
    # Heaviside step at z = 0.2
    delta_z = 1e-03 * params[3] * np.where(z_cmb <= 0.2, 1, -1)
    return z_cmb + delta_z


def mu_corr(params, dm_dh_grid):
    # For plotting only (delta z model)
    z_cosmo = get_zcosmo(params)
    return 5 * np.log10(DM_z(z_cosmo, dm_dh_grid) / DM_z(z_cmb, dm_dh_grid))


@njit
def theory_mu(params, DM):
    return params[4] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params):
    dm_dh_grid = DM_DH_grid(params)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], bao_qty, params, dm_dh_grid)
    chi_bao = solve_triangular(L_bao, delta_bao)
    chi2_bao = np.dot(chi_bao, chi_bao)

    z_cosmo = get_zcosmo(params)
    delta_sn = mu_values - theory_mu(params, DM_z(z_cosmo, dm_dh_grid))
    chi_sn = solve_triangular(L_sn, delta_sn)
    chi2_sn = np.dot(chi_sn, chi_sn)

    return chi2_bao + chi2_sn


@njit
def log_likelihood_jit(params):
    norm_bao = N_bao * log_2pi + logdet_bao
    norm_sn = N_sn * log_2pi + logdet_sn
    return -0.5 * (chi_squared(params)+ norm_bao + norm_sn)


def log_likelihood(params):
    return log_likelihood_jit(params)


def q0(Om, w0=-1.0):
    """Calculate the deceleration parameter at z=0."""
    return Om / 2 + (1.0 + 3 * w0) * (1.0 - Om) / 2


def j0(Om, w0=-1.0, wa=0.0):
    """Calculate the jerk parameter at z=0."""
    return 1.0 + (3 / 2) * (1.0 - Om) * (3 * w0 * (1.0 + w0) + wa)


def main():
    from scipy.stats import norm
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(55, 80))
    prior.add_parameter("om", dist=(0.10, 0.65))
    prior.add_parameter("obh2", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("dz_1000", dist=(-3.5, 3.5))
    prior.add_parameter("dM", dist=(-1.0, 1.0))

    with Pool(8) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    one_sigma_ci = [0.159, 0.5, 0.841]

    H0_16, H0_50, H0_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    Obh2_16, Obh2_50, Obh2_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    dz_1000_16, dz_1000_50, dz_1000_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    rd_samples = r_drag(samples[:, 2], Omh2_samples)
    q0_samples = q0(samples[:, 1])
    j0_samples = j0(samples[:, 1])

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)
    q0_16, q0_50, q0_84 = quantile(q0_samples, one_sigma_ci, weights=w)
    j0_16, j0_50, j0_84 = quantile(j0_samples, one_sigma_ci, weights=w)

    MAP_params = samples[np.argmax(log_l)]
    dof = N_bao + N_sn + 1 - len(MAP_params)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"1000 Δz: {dz_1000_50:.3f} +{(dz_1000_84 - dz_1000_50):.3f} -{(dz_1000_50 - dz_1000_16):.3f}")
    print(f"r_drag: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"q0: {q0_50:.3f} +{(q0_84 - q0_50):.3f} -{(q0_50 - q0_16):.3f}")
    print(f"j0: {j0_50:.3f} +{(j0_84 - j0_50):.3f} -{(j0_50 - j0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"Chi squared (MAP): {chi_squared(MAP_params):.2f}")
    print(f"Log Evidence: {sampler.log_z:.2f}")
    print(f"DOF: {dof}")

    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "1000 Δz", "ΔM"]
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
        range=np.repeat(0.9999, len(prior.keys)),
    )
    plt.show()

    dm_dh_grid = DM_DH_grid(MAP_params)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, MAP_params, dm_dh_grid),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_mat)),
        title="BAO DESI + DES",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(MAP_params, dm_dh_grid),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(MAP_params, DM_z(z_cmb, dm_dh_grid)),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ---------------------------------
# DESI DR2 + FS Lya (2026)
# Union3.1 (2026)
# BBN Schöngerg (2024)
# ---------------------------------
# Priors:
#
# All models:
# H0 ~U(55, 80)
# Om ~U(0.10, 0.65)
# ωb ~N(0.02218, 0.00055)
# dM ~U(-1.0, 1.0)
#
# wCDM:
# w ~U(-1.5, -0.5)
#
# w0waCDM (w0 + wa < 0 enforced):
# w0 ~U(-2.0, 0.0)
# wa ~U(-4.0, 2.0)
#
# Redshift step correction:
# 1000 Δz ~U(-3.5, 3.5)
# ---------------------------------


# ---------------------------------
# Flat ΛCDM
# H0: 68.6 +0.6 -0.6 km/s/Mpc
# Ωm: 0.3046 +0.0076 -0.0073
# ωb: 0.02218 +0.00055 -0.00055
# ωm: 0.14334 +0.00452 -0.00435
# r_d: 147.16 +1.43 -1.42 Mpc
# q0: -0.543 +0.011 -0.011
# j0: 1
# ΔM: -0.041 +0.021 -0.020 mag
# Chi squared (MAP): 43.39
# Log Evidence: 33.19
# DOF: 34
# ---------------------------------


# ---------------------------------
# Flat ΛCDM
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.2 positive z > 0.2 negative
# z_cosmo = z_cmb ± Δz
#
# 1000 Δz: 1.105 +0.396 -0.396
# H0: 68.6 +0.6 -0.6 km/s/Mpc
# Ωm: 0.3013 +0.0075 -0.0073
# ωb: 0.02219 +0.00054 -0.00054
# ωm: 0.14169 +0.00454 -0.00436
# r_d: 147.61 +1.43 -1.45 Mpc
# q0: -0.548 +0.011 -0.011
# j0: 1
# ΔM: -0.047 +0.021 -0.020 mag
# Chi squared (MAP): 35.59 (2.79 sigma significance)
# Log Evidence: 35.13 (Δ logZ = 1.94 against no flow)
# DOF: 33
# ---------------------------------


# ---------------------------------
# Flat wCDM w(z) = w0
# H0: 66.7 +1.4 -1.4 km/s/Mpc
# Ωm: 0.3038 +0.0076 -0.0074
# ωb: 0.02219 +0.00055 -0.00055
# ωm: 0.13518 +0.00701 -0.00692
# w: -0.928 +0.047 -0.047
# rd: 149.38 +2.12 -2.10 Mpc
# q0: -0.469 +0.049 -0.049
# j0: 0.791 +0.132 -0.119
# ΔM: -0.080 +0.033 -0.034 mag
# Chi squared (MAP): 40.93
# Log Evidence: 32.25
# DOF: 33
# ---------------------------------


# ---------------------------------
# Flat w(z) = w0 + wa * z / (1 + z)
# H0: 67.8 +1.3 -1.3 km/s/Mpc
# Ωm: 0.3296 +0.0127 -0.0138
# ωb: 0.02219 +0.00055 -0.00055
# ωm: 0.15171 +0.00805 -0.00918
# w0: -0.755 +0.106 -0.103
# wa: -0.998 +0.515 -0.508
# rd: 145.04 +2.50 -2.13 Mpc
# q0: -0.259 +0.116 -0.119
# j0: -0.558 +0.677 -0.601
# ΔM: -0.014 +0.035 -0.040 mag
# Chi squared (MAP): 37.19
# Log Evidence: 31.90
# DOF: 32
# ---------------------------------
