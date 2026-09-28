from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2022pantheonSHOES.data import get_data as get_sn_data
from y2005cc.data import method, get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

cc_legend, z_cc, H_cc, diag_stat_cc, cov_mat_sys_cc = get_cc_data(split_sys=True)
sn_legend, z_cmb, z_hel, mb_vals, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

L_sn = cho_factor(cov_matrix_sn, lower=True)[0]
L_bao = cho_factor(cov_matrix_bao, lower=True)[0]

logdet_sn = 2 * np.sum(np.log(np.diag(L_sn)))
logdet_bao = 2 * np.sum(np.log(np.diag(L_bao)))

N_cc = len(z_cc)
N_bao = len(bao_data)
N_sn = len(z_cmb)

non_d = method != "D"

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0, wa):
    # w0waCDM
    return (1.0 + z) ** (3 * (1.0 + w0 + wa)) * np.exp(-3 * wa * z / (1.0 + z))


@njit
def H_z(z, params):
    H0, Om = params[3], params[5]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_DH_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return (cum_dm, dh_grid)


@njit
def DH_z(z, dm_dh_grid):
    return interp_pchip(z, x=z_grid, y=dm_dh_grid[1])


@njit
def DM_z(z, dm_dh_grid):
    return interp_hermite(z, x=z_grid, y=dm_dh_grid[0], y_prime=dm_dh_grid[1])


@njit
def DV_z(z, DM, DH):
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, rd, dm_dh_grid):
    inv_rd = 1 / rd
    dm_vals = DM_z(z, dm_dh_grid)
    dh_vals = DH_z(z, dm_dh_grid)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    FAP_mask = qty == 3

    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = dh_vals[DH_mask] * inv_rd
    results[DM_mask] = dm_vals[DM_mask] * inv_rd
    results[DV_mask] = DV_z(z[DV_mask], dm_vals[DV_mask], dh_vals[DV_mask]) * inv_rd
    results[FAP_mask] = dm_vals[FAP_mask] / dh_vals[FAP_mask]
    return results


@njit
def get_z_cosmo(params):
    # Heaviside step at z = 0.2
    z_offset = 1e-03 * params[6] * np.where(z_cmb <= 0.15, 1, -1)
    return z_cmb + z_offset


def mu_corr(params, dm_dh_grid):
    # For plotting purposes
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, dm_dh_grid) / DM_z(z_cmb, dm_dh_grid))


@njit
def mu_theory(DM):
    return 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(params, dm_dh_grid):
    M = params[2]
    DM_cosmo = DM_z(get_z_cosmo(params), dm_dh_grid)
    delta_sn = mb_vals - M - mu_theory(DM_cosmo)
    y = solve_triangular(L_sn, delta_sn)
    return np.dot(y, y)


@njit
def chi2_bao(params, dm_dh_grid):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], bao_qty, params[4], dm_dh_grid)
    y = solve_triangular(L_bao, delta_bao)
    return np.dot(y, y)


@njit
def chi2_cch(params, L_cc):
    delta_cc = H_cc - H_z(z_cc, params)
    y = solve_triangular(L_cc, delta_cc)
    return np.dot(y, y)


@njit
def chi_squared(params, L_cc):
    dm_dh_grid = DM_DH_grid(params)
    return chi2_sn(params, dm_dh_grid) + chi2_bao(params, dm_dh_grid) + chi2_cch(params, L_cc)


@njit
def get_fz(params):
    z_pivot = 0.62
    fp, n = np.exp(params[0]), params[1]
    fz = np.full_like(z_cc, fp)
    fz[non_d] *= ((1.0 + z_cc[non_d]) / (1.0 + z_pivot)) ** n
    return fz


@njit
def log_likelihood_jit(params):
    cov_mat_cc = cov_mat_sys_cc + np.diag(diag_stat_cc ** 2 * get_fz(params)**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2 * np.sum(np.log(np.diag(L_cc)))

    norm_cc = N_cc * np.log(2 * np.pi) + logdet_cc
    norm_sn = N_sn * np.log(2 * np.pi) + logdet_sn
    norm_bao = N_bao * np.log(2 * np.pi) + logdet_bao
    return -0.5 * (chi_squared(params, L_cc) + norm_cc + norm_sn + norm_bao)


def log_likelihood(params):
    return log_likelihood_jit(params)


def main():
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from ohd.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()

    # ------ CCH covariance rescaling parameters ------
    # ln(fp): CCH covariance diagonal rescaling scale
    # n: CCH covariance rescaling shape
    # overestimated uncertainties f(z) = fp * [(1 + z) / (1 + z_pivot)]^n
    # cov_total[i, i] = cov_sys[i, i] + diag_cov[i, i] * fz[i]^2
    # cov_total[i, j] = cov_sys[i, j]
    prior.add_parameter("ln_fp_cc", dist=(-1.5, 0.5))
    prior.add_parameter("n_cc", dist=(-2, 4))

    # M: supernovae magnitude zero-point offset
    prior.add_parameter("M", dist=(-20, -19))

    # ------ cosmological parameters ------------------
    # H0: Hubble constant at present
    prior.add_parameter("H0", dist=(45, 90))
    # rd: sound horizon at drag epoch
    prior.add_parameter("rd", dist=(100, 200))
    # Ωm: matter density parameter today
    prior.add_parameter("Om", dist=(0.2, 0.50))
    # dz_1000: (1000 x Δz) redshift offset step correction
    prior.add_parameter("dz_1000", dist=(-1.0, 1.0))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    labels=["ln(f_{pivot})", "n", "M", "H_0", "r_{drag}", "Ω_m", "1000 Δz"]
    gd_samples = MCSamples(samples=samples, weights=w, names=prior.keys, labels=labels)
    gd_samples.addDerived(gd_samples["Om"] * (gd_samples["H0"] / 100) ** 2, name="Omh2", label="Ω_m h^2")
    gd_samples.addDerived(np.exp(gd_samples["ln_fp_cc"]), name="fp_cc", label="f_{pivot}")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    DOF = N_sn + N_bao + N_cc - len(best_fit)
    fz_cc = get_fz(best_fit)
    cov_mat_cc = cov_mat_sys_cc + np.diag(diag_stat_cc ** 2 * fz_cc**2)
    L_cc = cho_factor(cov_mat_cc, lower=True)[0]

    print(f"Chi2 (MAP): {chi_squared(best_fit, L_cc):.2f}")
    print(f"log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"DOF: {DOF}")

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=["H0", "Om", "rd", "dz_1000", "ln_fp_cc", "n_cc"],
        title_limit=1,
        color=["C0"],
        contour_colors=["C0"],
        filled=True,
    )
    plt.show()

    dm_dh_grid = DM_DH_grid(best_fit)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit[4], dm_dh_grid),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mb_vals - best_fit[2] - mu_corr(best_fit, dm_dh_grid),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(DM_z(z_cmb, dm_dh_grid)),
        label=f"$Ω_m$={best_fit[5]:.3f}",
        x_scale="log",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc,
        H=H_cc,
        H_err=np.sqrt(np.diag(cov_mat_sys_cc) + diag_stat_cc**2),
        label=f"{cc_legend} $H_0$: {best_fit[3]:.1f} km/s/Mpc",
        method=method,
        err_scaling=1 / fz_cc,
    )


if __name__ == "__main__":
    main()


# *******************************************
# Data sets:
# BAO DESI DR2 + FS Lya
# SN1a Union3.1
# Cosmic Chronometers
# *******************************************


# ----------------- Priors ------------------
# ln(fp):   U[-1.5, 0.5]
# n_cc:     U[-2, 4]
# M:        U[-20, -19]
# H0:       U[45, 90]
# rd:       U[100, 200]
# Ωm:       U[0.2, 0.5]
#
# wCDM:
# w:       U[-1.5, -0.5]
#
# w0waCDM:
# w0:       U[-3, 1]
# wa:       U[-3, 3]
# Enforced w0 + wa < -1/3
#
# Redshift offset step correction for SNe:
# dz_1000:  U[-1, 1] (1000 x Δz)
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# H0 = 70.4 +1.5 -1.3 km/s/Mpc
# Ωm = 0.3066 ± 0.0071
# Ωm h^2 = 0.1519 ± 0.0066
# rd = 143.2 +2.5 -3.1 Mpc
#
# M = -19.348 +0.046 -0.039 mag
# n = 1.48 ± 0.51
# ln(fp) = -0.57 +0.15 -0.17
# fp = 0.575 +0.069 -0.110
#
# Chi2 (MAP): 1457.61
# log likelihood (MAP): 695.91
# Log evidence: 678.00
# DOF: 1637
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# Redshift offset step correction for SNe observed redshifts
# turning point z <= 0.15 positive z > 0.15 negative
# z_cosmo = z_cmb ± Δz

# 1000 Δz = 0.27 ± 0.13 (2 sigma)
# H0 = 70.5 +1.5 -1.3 km/s/Mpc
# Ωm = 0.3036 ± 0.0071
# Ωm h^2 = 0.1510 ± 0.0065
# rd = 143.3 +2.6 -3.1 Mpc
#
# M = -19.353 +0.046 -0.040 mag
# n = 1.49 ± 0.51
# ln(fp) = -0.57 +0.15 -0.17
# fp = 0.575 +0.069 -0.110
#
# Chi2 (MAP): 1451.97
# log likelihood (MAP): 698.27
# Log evidence: 678.54
# DOF: 1636
# -------------------------------------------


# --------------- Flat wCDM -----------------
# w = -0.938 ± 0.037
# H0 = 69.8 +1.5 -1.3 km/s/Mpc
# Ωm = 0.3040 ± 0.0073
# Ωm h^2 = 0.1482 ± 0.0068
# rd = 143.0 +2.5 -3.1 Mpc
#
# M = -19.357 +0.047 -0.039 mag
# n = 1.50 ± 0.51
# ln(fp) = -0.57 +0.15 -0.17
# fp = 0.572 +0.068 -0.110
#
# Chi2 (MAP): 1455.25
# log likelihood (MAP): 697.29
# Log evidence: 677.02
# DOF: 1636
# -------------------------------------------


# -------------- Flat w0waCDM ---------------
# w0 = -0.887 +0.059 -0.069
# wa = -0.38 ± 0.42
# H0 = 69.6 +1.5 -1.4 km/s/Mpc
# Ωm = 0.312 +0.015 -0.0093
# Ωm h^2 = 0.1512 +0.0089 -0.0071
# rd = 143.2 +2.5 -3.1 Mpc
#
# M = -19.358 +0.046 -0.039 mag
# n = 1.51 ± 0.51
# ln(fp) = -0.57 +0.15 -0.18
# fp = 0.572 +0.068 -0.11
#
# Chi2 (MAP): 1452.84
# log likelihood (MAP): 697.76
# Log evidence: 674.36
# DOF: 1635
# -------------------------------------------
