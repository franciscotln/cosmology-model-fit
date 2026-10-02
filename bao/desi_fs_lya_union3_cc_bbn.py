from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from rdrag import r_drag
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2026union3_1.data import get_data
from y2005cc.data import method, get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

cc_legend, z_cc, H_cc, diag_stat_cc, cov_mat_sys_cc = get_cc_data(split_sys=True)
sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

L_sn = np.linalg.cholesky(cov_matrix_sn)
L_bao = np.linalg.cholesky(cov_matrix_bao)

N_cc = len(z_cc)
N_bao = len(bao_data)
N_sn = len(z_cmb)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0, wa):
    # w1w2CDM
    zp1 = 1.0 + z
    return zp1**(3 * (1.0 + w0 + wa)) * (2 * zp1**2 / (zp1**2 + 1.0))**(-3 * wa)


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
def bao_theory(z, qty, params, dm_dh_grid):
    Ombh2 = params[4]
    Omh2 = params[5] * (params[3] / 100)**2
    inv_rd = 1 / r_drag(Ombh2, Omh2)
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
    z_offset = 1e-03 * params[6] * np.where(z_cmb <= 0.2, 1, -1)
    return z_cmb + z_offset


def mu_corr(params, dm_dh_grid):
    # For plotting purposes
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, dm_dh_grid) / DM_z(z_cmb, dm_dh_grid))


@njit
def mu_theory(params, DM):
    offset = params[2]
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(params, dm_dh_grid):
    DM_cosmo = DM_z(get_z_cosmo(params), dm_dh_grid)
    delta_sn = mu_vals - mu_theory(params, DM_cosmo)
    y = solve_triangular(L_sn, delta_sn)
    return np.dot(y, y)


@njit
def chi2_bao(params, dm_dh_grid):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], bao_qty, params, dm_dh_grid)
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


method_f = method == "F"
z_pivot = 1.198
shape = np.ones_like(z_cc, dtype=np.float64)
shape[method_f] = ((1 + z_cc[method_f]) / (1 + z_pivot))**4
# statistical error in H is proportional to (1+z) * H^2


@njit
def get_fz(params):
    fp, n = np.exp(params[0]), params[1]
    return fp * shape**n


@njit
def log_likelihood_jit(params):
    cov_mat_cc = cov_mat_sys_cc + np.diag(diag_stat_cc ** 2 * get_fz(params)**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2 * np.sum(np.log(np.diag(L_cc)))

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc
    return -0.5 * (chi_squared(params, L_cc) + normalization_cc)


def log_likelihood(params):
    return log_likelihood_jit(params)


def main():
    from scipy.stats import norm
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
    # overestimated uncertainties f(z) = fp * [(1 + z) / (1 + z_pivot)]^4n
    # cov_total[i, i] = cov_sys[i, i] + diag_cov[i, i] * fz[i]^2
    # cov_total[i, j] = cov_sys[i, j]
    prior.add_parameter("ln_fp_cc", dist=(-2, 1))
    prior.add_parameter("n_cc", dist=(-2.5, 2.5))

    # ΔM: supernovae magnitude zero-point offset
    prior.add_parameter("dM", dist=(-1, 1))

    # ------ cosmological parameters ------------------
    # H0: Hubble constant at present
    prior.add_parameter("H0", dist=(45, 90))
    # ombh2: baryon density parameter at present
    prior.add_parameter("ombh2", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    # Ωm: matter density parameter today
    prior.add_parameter("om", dist=(0.2, 0.5))
    # dz_1000: (1000 x Δz) redshift offset step correction
    prior.add_parameter("dz_1000", dist=(-3.5, 3.5))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    labels=["ln(f_{pivot})", "n", "ΔM", "H_0", "Ω_b h^2", "Ω_m", "1000 Δz"]
    gd_samples = MCSamples(samples=samples, weights=w, names=prior.keys, labels=labels)
    gd_samples.addDerived(gd_samples["om"] * (gd_samples["H0"] / 100) ** 2, name="omh2", label="Ω_m h^2")
    gd_samples.addDerived(r_drag(gd_samples["ombh2"], gd_samples["omh2"]), name="rd", label="r_{drag}")
    gd_samples.addDerived(gd_samples["rd"] * gd_samples["H0"] / 100, name="hrd", label="h * r_{drag}")
    gd_samples.addDerived(np.exp(gd_samples["ln_fp_cc"]), name="fp_cc", label="f_{pivot}")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    DOF = N_sn + N_bao + N_cc - len(best_fit)
    fz_cc = get_fz(best_fit)
    cov_mat_cc = cov_mat_sys_cc + np.diag(diag_stat_cc ** 2 * fz_cc**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)

    print(f"Chi2 (MAP): {chi_squared(best_fit, L_cc):.2f}")
    print(f"log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"DOF: {DOF}")

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=["H0", "om", "ombh2", "dz_1000", "ln_fp_cc", "n_cc"],
        title_limit=1,
        color=["C0"],
        contour_colors=["C0"],
        filled=True,
    )
    plt.show()

    dm_dh_grid = DM_DH_grid(best_fit)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit, dm_dh_grid),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit, dm_dh_grid),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit, DM_z(z_cmb, dm_dh_grid)),
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
# BBN Schöngerg (2024)
# *******************************************


# ----------------- Priors ------------------
# ln(fp):   U[-2, 1]
# n_cc:     U[-2.5, 2.5]
# ΔM:       U[-1, 1]
# H0:       U[45, 90]
# rd:       U[100, 200]
# Ωm:       U[0.2, 0.5]
# ωb:       N(0.02218, 0.00055^2)
#
# wCDM:
# w:       U[-1.5, -0.5]
#
# w1w2CDM:
# w1:       U[-2, 0]
# w2:       U[-3, 2]
# Enforced w1 + w2 < 0
#
# Redshift offset step correction for SNe:
# dz_1000:  U[-3.5, 3.5] (1000 x Δz)
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# H0 = 68.76 ± 0.53 km/s/Mpc (2.4 sigma tension with CMB SPA)
# Ωb h^2 = 0.02230 ± 0.00052 (0.19 sigma tension)
# Ωm = 0.3066 ± 0.0072 (1.2 sigma tension)
# Ωm h^2 = 0.1449 ± 0.0041 (0.4 sigma tension)
# r_d = 146.7 ± 1.3 Mpc (0.22 sigma tension)
# h * r_d = 100.84 ± 0.63 Mpc (2.2 sigma tension)
#
# ΔM = -0.034 ± 0.018 mag
# n = 1.05 +0.31 -0.55
# ln(fp) = -0.43 ± 0.25
# fp = 0.67 +0.13 -0.19
#
# Chi2 (MAP): 82.27
# log likelihood (MAP): -172.20
# Log evidence: -187.65
# DOF: 69
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# Redshift offset step correction for SNe observed redshifts
# turning point z <= 0.2 positive z > 0.2 negative
# z_cosmo = z_cmb ± Δz

# H0 = 68.75 ± 0.53 km/s/Mpc
# Ωb h^2 = 0.02232 ± 0.00052
# Ω_m = 0.3035 ± 0.0071
# Ωm h^2 = 0.1435 ± 0.0041
# r_d = 147.0 ± 1.3 Mpc
# 1000 Δz = 1.09 ± 0.40
#
# ΔM = -0.039 ± 0.018 mag
# n = 1.05 +0.32 -0.56
# ln(fp) = -0.43 ± 0.25
# fp = 0.67 +0.13 -0.19
#
# Chi2 (MAP): 72.47
# log likelihood (MAP): -168.38
# Log evidence: -185.84
# DOF: 68
# -------------------------------------------


# --------------- Flat wCDM -----------------
# H0 = 67.7 ± 1.1 km/s/Mpc
# Ωb h^2 = 0.02237 ± 0.00053
# Ωm = 0.3070 ± 0.0072
# Ωm h^2 = 0.1408 ± 0.0057
# r_d = 147.7 ± 1.6 Mpc
# w = -0.957 ± 0.041
#
# ΔM = -0.054 ± 0.026 mag
# n = 1.04 +0.32 -0.56
# ln(fp) = -0.43 ± 0.25
# fp = 0.67 +0.13 -0.19
#
# Chi2 (MAP): 78.32
# log likelihood (MAP): -171.17
# Log evidence: -189.36
# DOF: 68
# -------------------------------------------


# -------------- Flat w1w2CDM ---------------
# w(z) = w1 + w2 * ((1 + z)^2 - 1) / ((1 + z)^2 + 1)

# H0 = 67.9 ± 1.0 km/s/Mpc
# Ωb h^2 = 0.02219 ± 0.00052
# Ωm = 0.330 ± 0.012
# Ωm h^2 = 0.1520 ± 0.0065
# r_d = 145.0 +1.6 -1.8 Mpc
# w1 = -0.774 ± 0.093
# w2 = -0.81 +0.38 -0.34
#
# ΔM = -0.015 +0.028 -0.026 mag
# n = 1.06 +0.32 -0.54
# ln(fp) = -0.40 ± 0.25
# fp = 0.69 +0.13 -0.19
#
# Chi2 (MAP): 75.37
# log likelihood (MAP): -169.34
# Log evidence: -189.11
# DOF: 67
# -------------------------------------------
