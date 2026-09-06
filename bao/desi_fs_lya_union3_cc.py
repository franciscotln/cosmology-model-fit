from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2026union3_1.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, H_cc_err, cov_mat_sys_cc = get_cc_data(split_sys=True)
sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

L_sn = cho_factor(cov_matrix_sn, lower=True)[0]
L_bao = cho_factor(cov_matrix_bao, lower=True)[0]

N_cc = len(z_cc_vals)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    inv_a3 = (1.0 + z) ** 3
    return (2 * inv_a3 / (1.0 + w0 + (1.0 - w0) * inv_a3)) ** 2


@njit
def H_z(z, params):
    H0, Om = params[3], params[5]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return (cum_dm, dh_grid)


@njit
def DH_z(z, dm_grid):
    return interp_pchip(z, z_grid, dm_grid[1])


@njit
def DM_z(z, dm_grid):
    return interp_hermite(z, z_grid, *dm_grid)


@njit
def DV_z(z, DM, DH):
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, rd, dm_grid):
    inv_rd = 1 / rd
    dm_vals = DM_z(z, dm_grid)
    dh_vals = DH_z(z, dm_grid)

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
    v_km_s = 100 * params[6] * np.where(z_cmb <= 0.2, 1, -1)
    return -1.0 + (1.0 + z_cmb) / (1.0 + v_km_s / c)


def mu_corr(params, dm_grid, z_obs):
    # For plotting purposes
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, dm_grid) / DM_z(z_obs, dm_grid))


@njit
def mu_theory(params, DM):
    offset = params[2]
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(params, dm_grid):
    DM_cosmo = DM_z(get_z_cosmo(params), dm_grid)
    delta_sn = mu_vals - mu_theory(params, DM_cosmo)
    y = solve_triangular(L_sn, delta_sn)
    return np.dot(y, y)


@njit
def chi2_bao(params, dm_grid):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], bao_qty, params[4], dm_grid)
    y = solve_triangular(L_bao, delta_bao)
    return np.dot(y, y)


@njit
def chi2_cch(params, L_cc):
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    y = solve_triangular(L_cc, delta_cc)
    return np.dot(y, y)


@njit
def chi_squared(params, L_cc):
    dm_grid = DM_grid(params)
    return chi2_sn(params, dm_grid) + chi2_bao(params, dm_grid) + chi2_cch(params, L_cc)


z_pivot = 0.615


@njit
def log_likelihood_jit(params):
    fp_cc, n_cc = np.exp(params[0]), params[1]
    fz_cc = fp_cc * ((1.0 + z_cc_vals) / (1.0 + z_pivot)) ** n_cc
    cov_mat_cc = np.diag(H_cc_err ** 2 * fz_cc**2) + cov_mat_sys_cc
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2.0 * np.sum(np.log(np.diag(L_cc)))

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc
    return -0.5 * (chi_squared(params, L_cc) + normalization_cc)


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
    # ln(fp): CCH covariance diagonal rescaling
    # (overestimated uncertainties f(z) = fp * [(1+z) / (1+z_pivot)]^n)
    # n: CCH covariance rescaling cov_total[i, j] = cov_sys[i, j] + diag_cov[i, j] * fz[i] * fz[j]
    prior.add_parameter("ln_fp_cc", dist=(np.log(0.3), np.log(1.2)))
    prior.add_parameter("n_cc", dist=(-4.0, 4.0))

    # ------ cosmological parameters ------------------
    # ΔM: magnitude zero-point offset
    prior.add_parameter("dM", dist=(-1, 1))
    # H0: Hubble constant at present
    prior.add_parameter("H0", dist=(45, 90))
    # rd: sound horizon at drag epoch
    prior.add_parameter("rd", dist=(100, 200))
    # Ωm: matter density parameter today
    prior.add_parameter("Om", dist=(0.2, 0.50))
    # v: velocity step correction observed redshift SNe
    # prior.add_parameter("v", dist=(-8.5, 8.5))
    prior.add_parameter("v", dist=(-1.0, -1/3))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    labels=["ln(f_{pivot})", "n", "ΔM", "H_0", "r_{drag}", "Ω_m", "v_{100}"]
    gd_samples = MCSamples(
        samples=samples,
        weights=w,
        names=prior.keys,
        labels=labels,
        loglikes=log_l,
    )
    gd_samples.addDerived(gd_samples["Om"] * (gd_samples["H0"] / 100) ** 2, name="Omh2", label="Ω_m h^2")
    gd_samples.addDerived(np.exp(gd_samples["ln_fp_cc"]), name="fp_cc", label="f_{pivot}")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    DOF = len(z_cmb) + len(bao_data) + N_cc - len(best_fit)
    fz_cc = np.exp(best_fit[0]) * ((1.0 + z_cc_vals) / (1.0 + z_pivot)) ** best_fit[1]
    cov_mat_cc = np.diag(H_cc_err ** 2 * fz_cc**2) + cov_mat_sys_cc
    L_cc = np.linalg.cholesky(cov_mat_cc)

    print(f"Chi squared (MAP): {chi_squared(best_fit, L_cc):.2f}")
    print(f"log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"DOF: {DOF}")

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=["H0", "Om", "rd", "v", "ln_fp_cc", "n_cc"],
        title_limit=1,
        color=["C0"],
        contour_colors=["C0"],
        filled=True,
    )
    plt.show()

    dm_grid = DM_grid(best_fit)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit[4], dm_grid),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit, dm_grid, z_cmb),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit, DM_z(z_cmb, dm_grid)),
        label=f"$Ω_m$={best_fit[5]:.3f}",
        x_scale="log",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=H_cc_err,
        label=f"{cc_legend} $H_0$: {best_fit[3]:.1f} km/s/Mpc",
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
# ln(fp):   U[ln(0.3), ln(1.2)]
# n_cc:     U[-4, 4]
# ΔM:       U[-1, 1]
# H0:       U[45, 90]
# rd:       U[100, 200]
# Ωm:       U[0.2, 0.5]

# wzCDM:
# w0:       U[-1, -1/3]

# wCDM:
# w0:       U[-1.5, -0.5]

# w0waCDM:
# w0:       U[-1.5, 0]
# wa:       U[-5, 3]
# Enforced w0 + wa < 0, forbidden prior region removed in evidence calculation.

# Velocity step correction in observed redshift SNe:
# v:        U[-8.5, 8.5] x 100 km/s
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# H0: 67.9 +- 2.8 km/s/Mpc
# rd: 149.0 +5.7 -6.6 Mpc
# Ωm: 0.3049 +- 0.0073
# Ωm h^2: 0.141 +- 0.012
# ΔM: -0.067 +- 0.091 mag
# ln(fp): -0.49 +0.11 -0.12
# fp = 0.615 +0.057 -0.083
# n_cc: 1.35 +- 0.46
# Chi squared (MAP): 82.65
# log likelihood (MAP): -171.75
# Log evidence: -189.76
# DOF: 69
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# Velocity step correction SNe redshift
# turning point z <= 0.2 inflow z > 0.2 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# H0: 68.1 +- 2.9 km/s/Mpc
# rd: 148.9 +5.6 -6.6 Mpc
# Ωm: 0.3015 +- 0.0073
# Ωm h^2: 0.140 +- 0.012
# v: -3.0 +- 1.1 x 100 km/s
# ΔM: -0.064 +-0.091 mag
# ln(fp): -0.49 +0.11 -0.12
# fp = 0.616 +0.057 -0.083
# n_cc: 1.35 +- 0.46
# Chi squared (MAP): 73.58
# log likelihood (MAP): -167.66
# Log evidence: -187.50 (Δ logZ = 2.26 in favour of velocity step correction)
# DOF: 68
# -------------------------------------------


# --------------- Flat wCDM -----------------
# H0: 66.9 +- 2.9 km/s/Mpc
# rd: 149.3 +5.7 -6.6 Mpc
# Ωm: 0.3041 +- 0.0074
# Ωm h^2: 0.136 +- 0.012
# w0: -0.936 +- 0.046
# ΔM: -0.078 +- 0.091 mag
# ln(fp): -0.49 +0.11 -0.12
# fp: 0.619 +0.058 -0.084
# n_cc: 1.33 +- 0.47
# Chi squared (MAP): 78.03
# log likelihood (MAP): -170.76
# Log evidence: -190.93 (Δ logZ = -1.17 in favour of ΛCDM)
# DOF: 68
# -------------------------------------------


# --------------- Flat wzCDM ----------------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
# H0: 66.3 +- 2.9 km/s/Mpc
# rd: 149.6 +5.7 -6.7 Mpc
# Ωm: 0.3112 +- 0.0080
# Ωm h^2: 0.137 +- 0.012
# w0: -0.860 +- 0.065
# wa: d w(z)/d z at z=0 = -1.5 * (1 - w0^2) = -0.39
# ΔM: -0.083 +- 0.092 mag
# ln(fp): -0.48 +0.11 -0.12
# fp: 0.620 +0.057 -0.084
# n_cc: 1.34 +- 0.47
# Chi squared (MAP): 78.00
# log likelihood (MAP): -170.01
# Log evidence: -189.35 (Δ logZ = 0.41 in favour of wzCDM)
# DOF: 68
# -------------------------------------------


# -------------- Flat w0waCDM ---------------
# TODO: re-run (LCDM is preferred though)
# -------------------------------------------
