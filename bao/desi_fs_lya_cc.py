from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from cmb.data_early_lcdm_compression import r_drag
from y2005cc.data import method, get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, diag_stat_cc, cc_sys_cov_matrix = get_cc_data(split_sys=True)
bao_legend, data, bao_cov_matrix = get_bao_data()

cho_bao = np.linalg.cholesky(bao_cov_matrix)
N_bao = len(data["z"])
N_cc = len(z_cc_vals)

c = c0 / 1000  # Speed of light in km/s

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]

z_piv_w0wa = 0.38 # corr(wp, wa) = -0.0034


@njit
def Ode_z(z, wp, wa):
    zp1 = 1. + z
    # w0waCDM
    return zp1**(3 * (1. + wp + (wa / (1. + z_piv_w0wa)))) * np.exp(-3 * wa * z / zp1)


@njit
def H_z(z, params):
    h, omh2, wp, wa = params[2], params[4], params[5], params[6]
    om = omh2 / h**2
    return 100 * h * np.sqrt(om * (1.0 + z)**3 + (1.0 - om) * Ode_z(z, wp, wa))


@njit
def DM_DH_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return cum_dm, dh_grid


@njit
def DM_z(z, dm_dh_interp):
    return interp_hermite(z, x=z_grid, y=dm_dh_interp[0], y_prime=dm_dh_interp[1])


@njit
def DH_z(z, dm_dh_interp):
    return interp_pchip(z, z_grid, dm_dh_interp[1])


@njit
def DV_z(z, DM, DH):
    return (z * DH * DM**2) ** (1 / 3)


dv_rs, dm_rs, dh_rs, f_ap = range(4)
qty_map = {"DV_over_rs": dv_rs, "DM_over_rs": dm_rs, "DH_over_rs": dh_rs, "F_AP": f_ap}
desi_qty = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def theory_bao(z, qty, params):
    DV_mask = qty == dv_rs
    DM_mask = qty == dm_rs
    DH_mask = qty == dh_rs
    FAP_mask = qty == f_ap

    dm_dh_grid = DM_DH_grid(params)

    inv_rd = 1 / r_drag(params[3], params[4])
    dm_vals = DM_z(z, dm_dh_grid)
    dh_vals = DH_z(z, dm_dh_grid)
    dv_vals = DV_z(z[DV_mask], dm_vals[DV_mask], dh_vals[DV_mask])

    results = np.empty(z.size, dtype=np.float64)

    results[DH_mask] = dh_vals[DH_mask] * inv_rd
    results[DM_mask] = dm_vals[DM_mask] * inv_rd
    results[DV_mask] = dv_vals * inv_rd
    results[FAP_mask] = dm_vals[FAP_mask] / dh_vals[FAP_mask]
    return results


@njit
def chi_squared(params, L_cc):
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    y_cc = solve_triangular(L_cc, delta_cc)

    delta_bao = data["value"] - theory_bao(data["z"], desi_qty, params)
    y_bao = solve_triangular(cho_bao, delta_bao)
    return np.dot(y_cc, y_cc) + np.dot(y_bao, y_bao)


method_f = method == "F"
z_pivot = 1.21
params_fid = [0.0, 1.0, 0.6749, 0.02223, 0.1421, -1.0, 0.0]
shape_fid = (1. + z_cc_vals[method_f]) * H_z(z_cc_vals[method_f], params_fid)**2
shape_piv = (1. + z_pivot) * H_z(z_pivot, params_fid)**2
f_shape = shape_fid / shape_piv


@njit
def get_fz(params):
    fp, n = np.exp(params[0]), params[1]
    fz = np.full_like(z_cc_vals, fp)
    fz[method_f] *= f_shape ** n
    return fz


@njit
def log_likelihood_jit(params):
    cov_mat_cc = cc_sys_cov_matrix + np.diag(diag_stat_cc**2 * get_fz(params)**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2.0 * np.sum(np.log(np.diag(L_cc)))

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc
    return -0.5 * chi_squared(params, L_cc) - 0.5 * normalization_cc


def log_likelihood(params):
    if params[5] + (params[6] / (1.0 + z_piv_w0wa)) > -1/3:
        return -np.inf
    return log_likelihood_jit(params)


def main():
    from multiprocessing import Pool
    from getdist import MCSamples, plots
    from nautilus import Sampler, Prior
    import matplotlib.pyplot as plt
    from ohd.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    # ---- CCH parameters for overestimated errors ----
    prior.add_parameter("ln_fp_cc", dist=(-1.5, 0.5))
    prior.add_parameter("n_cc", dist=(-2.0, 4.0))
    # ---- cosmological parameters ----
    prior.add_parameter("h", dist=(0.5, 1.0))
    prior.add_parameter("obh2", dist=(0.01, 0.04))
    prior.add_parameter("omh2", dist=(0.05, 0.25))
    prior.add_parameter("wp", dist=(-1.5, -0.5))
    prior.add_parameter("wa", dist=(-6.0, 6.0))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    labels=["ln(f_{p,cc})", "n_{cc}", "h", "Ω_b h^2", "Ω_m h^2", "w_{piv}", "w_a"]
    gd_samples = MCSamples(samples=samples, weights=weights, names=prior.keys, labels=labels)
    gd_samples.addDerived(100 * gd_samples["h"], name="H0", label="H_0")
    gd_samples.addDerived(gd_samples["omh2"] / gd_samples["h"]**2, name="om", label="Ω_m")
    gd_samples.addDerived(r_drag(gd_samples["obh2"], gd_samples["omh2"]), name="rdrag", label="r_{drag}")
    gd_samples.addDerived(np.exp(gd_samples["ln_fp_cc"]), name="fp_cc", label="f_{p,cc}")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    fz_cc = get_fz(best_fit)
    cov_mat_cc = np.diag(diag_stat_cc**2 * fz_cc**2) + cc_sys_cov_matrix
    L_cc = np.linalg.cholesky(cov_mat_cc)

    print(f"Chi squared (MAP): {chi_squared(best_fit, L_cc):.2f}")
    print(f"log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"DOF: {N_bao + N_cc - len(best_fit)}")

    plots.getSubplotPlotter().triangle_plot(
        gd_samples,
        params=["H0", "om", "obh2", "wp", "wa", "rdrag"],
        filled=True,
        title_limit=1,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()
    plots.getSubplotPlotter().triangle_plot(
        gd_samples,
        params=["H0", "om", "n_cc", "ln_fp_cc"],
        filled=True,
        title_limit=1,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_bao(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=f"{bao_legend}: $r_d$={gd_samples['rdrag'].mean():.2f}",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cc_sys_cov_matrix) + diag_stat_cc**2),
        label=f"{cc_legend}: $H_0$={gd_samples['H0'].mean():.1f} km/s/Mpc",
        method=method,
        err_scaling=1 / fz_cc,
    )


if __name__ == "__main__":
    main()


# ********************************
# Data sets:
# - DESI DR2 + FS Lya
# - CCH compilation
# ********************************


# ---------------------------------
# Assuming standard early universe physics with r_drag
# as a function of the baryon density and matter density.
# ---------------------------------


# ----------- Flat ΛCDM -----------
# H0 = 69.9 ± 1.4 km/s/Mpc
# Ωm = 0.3022 ± 0.0076
# Ωb h^2 = 0.0239 ± 0.0018
# Ωm h^2 = 0.1476 ± 0.0063
# rd = 144.8 ± 2.8 Mpc
#
# n_cc = 1.23 +0.36 -0.62
# ln(fp_cc) = -0.41 ± 0.25
# fp_cc = 0.69 +0.13 -0.19
#
# Chi squared (MAP): 51.45
# log likelihood (MAP): -157.27
# Log evidence: -169.05
# DOF: 48
# ---------------------------------


# ----------- Flat wCDM -----------
# H0 = 69.5 ± 1.6 km/s/Mpc
# Ωm = 0.3025 ± 0.0079
# Ωb h^2 = 0.0245 +0.0021 -0.0024
# Ωm h^2 = 0.1461 ± 0.0069
# w = -0.968 ± 0.070 (prior ~U[-3, 1])
# rd = 144.6 ± 2.9 Mpc
#
# n_cc = 1.22 +0.35 -0.61
# ln(fp_cc) = -0.41 ± 0.25
# fp_cc = 0.69 +0.13 -0.19
#
# Chi squared (MAP): 51.31
# log likelihood (MAP): -157.13
# Log evidence: -172.04
# DOF: 47
# ---------------------------------


# ---------- Flat w0waCDM----------
# Enforced w0 + wa <= -1/3 in the likelihood
#
# H0 = 64.9 +2.7 -3.1 km/s/Mpc
# Ωm = 0.369 ± 0.037
# Ωb h^2 = 0.0221 +0.0018 -0.0023
# Ωm h^2 = 0.1546 ± 0.0075
# w0 = -0.38 ± 0.35 (prior ~U[-3, 1])
# wa = -2.2 ± 1.2 (prior ~U[-6, 6])
# rd = 144.5 ± 2.9 Mpc
#
# n_cc = 1.28 +0.36 -0.62
# ln(fp_cc) = -0.36 ± 0.25
# fp_cc = 0.72 +0.13 -0.20
#
# Chi squared (MAP): 45.79
# log likelihood (MAP): -155.38
# Log evidence: -171.93
# DOF: 46
#
#
# At z_pivot = 0.38 corr(wp, wa) = -0.0034
# H0 = 64.9 ± 2.9 km/s/Mpc
# Ωm = 0.369 ± 0.037
# Ωb h^2 = 0.0221 +0.0018 -0.0023
# Ωm h^2 = 0.1545 ± 0.0075
# wp = -0.979 ± 0.067 (prior ~ U[-1.5, -0.5])
# wa = -2.2 ± 1.2 (prior ~ U[-6, 6])
# rd = 144.5 ± 2.9 Mpc
#
# n_cc = 1.28 +0.36 -0.62
# ln(fp_cc) = -0.36 ± 0.25
# fp_cc = 0.72 +0.13 -0.20
#
# Chi squared (MAP): 46.24
# log likelihood (MAP): -155.40
# Log evidence: -170.54
# DOF: 46
# ---------------------------------
