from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, H_err, cc_stat_cov_matrix = get_cc_data(split_sys=True)
bao_legend, data, bao_cov_matrix = get_bao_data()

cho_bao = np.linalg.cholesky(bao_cov_matrix)
N_cc = len(z_cc_vals)

c = c0 / 1000  # Speed of light in km/s

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0):
    cubic = (1.0 + z) ** 3
    return cubic**(1. + w0) #  wCDM
    # Thawing quintessence wzCDM
    # return (2 * cubic / (1.0 + w0 + (1.0 - w0) * cubic)) ** 2


@njit
def H_z(z, params):
    h0, om, w0 = params[2], params[4], params[5]
    return h0 * np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om) * Ode_z(z, w0))


@njit
def DM_grid(params):
    dh_grid = c / H_z(z_grid, params)
    n = z_grid.size
    cum_dm = np.zeros(n, dtype=np.float64)
    d_dh = np.empty(n, dtype=np.float64)
    # Central difference for internal points
    d_dh[1:-1] = (dh_grid[2:] - dh_grid[:-2]) / (2 * dz)
    # Forward/Backward difference at boundaries
    d_dh[0] = (dh_grid[1] - dh_grid[0]) / dz
    d_dh[-1] = (dh_grid[-1] - dh_grid[-2]) / dz
    # Integrate with 4th-order cubic correction per interval
    dz_sq_over_12 = (dz ** 2) / 12
    acc = 0.0

    for i in range(n - 1):
        # trapezoidal area + 1st-derivative endpoint correction
        trap = 0.5 * dz * (dh_grid[i] + dh_grid[i + 1])
        corr = dz_sq_over_12 * (d_dh[i] - d_dh[i + 1])

        acc += trap + corr
        cum_dm[i + 1] = acc

    return (cum_dm, dh_grid)


@njit
def DM_z(z, dm_interp):
    return interp_hermite(z, z_grid, dm_interp[0], dm_interp[1])


@njit
def DH_z(z, dm_interp):
    return interp_pchip(z, z_grid, dm_interp[1])


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

    dm_grid = DM_grid(params)

    inv_rd = 1 / params[3]
    dm_vals = DM_z(z, dm_grid)
    dh_vals = DH_z(z, dm_grid)
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


@njit
def log_likelihood_jit(params):
    f0_cc, n_cc = params[0:2]
    f_cc_arr = np.exp(f0_cc) * (1.0 + z_cc_vals)**n_cc
    if np.any(f_cc_arr <= 1e-4):
        return -np.inf

    cov_mat_cc = np.diag(H_err**2 / f_cc_arr**2) + cc_stat_cov_matrix
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2.0 * np.sum(np.log(np.diag(L_cc)))

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc
    return -0.5 * chi_squared(params, L_cc) - 0.5 * normalization_cc


def log_likelihood(params):
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
    prior.add_parameter("ln_f0_cc", dist=(-0.5, 2.5))
    prior.add_parameter("n_cc", dist=(-4.0, 4.0))
    # ---- cosmological parameters ----
    prior.add_parameter("H0", dist=(45.0, 90.0))
    prior.add_parameter("rd", dist=(120.0, 175.0))
    prior.add_parameter("om", dist=(0.1, 0.7))
    prior.add_parameter("w0", dist=(-1.5, -0.5))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    labels=["\\ln(f_{0,cc})", "n_{cc}", "H_0", "r_{drag}", "Ω_m", "w_0"]
    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(
        gd_samples["om"] * (gd_samples["H0"] / 100)**2 , name="omh2", label="Ω_m h^2",
    )
    gd_samples.addDerived(np.exp(gd_samples["ln_f0_cc"]), name="f0_cc", label="f_{0,cc}")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    f_cc_arr = np.exp(best_fit[0]) * (1.0 + z_cc_vals)**best_fit[1]
    cov_mat_cc = np.diag(H_err**2 / f_cc_arr**2) + cc_stat_cov_matrix
    L_cc = np.linalg.cholesky(cov_mat_cc)

    print(f"Chi squared (MAP): {chi_squared(best_fit, L_cc):.2f}")
    print(f"log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"DOF: {len(data) + len(z_cc_vals) - len(best_fit)}")

    plots.getSubplotPlotter().triangle_plot(
        gd_samples,
        params=["H0", "om", "rd", "w0", "ln_f0_cc", "n_cc"],
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
        title=f"{bao_legend}: $r_d$={best_fit[3]:.2f}",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=H_err,
        label=f"{cc_legend}: $H_0$={best_fit[2]:.1f} km/s/Mpc",
        err_scaling=f_cc_arr,
    )


if __name__ == "__main__":
    main()


# ********************************
# Data sets:
# - DESI DR2 + FS Lya
# - CCH compilation
# ********************************


# ----------- Flat ΛCDM -----------
# H0 = 68.0 ± 2.9 km/s/Mpc
# rd = 149.0 +5.7 -6.5 Mpc
# Ωm = 0.3019 ± 0.0076
# Ωm h^2 = 0.140 ± 0.012
# n_cc = -1.35\pm 0.46
# ln(f0_cc) = 1.14 +0.26 -0.23
# f0_cc = 3.22 +0.64 -0.90
# Chi squared (MAP): 50.88
# log likelihood (MAP): -156.53
# Log evidence: -170.08
# DOF: 48
# ---------------------------------


# ----------- Flat wCDM -----------
# H0 = 67.7 ± 3.1 km/s/Mpc
# rd = 149.1 +5.8 -6.6 Mpc
# Ωm = 0.3019 ± 0.0080
# Ωm h^2 = 0.139 +0.011 -0.013
# w0 = -0.982 ± 0.070 (prior U[-1.5, -0.5])
# n_cc = -1.34 ± 0.46
# ln(f0_cc) = 1.13 +0.27 - 0.23
# Chi squared (MAP): 49.24
# log likelihood (MAP): -156.50
# Log evidence: -171.78
# DOF: 47
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
# H0 = 66.2 ± 3.0
# rd = 149.6 +5.7 -6.7
# Ωm = 0.3120 +0.0093 -0.011
# Ωm h^2 = 0.137 ± 0.012
# w0 = -0.855 +0.048 -0.14 (prior U[-1, 0]. Posterior truncated to the left of the mean)
# n_cc = -1.34 ± 0.47
# ln(f0_cc) = 1.13 +0.27 -0.23
# f0_cc = 3.18 +0.63 -0.90
# Chi squared (MAP): 50.34
# log likelihood (MAP): -156.38
# Log evidence: -171.29
# DOF: 47
# ---------------------------------


# ---------- Flat w0waCDM----------
# Enforced w0 + wa < 0 in likelihood
#
# H0 = 64.1 +3.5 -4.1 km/s/Mpc
# rd = 150.4 +5.9 -6.9 Mpc
# Ωm = 0.346 +0.038 -0.021
# Ωm h^2 = 0.142 ± 0.013
# w0 = -0.59 +0.34 -0.21 (prior U[-3, 1])
# wa = < -1.47 (prior U[-3, 2])
# n_{cc} = -1.37 ± 0.47
# ln(f0_cc) = 1.13 +0.27 -0.24
# f0_cc = 3.19 +0.64 -0.92
# Chi squared (MAP): 47.32
# log likelihood (MAP): -155.36
# Log evidence: -172.72
# DOF: 46
# ---------------------------------
