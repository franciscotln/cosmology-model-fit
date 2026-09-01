from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cc_cov_matrix = get_cc_data()
bao_legend, data, bao_cov_matrix = get_bao_data()

cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]
cho_cc = cho_factor(cc_cov_matrix, lower=True)[0]

logdet_cc = np.linalg.slogdet(cc_cov_matrix)[1]
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
def chi_squared(params, f_cc_arr):
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    y_cc = solve_triangular(cho_cc, f_cc_arr * delta_cc)

    delta_bao = data["value"] - theory_bao(data["z"], desi_qty, params)
    y_bao = solve_triangular(cho_bao, delta_bao)
    return np.dot(y_cc, y_cc) + np.dot(y_bao, y_bao)


@njit
def log_likelihood_jit(params):
    f0_cc, n_cc = params[0:2]
    f_cc_arr = np.exp(f0_cc) * (1.0 + z_cc_vals)**n_cc
    if np.any(f_cc_arr <= 1e-4):
        return -np.inf

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2.0 * np.log(f_cc_arr).sum()
    return -0.5 * chi_squared(params, f_cc_arr) - 0.5 * normalization_cc


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
    labels=["\\ln f_{0,cc}", "n_{cc}", "H_0", "r_{drag}", "\\Omega_m", "w_0"]
    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(
        gd_samples["om"] * (gd_samples["H0"] / 100)**2 , name="omh2", label="\\Omega_m h^2",
    )
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    f_cc_arr = np.exp(best_fit[0]) * (1.0 + z_cc_vals)**best_fit[1]

    print(f"Chi squared (MAP): {chi_squared(best_fit, f_cc_arr):.2f}")
    print(f"log likelihood (MAP): {log_likelihood(best_fit):.2f}")
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
        H_err=np.sqrt(np.diag(cc_cov_matrix)) / f_cc_arr,
        label=f"{cc_legend}: $H_0$={best_fit[2]:.1f} km/s/Mpc",
    )


if __name__ == "__main__":
    main()


# ********************************
# Data sets:
# - DESI DR2 + FS Lya
# - CCH compilation
# ********************************


# ----------- Flat ΛCDM -----------
# H0 = 68.3 ± 1.8 km/s/Mpc
# rd = 148.2 +3.6 -4.1 Mpc
# Ωm = 0.3017 ± 0.0076
# Ωm h^2 = 0.1409 ± 0.0079
# ln(f0_cc) = 1.14 +0.26 -0.23
# n_cc = -1.33 ± 0.46
# Chi squared (MAP): 51.39
# log likelihood (MAP): -155.80
# Log evidence: -169.87
# DOF: 48
# ---------------------------------


# ----------- Flat wCDM -----------
# H0 = 68.1 ± 2.0 km/s/Mpc
# rd = 148.2 +3.7 -4.1 Mpc
# Ωm = 0.3016 ± 0.0079
# Ωm h^2 = 0.1401 ± 0.0083
# w0 = -0.986 ± 0.070 (prior U[-1.5, -0.5])
# ln(f0_cc) = 1.13 +0.26 - 0.23
# n_cc = -1.3 ± 0.46
# Chi squared (MAP): 51.90
# log likelihood (MAP): -155.78
# Log evidence: -171.58
# DOF: 47
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
# H0 = 66.9 ± 2.0 km/s/Mpc
# rd = 147.9 +3.7 -4.2 Mpc
# Ωm = 0.3116 +0.0091 -0.0110
# Ωm h^2 = 0.1397 ± 0.0080
# w0 = -0.859 +0.047 -0.13 (prior U[-1, 0]. Posterior truncated to the left of the mean)
# wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
# ln(f0_cc) = 1.12 +0.26 -0.24
# n_cc = -1.32 ± 0.47
# Chi squared (MAP): 50.91
# log likelihood (MAP): -155.69
# Log evidence: -171.11
# DOF: 47
# ---------------------------------


# ---------- Flat w0waCDM----------
# Enforced w0 + wa < 0 in likelihood
#
# H0 = 64.9 +2.6 -3.1
# rd = 148.6 +3.8 -4.3
# Ωm = 0.344 +0.038 -0.022
# w0 = -0.60 +0.33 -0.22 (prior U[-3, 1])
# wa = -1.43 +0.56 -1.4 (prior U[-3, 2])
# Ωm h^2 = 0.1441 ± 0.0096
# ln(f0_cc) = 1.12 +0.27 -0.24
# n_cc = -1.34 ± 0.47
# Chi squared (MAP): 49.25
# log likelihood (MAP): -154.73
# Log evidence: -172.57
# DOF: 46
# ---------------------------------
