from numba import njit
import numpy as np
from interpolator import interp_pchip, interp_hermite
from solve_triangular import solve_triangular
from y2025BAO.data_fs_lya import get_data
import cmb.data_spt_planck_act_compression as cmb

or_h2 = cmb.Or_h2
omnu_h2 = cmb.Omnu_h2
c = cmb.c_km_per_s  # Speed of light in km/s
RD = 147.09  # Mpc, fixed

legend, data, cov_matrix = get_data()
L_cov = np.linalg.cholesky(cov_matrix)

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0, wa):
    # w1w2CDM
    zp1 = 1. + z
    return zp1**(3. * (1. + w0 + wa)) * (2. * zp1**2 / (zp1**2 + 1.))**(-3. * wa)


@njit
def h_z(z, params):
    h, om = params
    h2 = h**2
    om_h2 = om * h2
    obcdm_h2 = om_h2 - omnu_h2
    olambda_h2 = h2 - om_h2 - or_h2
    zp1 = 1. + z
    return 100 * np.sqrt(or_h2 * zp1**4 + obcdm_h2 * zp1**3 + omnu_h2 * cmb.Omnu_z(z) + olambda_h2)


@njit
def DM_grid(params):
    dh_grid = c / h_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return (cum_dm, dh_grid)


@njit
def bao_theory(z, qty, theta):
    dm_grid, dh_grid = DM_grid(theta)

    dh_vals = interp_pchip(z, x=z_grid, y=dh_grid)
    dm_vals = interp_hermite(z, x=z_grid, y=dm_grid, y_prime=dh_grid)

    dv_mask = qty == 0
    dm_mask = qty == 1
    dh_mask = qty == 2
    f_ap_mask = qty == 3

    results = np.empty(z.size, dtype=np.float64)
    results[dh_mask] = dh_vals[dh_mask] / RD
    results[dm_mask] = dm_vals[dm_mask] / RD
    results[dv_mask] = (z[dv_mask] * dh_vals[dv_mask] * dm_vals[dv_mask] ** 2) ** (1 / 3) / RD
    results[f_ap_mask] = dm_vals[f_ap_mask] / dh_vals[f_ap_mask]
    return results


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def chi_squared(theta):
    delta_bao = data["value"] - bao_theory(data["z"], bao_qty, theta)
    y = solve_triangular(L_cov, delta_bao)
    return np.dot(y, y)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from multiprocessing import Pool
    from getdist import plots, MCSamples
    from nautilus import Sampler, Prior
    import matplotlib.pyplot as plt
    from bao.plot_predictions import plot_bao_predictions, plot_bao_residuals

    prior = Prior()
    prior.add_parameter("h", dist=(0.5, 0.8))
    prior.add_parameter("om", dist=(0.1, 0.8))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)

    MAP_index = np.argmax(log_l)
    best_fit = samples[MAP_index]
    dof = len(data["z"]) - len(best_fit)

    residuals = data["value"] - bao_theory(data["z"], bao_qty, best_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - ss_res / ss_tot
    chi2 = chi_squared(best_fit)

    labels=["h", "Ω_m"]
    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        names=prior.keys,
        labels=labels,
        label="BAO + FS Lyman-alpha",
    )
    gd_samples.addDerived(gd_samples["h"] * RD, name="hrd", label="h * r_{drag}")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    print(f"χ2 (MAP): {chi2:.2f}")
    print(f"DOF: {dof}")
    print(f"χ2/dof: {chi2 / dof:.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    plots.getSubplotPlotter().triangle_plot(
        gd_samples,
        params=["hrd", "om"],
        filled=True,
        title_limit=1,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=legend,
    )
    plot_bao_residuals(data, residuals, np.sqrt(np.diag(cov_matrix)))


if __name__ == "__main__":
    main()


# *******************************************
# Dataset: DESI BAO DR2 2025 + FS Lyman-alpha
# *******************************************

# --------------- Flat ΛCDM -----------------
# h * rd: 101.19 +- 0.67 Mpc
# Ωm: 0.3013 +- 0.0077
# χ2: 12.80
# DOF: 12
# χ2/dof: 1.07
# Log evidence: -14.1
# R^2: 0.9987
# RMSD: 0.298
# -------------------------------------------

# --------------- Flat wCDM -----------------
# h * rd: 100.5 +1.6 -1.8 Mpc
# Ωm: 0.3018 +- 0.0081
# w: -0.967 +- 0.073 (prior ~U(-1.4, -0.4))
# χ2: 12.55
# DOF: 11
# χ2/dof: 1.14
# Log evidence: -15.7
# R^2: 0.9988
# RMSD: 0.287
# -------------------------------------------

# ----------- Flat w1w2CDM --------
# Enforced w1 + w2 < 0 in the likelihood
#
# w(z) = w1 + w2 * ((1 + z)^2 - 1) / ((1 + z)^2 + 1)
#
# h * rd = 90.8 +3.8 -4.7 Mpc
# Ωm = 0.398 ± 0.044
# w1 = -0.13 ± 0.40 (prior ~U(-4, 2))
# w2 = -2.6 ± 1.2 (prior ~U(-8, 4))
# χ2 (MAP): 7.18
# DOF: 10
# χ2/dof: 0.72
# Log evidence: -16.4
# R^2: 0.9995
# RMSD: 0.194
# -------------------------------------------

# -------------- Flat w0waCDM ---------------
# Enforced w0 + wa < 0 in the likelihood
# Full wa posterior distribution
# h * rd: 90.3 +3.9 -4.9 Mpc
# Ωm: 0.402 +- 0.045
# w0: -0.04 +- 0.44 (prior ~U(-4, 2))
# wa: -3.3 +- 1.5 (prior ~U(-10, 4))
# χ2: 7.19
# DOF: 10
# χ2/dof: 0.72
# Log evidence: -16.3
# R^2: 0.9995
# RMSD: 0.195

# Truncated wa posterior distribution
# h * rd: 94.1 +2.0 -3.7 Mpc
# Ωm: 0.363 +0.033 -0.017
# w0: -0.43 +0.30 -0.14 (prior ~U(-3, 1))
# wa: < -1.94 (prior ~U(-3, 2)) - left side truncated
# χ2: 7.23
# DOF: 10
# χ2/dof: 0.72
# Log evidence: -15.7
# R^2: 0.9994
# RMSD: 0.196
# -------------------------------------------