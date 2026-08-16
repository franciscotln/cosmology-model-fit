from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_pchip, interp_hermite
from y2025BAO.data_fs_lya import get_data

c = c0 / 1000  # Speed of light in km/s
RD = 147.09  # Mpc, fixed

legend, data, cov_matrix = get_data()
inv_cov_bao = np.linalg.inv(cov_matrix)

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def ode_z(z, w0):
    cubed = (1.0 + z) ** 3
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2


@njit
def h_z(z, params):
    h, o_m, w0 = params
    o_l = 1.0 - o_m
    return 100 * h * np.sqrt(o_m * (1.0 + z) ** 3 + o_l * ode_z(z, w0))


@njit
def bao_theory(z, qty, theta):
    dh_grid = c / h_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    dm_grid = np.zeros(z_grid.size, dtype=np.float64)
    dm_grid[1:] = np.cumsum(dz * dh)

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
    return delta_bao @ inv_cov_bao @ delta_bao


@njit
def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from multiprocessing import Pool
    from corner import quantile
    from getdist import plots, MCSamples
    from nautilus import Sampler, Prior
    import matplotlib.pyplot as plt
    from bao.plot_predictions import plot_bao_predictions, plot_bao_residuals

    prior = Prior()
    prior.add_parameter("h", dist=(0.5, 0.8))
    prior.add_parameter("om", dist=(0.1, 0.8))
    prior.add_parameter("w0", dist=(-1.0, 0.0))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    one_sigma_ci = [0.159, 0.5, 0.841]

    h_16, h_50, h_84 = quantile(samples[:, 0], one_sigma_ci, weights)
    om_16, om_50, om_84 = quantile(samples[:, 1], one_sigma_ci, weights)
    w0_16, w0_50, w0_84 = quantile(samples[:, 2], one_sigma_ci, weights)

    best_fit = [h_50, om_50, w0_50]
    dof = len(data["z"]) - len(best_fit)

    residuals = data["value"] - bao_theory(data["z"], bao_qty, best_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - ss_res / ss_tot
    chi2 = chi_squared(best_fit)

    print(f"h * rd: {RD * h_50:.2f} +{RD * (h_84 - h_50):.2f} -{RD * (h_50 - h_16):.2f}")
    print(f"Ωm: {om_50:.4f} +{om_84-om_50:.4f} -{om_50-om_16:.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"χ2: {chi2:.2f}")
    print(f"DOF: {dof}")
    print(f"χ2/dof: {chi2 / dof:.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=log_l,
        names=prior.keys,
        labels=["h", "Ω_m", "w_0"],
        label="BAO + FS Lyman-alpha",
    )
    gd_samples.addDerived(gd_samples["h"] * RD, name="hrd", label="h * r_{drag}")
    gd_samples.updateBaseStatistics()
    plots.getSubplotPlotter().triangle_plot(
        gd_samples,
        params=['hrd', 'om', 'w0'],
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
# h * rd: 101.17 +0.67 -0.67
# Ωm: 0.3017 +0.0077 -0.0077
# χ2: 12.81
# DOF: 12
# χ2/dof: 1.07
# Log evidence: -14.2
# R^2: 0.9987
# RMSD: 0.298
# -------------------------------------------

# --------------- Flat wCDM -----------------
# h * rd: 100.5 +1.7 -1.7
# Ωm: 0.3022 +0.0081 -0.0081
# w0: -0.967 +0.074 -0.074 (prior ~U(-1.4, -0.4))
# χ2: 12.60
# DOF: 11
# χ2/dof: 1.15
# Log evidence: -15.7
# R^2: 0.9988
# RMSD: 0.287
# -------------------------------------------

# --------------- Flat wzCDM ----------------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
# h * rd: 98.3 +2.2 -1.5
# Ωm: 0.315 +0.010 -0.012
# w0: -0.824 +0.068 -0.15 (prior ~U(-1, 0)) - left side truncated
# χ2: 12.08
# DOF: 11
# χ2/dof: 1.10
# Log evidence: -15.0
# R^2: 0.9990
# RMSD: 0.268
# -------------------------------------------

# -------------- Flat w0waCDM ---------------
# Full wa posterior distribution
# h * rd: 90.3 +3.9 -4.8
# Ωm: 0.403 +0.045 -0.045
# w0: -0.04 +0.43 -0.43 (prior ~U(-2.5, 2.5))
# wa: -3.3 +1.5 -1.5 (prior ~U(-10, 4))
# χ2: 7.22
# DOF: 10
# χ2/dof: 0.72
# Log evidence: -16.1
# R^2: 0.9995
# RMSD: 0.195

# Truncated wa posterior distribution
# h * rd: 94.1 +2.0 -3.6
# Ωm: 0.363 +0.033 -0.016
# w0: -0.43 +0.30 -0.14 (prior ~U(-3, 1))
# wa: < -1.95 (prior ~U(-3, 2)) - left side truncated
# χ2: 7.89
# DOF: 10
# χ2/dof: 0.79
# Log evidence: -15.7
# R^2: 0.9993
# RMSD: 0.216
# -------------------------------------------