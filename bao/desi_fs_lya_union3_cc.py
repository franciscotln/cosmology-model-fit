from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2026union3_1.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

L_sn = cho_factor(cov_matrix_sn, lower=True)[0]
L_bao = cho_factor(cov_matrix_bao, lower=True)[0]
L_cc = cho_factor(cov_matrix_cc, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
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
    return params[2] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params, f_array):
    dm_grid = DM_grid(params)

    DM_cosmo = DM_z(get_z_cosmo(params), dm_grid)
    delta_sn = mu_vals - mu_theory(params, DM_cosmo)
    chi_sn = solve_triangular(L_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], bao_qty, params[4], dm_grid)
    chi_bao = solve_triangular(L_bao, delta_bao)

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = solve_triangular(L_cc, f_array * delta_cc)

    return chi_sn + chi_bao + chi_cc


@njit
def log_likelihood(params):
    f0, fa = params[0: 2]
    fcc_arr = f0 + fa * z_cc_vals / (1.0 + z_cc_vals)
    if np.any(fcc_arr <= 1e-4):
        return -np.inf

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2.0 * np.log(fcc_arr).sum()
    return -0.5 * chi_squared(params, fcc_arr) - 0.5 * normalization_cc


def main():
    from corner import quantile
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from ohd.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    # f0: CCH covariance rescaling (overestimated uncertainties f(z) = f0 + fa * z / (1 + z))
    prior.add_parameter("f0", dist=(0.1, 6.0))
    # fa: CCH covariance rescaling (cov[i, j] = base_cov[i, j] / (fz[i] * fz[j]))
    prior.add_parameter("fa", dist=(-9.0, 9.0))
    # ΔM: magnitude zero-point offset
    prior.add_parameter("dM", dist=(-1, 1))
    # H0: Hubble constant at present
    prior.add_parameter("H0", dist=(45, 90))
    # rd: sound horizon at drag epoch
    prior.add_parameter("rd", dist=(100, 200))
    # Ωm: matter density parameter today
    prior.add_parameter("Om", dist=(0.2, 0.50))
    # v: velocity step correction observed redshift SNe
    prior.add_parameter("v", dist=(-8.5, 8.5))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    labels=["f_0", "f_a", "ΔM", "H_0", "r_{drag}", "Ω_m", "v_{100}"]
    gd_samples = MCSamples(
        samples=samples,
        weights=w,
        names=prior.keys,
        labels=labels,
        loglikes=log_l,
    )
    gd_samples.addDerived(
        gd_samples["Om"] * (gd_samples["H0"] / 100) ** 2, name="Omh2", label="Ω_m h^2"
    )
    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples, title_limit=1, color=["C0"], contour_colors=["C0"], filled=True
    )
    plt.show()

    one_sigma_ci = [0.159, 0.5, 0.841]

    f0_16, f0_50, f0_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    fa_16, fa_50, fa_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    h0_16, h0_50, h0_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)
    v_16, v_50, v_84 = quantile(samples[:, 6], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 5] * (samples[:, 3] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)

    best_fit = samples[np.argmax(log_l)]
    DOF = len(z_cmb) + len(bao_data) + N_cc - len(best_fit)

    fcc_arr = best_fit[0] + best_fit[1] * z_cc_vals / (1.0 + z_cc_vals)

    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"Ωm h^2: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"f0: {f0_50:.2f} +{(f0_84 - f0_50):.2f} -{(f0_50 - f0_16):.2f}")
    print(f"fa: {fa_50:.2f} +{(fa_84 - fa_50):.2f} -{(fa_50 - fa_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit, fcc_arr):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"DOF: {DOF}")

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
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / fcc_arr,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
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
# f0:   U[0.1, 6.0]
# fa:   U[-9.0, 9.0]
# ΔM:   U[-1.0, 1.0]
# H0:   U[45.0, 90.0]
# rd:   U[100.0, 200.0]
# Ωm:   U[0.2, 0.50]

# wzCDM:
# w0:   U[-1.0, -1/3]

# wCDM:
# w0:   U[-1.5, -0.5]

# w0waCDM:
# w0:   U[-1.5, 0.0]
# wa:   U[-5.0, 3.0]
# Enforced w0 + wa < 0, forbidden prior region removed in evidence calculation.

# Velocity step correction in observed redshift SNe:
# v:    U[-8.5, 8.5] x 100 km/s
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# H0: 68.2 +- 1.8 km/s/Mpc
# r_d: 148.2 +- 4.0 Mpc
# Ωm: 0.3047 +- 0.0073
# Ωm h^2: 0.1416 +- 0.0080
# ΔM: -0.056 +- 0.059 mag
# f0: 2.99 +- 0.56
# fa: -3.3 +- 1.1
# Chi squared: 81.89
# Log evidence: -190.55
# Degrees of freedom: 69
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# Velocity step correction SNe redshift
# turning point z <= 0.2 inflow z > 0.2 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# H0: 68.3 +- 1.8 km/s/Mpc
# r_d: 148.4 +3.7 -4.1 Mpc
# Ωm: 0.3013 +- 0.0073
# Ωm h^2: 0.1405 +- 0.0079
# v: -3.0 +- 1.1 x 100 km/s
# ΔM: -0.057 +-0.058 mag
# f0: 3.00 +- 0.56
# fa: -3.4 +- 1.1
# Chi squared: 72.43 (3.07 sigma significance)
# Log evidence: -188.30 (Δ logZ = 2.25 in favour of velocity step correction)
# Degrees of freedom: 68
# -------------------------------------------


# --------------- Flat wCDM -----------------
# H0: 67.5 +- 1.9 km/s/Mpc
# r_d: 147.9 +3.7 -4.2 Mpc
# Ωm: 0.3039 +- 0.0074
# Ωm h^2: 0.1385 +- 0.0082
# w0: -0.937 +- 0.046
# ΔM: -0.058 +- 0.059 mag
# f0: 2.95 +- 0.56
# fa: -3.3 +- 1.1
# Chi squared: 80.26 (1.28 sigma significance)
# Log evidence: -191.77 (Δ logZ = -1.22 in favour of ΛCDM)
# Degrees of freedom: 68
# -------------------------------------------


# --------------- Flat wzCDM ----------------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
# H0: 66.9 +- 1.9 km/s/Mpc
# r_d: 148.1 +- 4.0 Mpc
# Ωm: 0.3109 +- 0.0080
# Ωm h^2: 0.1393 +- 0.0080
# w0: -0.862 +- 0.065
# wa: d w(z)/d z at z=0 = -1.5 * (1 - w0^2) = -0.39
# ΔM: -0.062 +- 0.059 mag
# f0: 2.95 +- 0.56
# fa: -3.3 +- 1.1
# Chi squared: 77.78 (2.03 sigma significance)
# Log evidence: -190.22 (Δ logZ = 0.33 in favour of wzCDM)
# Degrees of freedom: 68
# -------------------------------------------


# -------------- Flat w0waCDM ---------------
# H0: 66.3 +- 2.0 km/s/Mpc
# r_d: 148.8 +- 4.1 Mpc
# Ωm: 0.328 +0.015 -0.012
# Ωm h^2: 0.1440 +-0.0087
# w0: -0.77 +- 0.11
# wa: -0.97 +- 0.53
# ΔM: -0.070 +- 0.059 mag
# f0: 2.99 +- 0.56
# fa: -3.4 +- 1.1
# Chi squared: 79.22 (1.12 sigma significance)
# Log evidence: -192.22 + 0.37 (Δ logZ = -1.30 in favour of ΛCDM)
# Degrees of freedom: 67
# -------------------------------------------
