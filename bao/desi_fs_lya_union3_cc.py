from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2026union3_1.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(cov_matrix_bao)
inv_cov_cc = np.linalg.inv(cov_matrix_cc)

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    inv_a3 = (1.0 + z) ** 3
    return (2 * inv_a3 / (1.0 + w0 + (1.0 - w0) * inv_a3)) ** 2


@njit
def H_z(z, params):
    H0, Om = params[2], params[4]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
desi_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    rdrag = params[3]
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    FAP_mask = qty == 3
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params) / rdrag
    results[DM_mask] = DM_z(z[DM_mask], params) / rdrag
    results[DV_mask] = DV_z(z[DV_mask], params) / rdrag
    results[FAP_mask] = DM_z(z[FAP_mask], params) / DH_z(z[FAP_mask], params)
    return results


@njit
def mu_corr(params, DM_obs):
    # Heaviside step at z = 0.2
    v_km_s = 100 * params[5] * np.where(z_cmb <= 0.2, 1, -1)
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + v_km_s / c)
    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_obs)


@njit
def mu_theory(params, DM):
    return params[1] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


def chi_squared(params):
    DM = DM_z(z_cmb, params)
    delta_sn = mu_vals - mu_theory(params, DM) - mu_corr(params, DM)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    f_cc = params[0]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = delta_cc @ (inv_cov_cc * f_cc**2) @ delta_cc

    return chi_sn + chi_bao + chi_cc


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


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
    # f_cc: CCH covariance rescaling (overestimated uncertainties)
    prior.add_parameter("fcc", dist=(0.01, 3.0))
    # ΔM: magnitude offset
    prior.add_parameter("dM", dist=(-1, 1))
    # H0: Hubble constant at present
    prior.add_parameter("H0", dist=(45, 90))
    # rd: sound horizon at drag epoch
    prior.add_parameter("rd", dist=(100, 200))
    # Ωm: matter density parameter today
    prior.add_parameter("Om", dist=(0.2, 0.50))
    # v: velocity step correction observed redshift SNe
    prior.add_parameter("v", dist=(-10.5, 4.5))

    with Pool(8) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    gd_samples = MCSamples(
        samples=samples,
        weights=w,
        names=prior.keys,
        labels=["f_{cc}", "ΔM", "H_0", "r_{drag}", "Ω_m", "v_{100}"],
        loglikes=log_l,
    )
    gd_samples.addDerived(
        gd_samples["Om"] * (gd_samples["H0"] / 100) ** 2, name="Omh2", label="Ω_m h^2"
    )
    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples, title_limit=1, color=["C0"], contour_colors=["C0"]
    )
    plt.show()

    one_sigma_ci = [0.159, 0.5, 0.841]

    fcc_16, fcc_50, fcc_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    h0_16, h0_50, h0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    v_16, v_50, v_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 4] * samples[:, 2] ** 2 / 100**2
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)

    best_fit = [fcc_50, dM_50, h0_50, rd_50, Om_50, v_50]
    deg_of_freedom = len(z_cmb) + len(bao_data) + N_cc - len(best_fit)

    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωm h^2: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}"
    )
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit, DM_z(z_cmb, best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit, DM_z(z_cmb, best_fit)),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / fcc_50,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )


if __name__ == "__main__":
    main()


# *******************************************
# Dataset: BAO DESI DR2 + FS Lya + SN1a Union3.1 + Cosmic Chronometers

# Priors:
# f_cc: U(0.01, 3.0)
# ΔM:   U(-1.0, 1.0)
# H0:   U(45.0, 90.0)
# rd:   U(100.0, 200.0)
# Ωm:   U(0.2, 0.50)

# wzCDM:
# w0:   U(-1.0, -1/3)

# wCDM:
# w0:   U(-1.5, -0.5)

# w0waCDM:
# w0:   U(-1.5, 0.0)
# wa:   U(-5.0, 3.0)
# Enforced w0 + wa < 0, forbidden prior region removed in evidence calculation.

# Velocity step correction:
# v ~U(-10.5, 4.5) x 100 km/s
# *******************************************


# --------------- Flat ΛCDM -----------------
# ΔM: -0.047 +- 0.071 mag
# H0: 68.4 +- 2.3 km/s/Mpc
# r_d: 147.5 +4.4 -4.9 Mpc
# Ωm: 0.3054 +- 0.0073
# Ωm h^2: 0.1432 +- 0.0093
# f_cc: 1.51 +- 0.17
# Chi squared: 80.60
# Log evidence: -187.61
# Degrees of freedom: 69
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# Velocity step correction SNe redshift (turning point z <= 0.2 inflow z > 0.2 outflow)
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# v: -3.0 +- 1.1 x 100 km/s
# ΔM: -0.043 +-0.071 mag
# H0: 68.7 +- 2.3 km/s/Mpc
# r_d: 147.4 +4.4 -4.9 Mpc
# Ωm: 0.3020 +- 0.0073
# Ωm h^2: 0.1426 +- 0.0093
# f_cc: 1.51 +- 0.17
# Chi squared: 72.32 (2.88 sigma significance)
# Log evidence: -185.26 (Δ logZ = 2.35 against no velocity step correction)
# Degrees of freedom: 68
# -------------------------------------------


# --------------- Flat wCDM -----------------
# ΔM: -0.055 +- 0.071 mag
# H0: 67.5 +- 2.3 km/s/Mpc
# r_d: 147.7 +4.4 -5.0 Mpc
# Ωm: 0.3045 +- 0.0075
# Ωm h^2: 0.1389 +- 0.0095
# w0: -0.931 +- 0.046
# f_cc: 1.50 +- 0.17
# Chi squared: 78.15 (1.57 sigma significance)
# Log evidence: -188.62 (Δ logZ = -1.01 in favour of ΛCDM)
# Degrees of freedom: 68
# -------------------------------------------


# --------------- Flat wzCDM ----------------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
# ΔM: -0.057 +- 0.071 mag
# H0: 67.0 +- 2.3 km/s/Mpc
# r_d: 147.7 +- 4.8 Mpc
# Ωm: 0.3120 +- 0.0081
# Ωm h^2: 0.1403 +- 0.0093
# w0: -0.856 +- 0.066
# wa: d w(z)/d z at z=0 = -1.5 * (1 - w0^2)
# f_cc: 1.50 +- 0.17
# Chi squared: 76.50 (2.02 sigma significance)
# Log evidence: -187.10 (Δ logZ = 0.51 against ΛCDM)
# Degrees of freedom: 68
# -------------------------------------------


# -------------- Flat w0waCDM ---------------
# ΔM: -0.051 +- 0.072 mag
# H0: 66.8 +- 2.3 km/s/Mpc
# r_d: 147.5 +4.4 -5.0 Mpc
# Ωm: 0.327 +0.015 -0.012
# Ωm h^2: 0.146 +-0.011
# w0: -0.78 +- 0.10
# wa: -0.89 +- 0.53
# f_cc: 1.49 +- 0.17
# Chi squared: 74.31 (2.02 sigma significance)
# Log evidence: -189.35 + 0.37 (Δ logZ = -1.37 in favour of ΛCDM)
# Degrees of freedom: 67
# -------------------------------------------
