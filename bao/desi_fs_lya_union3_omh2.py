from numba import njit
import numpy as np
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from cmb.data_spt_planck_act_compression import Omnu_z, Omnu_h2, Or_h2, c_km_per_s
from y2026union3_1.data import get_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

L_sn = np.linalg.cholesky(cov_matrix_sn)
L_bao = np.linalg.cholesky(cov_matrix_bao)

logdet_sn = 2 * np.sum(np.log(np.diag(L_sn)))
logdet_bao = 2 * np.sum(np.log(np.diag(L_bao)))

N_bao = len(bao_data)
N_sn = len(z_cmb)

c = c_km_per_s

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
    Om_h2, Om = params[2], params[3]
    h2 = Om_h2 / Om
    zp1 = 1. + z

    rad_term = Or_h2 * zp1**4
    neutrino_term = Omnu_h2 * Omnu_z(z)
    bcdm_term = (Om_h2 - Omnu_h2) * zp1**3
    lambda_term = h2 - Om_h2 - Or_h2
    return 100. * np.sqrt(rad_term + neutrino_term + bcdm_term + lambda_term)


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
    inv_rd = 1 / params[1]
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
    z_offset = 1e-03 * params[4] * np.where(z_cmb <= 0.2, 1, -1)
    return z_cmb + z_offset


def mu_corr(params, dm_dh_grid):
    # For plotting purposes
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, dm_dh_grid) / DM_z(z_cmb, dm_dh_grid))


@njit
def mu_theory(params, DM):
    offset = params[0]
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
def chi_squared(params):
    dm_dh_grid = DM_DH_grid(params)
    return chi2_sn(params, dm_dh_grid) + chi2_bao(params, dm_dh_grid)


@njit
def log_likelihood_jit(params):
    norm_sn = logdet_sn + N_sn * np.log(2 * np.pi)
    norm_bao = logdet_bao + N_bao * np.log(2 * np.pi)
    return -0.5 * (chi_squared(params) + norm_sn + norm_bao)


def log_likelihood(params):
    return log_likelihood_jit(params)


def main():
    from scipy.stats import norm
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()

    # ΔM: supernovae magnitude zero-point offset
    prior.add_parameter("dM", dist=(-1., 1.))

    # ------ cosmological parameters ------------------
    # H0: Hubble constant at present
    prior.add_parameter("rd", dist=(100., 200.))
    # omegamh2: total matter density parameter at present
    prior.add_parameter("omegamh2", dist=norm(loc=0.14332, scale=0.00091))
    # Ωm: matter density parameter today
    prior.add_parameter("om", dist=(0.2, 0.5))
    # dz_1000: (1000 x Δz) redshift offset step correction
    prior.add_parameter("dz_1000", dist=(-3.5, 3.5))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    labels=["ΔM", "r_{drag}", "Ω_m h^2", "Ω_m", "1000 Δz"]
    gd_samples = MCSamples(samples=samples, weights=w, names=prior.keys, labels=labels)
    gd_samples.addDerived(
        100 * np.sqrt(gd_samples["omegamh2"] / gd_samples["om"]),
        name="H0",
        label="H_0",
    )
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    DOF = N_sn + N_bao - len(best_fit)
    chi2 = chi_squared(best_fit)

    print(f"log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"Chi2 (MAP): {chi2:.2f}")
    print(f"DOF: {DOF}")
    print(f"Reduced Chi2 (MAP): {chi2/DOF:.2f}")

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
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
        y_err= np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit, DM_z(z_cmb, dm_dh_grid)),
        label=f"$Ω_m$={best_fit[3]:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# *******************************************
# Data sets:
# BAO DESI DR2 + FS Lya
# SN1a Union3.1
# ωm from SPA ~ N(0.14332, 0.00091^2)
# *******************************************


# ----------------- Priors ------------------
# ΔM (mag):  U[-1, 1]
# rd (Mpc):  U[100, 200]
# Ωm:        U[0.2, 0.5]
# ωm:        N(0.14332, 0.00091^2)
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
# H0 = 68.58 ± 0.86 km/s/Mpc
# r_d = 147.1 ± 1.1 Mpc
# Ωm = 0.3048 ± 0.0074
# Ωm h^2 = 0.14331 ± 0.00090
# ΔM = -0.041 ± 0.022 mag
#
# log likelihood (MAP): 45.86
# Log evidence: 33.19
# Chi2 (MAP): 42.77
# DOF: 32
# Reduced Chi2 (MAP): 1.34
# -------------------------------------------


# --------------- Flat ΛCDM -----------------
# Redshift offset step correction for SNe observed redshifts
# turning point z <= 0.2 positive z > 0.2 negative
# z_cosmo = z_cmb ± Δz

# H0 = 68.96 ± 0.88 km/s/Mpc
# r_d = 146.7 ± 1.1 Mpc
# Ωm = 0.3014 ± 0.0074
# Ωm h^2 = 0.14331 ± 0.00090
# 1000 Δz = 1.10 ± 0.40
# ΔM = -0.035 ± 0.022 mag
#
# log likelihood (MAP): 49.75
# Log evidence: 35.10
# Chi2 (MAP): 35.00
# DOF: 31
# Reduced Chi2 (MAP): 1.13
# -------------------------------------------


# --------------- Flat wCDM -----------------
# H0 = 68.68 ± 0.88 km/s/Mpc
# r_d = 145.0 +2.0 -1.8 Mpc
# Ωm = 0.3040 ± 0.0075
# Ωm h^2 = 0.14331 ± 0.00091
# w = -0.927 ± 0.047
# ΔM = -0.016 ± 0.028 mag
#
# log likelihood (MAP): 47.10
# Log evidence: 32.28
# Chi2 (MAP): 40.30
# DOF: 31
# Reduced Chi2 (MAP): 1.30
# -------------------------------------------


# -------------- Flat w1w2CDM ---------------
# w(z) = w1 + w2 * ((1 + z)^2 - 1) / ((1 + z)^2 + 1)

# H0 = 66.1 +1.1 -1.5 km/s/Mpc
# r_d = 148.9 +2.5 -1.6 Mpc
# Ωm = 0.328 +0.015 -0.012
# Ωm h^2 = 0.14332 ± 0.00090
# w1 = -0.769 ± 0.098
# w2 = -0.81 ± 0.43
# ΔM = -0.072 +0.024 -0.036 mag
#
# log likelihood (MAP): 48.93
# Log evidence: 31.86
# Chi2 (MAP): 36.64
# DOF: 30
# Reduced Chi2 (MAP): 1.22
# -------------------------------------------
