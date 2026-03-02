from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2026union3_1.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data
from y2024DESBAO.data import get_data as get_des_bao_data


c = c0 / 1000  # km/s

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_desi_legend, desi_bao_data, desi_bao_cov_mat = get_bao_data()
des_bao_legend, des_bao_data, des_bao_cov_mat = get_des_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(desi_bao_cov_mat)
inv_cov_des_bao = np.linalg.inv(des_bao_cov_mat)

z_max = max(np.max(z_cmb), np.max(desi_bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def r_drag(wb, wm):
    # arXiv:2106.00428v2 (eq 8)
    a1 = 0.00257366
    a2 = 0.05032
    a3 = 0.013
    a4 = 0.7720642
    a5 = 0.24346362
    a6 = 0.00641072
    a7 = 0.5350899
    a8 = 32.7525
    a9 = 0.315473

    term_A_denominator = a1 * wb**a2 + a3 * wb**a4 * wm**a5 + a6 * wm**a7
    term_A = 1.0 / term_A_denominator
    term_B = a8 / (wm**a9)
    return term_A - term_B


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    cubed = (1.0 + z) ** 3
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2


@njit
def H_z(z, params):
    H0, Om = params[0], params[1]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
desi_qty = np.array([qty_map[q] for q in desi_bao_data["quantity"]], dtype=np.int64)
des_qty = np.array([qty_map[q] for q in des_bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / r_drag(Obh2, Omh2)


pivot_mask = z_cmb <= 0.2


@njit
def mu_corr(params):
    z_pec = 100 * params[3] / c
    z_cosmo1 = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    z_cosmo2 = -1.0 + (1.0 + z_cmb) / (1.0 - z_pec)

    DM_ref = DM_z(z_cmb, params)

    return np.where(
        pivot_mask,
        5.0 * np.log10(DM_z(z_cosmo1, params) / DM_ref),
        5.0 * np.log10(DM_z(z_cosmo2, params) / DM_ref),
    )


@njit
def theory_mu(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[4] + 25.0 + 5 * np.log10(dL)


@njit
def chi_squared(params):
    delta_bao = desi_bao_data["value"] - bao_theory(
        desi_bao_data["z"], desi_qty, params
    )
    chi_bao_desi = delta_bao @ inv_cov_bao @ delta_bao

    delta_bao_des = des_bao_data["value"] - bao_theory(
        des_bao_data["z"], des_qty, params
    )
    chi_bao_des = delta_bao_des @ inv_cov_des_bao @ delta_bao_des

    delta_sn = mu_values - theory_mu(params) - mu_corr(params)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    return chi_bao_desi + chi_bao_des + chi_sn


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def q0(Om, w0=-1.0):
    """Calculate the deceleration parameter at z=0."""
    return Om / 2 + (1.0 + 3 * w0) * (1.0 - Om) / 2


def j0(Om, w0=-1.0, wa=0.0):
    """Calculate the jerk parameter at z=0."""
    return 1.0 + (3 / 2) * (1.0 - Om) * (3 * w0 * (1.0 + w0) + wa)


def main():
    from scipy.stats import norm
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(55, 80))
    prior.add_parameter("Ωm", dist=(0.10, 0.65))
    prior.add_parameter("ωb", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("v", dist=(-12.0, 5.0))
    prior.add_parameter("dM", dist=(-1.0, 1.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    one_sigma_ci = [0.159, 0.5, 0.841]

    H0_16, H0_50, H0_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    Obh2_16, Obh2_50, Obh2_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    v_16, v_50, v_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    rd_samples = r_drag(samples[:, 2], Omh2_samples)
    q0_samples = q0(samples[:, 1])
    j0_samples = j0(samples[:, 1])

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)
    q0_16, q0_50, q0_84 = quantile(q0_samples, one_sigma_ci, weights=w)
    j0_16, j0_50, j0_84 = quantile(j0_samples, one_sigma_ci, weights=w)

    best_fit = [H0_50, Om_50, Obh2_50, v_50, dM_50]
    MAP_params = samples[np.argmax(log_l)]
    deg_of_freedom = len(des_bao_data) + len(desi_bao_data) + len(z_cmb) - len(best_fit)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"q0: {q0_50:.3f} +{(q0_84 - q0_50):.3f} -{(q0_50 - q0_16):.3f}")
    print(f"j0: {j0_50:.3f} +{(j0_84 - j0_50):.3f} -{(j0_50 - j0_16):.3f}")
    print(f"Chi squared (MAP): {chi_squared(MAP_params):.2f}")
    print(f"Log Evidence: {sampler.log_z:.2f}")
    print(f"Degs of freedom: {deg_of_freedom}")

    corner(
        samples,
        weights=w,
        labels=prior.keys,
        quantiles=one_sigma_ci,
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
        range=np.repeat(0.9999, len(prior.keys)),
    )
    plt.show()
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=desi_bao_data,
        errors=np.sqrt(np.diag(desi_bao_cov_mat)),
        title=bao_desi_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(best_fit),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
DESI DR2 + Union3 + BBN Schöngerg2024

Priors:

All models:
H0 U(55, 80)
Om U(0.10, 0.65)
ωb N(0.02218, 0.00055)
dM U(-1.0, 1.0)

wCDM:
w0 U(-1.3, -0.3)

wzCDM:
w0 U(-1.0, -1/3)

w0waCDM (w0 + wa < 0 enforced):
w0 U(-1.3, 0.0)
wa U(-4.0, 2.0)

flow correction:
v ~U(-12, 5) x 100 km/s
"""

"""
Flat ΛCDM
H0: 68.8 +0.6 -0.6 km/s/Mpc
Ωm: 0.3019 +0.0082 -0.0080
ωb: 0.02219 +0.00055 -0.00055
ωm: 0.14283 +0.00502 -0.00482
ΔM: -0.038 +0.022 -0.021
r_d: 147.15 +1.54 -1.55 Mpc
q0: -0.547 +0.012 -0.012
j0: 1
Chi squared (MAP): 41.66
Log Evidence: -32.66
Degrees of freedom: 31
"""

"""
Flat ΛCDM
Isotropic velocity SNe observed redshifts (turning point z <= 0.2 inflow z > 0.2 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

v: -311 +106 -106 km/s
ΔM: -0.044 +0.021 -0.021
H0: 68.7 +0.6 -0.6 km/s/Mpc
Ωm: 0.297 +0.008 -0.008
ωb: 0.02219 +0.00055 -0.00055
ωm: 0.1405 +0.0049 -0.0048
r_d: 147.76 +1.53 -1.54 Mpc
q0: -0.554 +0.012 -0.012
j0: 1
Chi squared (MAP): 32.94 (2.95 sigma significance)
Log Evidence: -30.17 (Δ logZ = 2.49 against no flow)
Degs of freedom: 31
"""

"""
Flat wCDM w(z) = w0
H0: 65.9 +1.5 -1.5 km/s/Mpc
Ωm: 0.2971 +0.0088 -0.0087
ωb: 0.02219 +0.00055 -0.00056
ωm: 0.12931 +0.00823 -0.00813
w0: -0.898 +0.050 -0.050
ΔM: -0.100 +0.038 -0.039
r_d: 150.87 +2.55 -2.44 Mpc
q0: -0.446 +0.049 -0.050
j0: 0.710 +0.133 -0.119
Chi squared (MAP): 37.30
Log Evidence: -32.63 (Δ logZ = 0.03 against ΛCDM)
Degrees of freedom: 31
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
H0: 66.2 +1.2 -1.2 km/s/Mpc
Ωm: 0.3075 +0.0086 -0.0084
ωb: 0.02219 +0.00055 -0.00055
ωm: 0.13462 +0.00572 -0.00558
w0: -0.819 +0.072 -0.072
wa: -0.493 +0.186 -0.169 [derived wa = -1.5 * (1 - w0**2)]
ΔM: -0.079 +0.027 -0.027
r_d: 149.37 +1.79 -1.77 Mpc
q0: -0.351 +0.078 -0.079
j0: 0.027 +0.352 -0.298
Chi squared (MAP): 35.88
Log Evidence: -31.12 (Δ logZ = 1.54 against ΛCDM)
Degrees of freedom: 31
"""

"""
Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.3 +1.5 -1.6 km/s/Mpc
Ωm: 0.3221 +0.0151 -0.0177
ωb: 0.02219 +0.00054 -0.00055
ωm: 0.14630 +0.01065 -0.01277
w0: -0.773 +0.106 -0.099
wa: -0.794 +0.545 -0.547
ΔM: -0.032 +0.044 -0.054
r_d: 146.26 +3.56 -2.78 Mpc
q0: -0.286 +0.121 -0.121
j0: -0.339 +0.717 -0.650
Chi squared (MAP): 35.17
Log Evidence: -33.26 (TODO: remove forbidden prior volume. Increases the evidence but ΛCDM is preferred)
Degs of freedom: 30
"""
