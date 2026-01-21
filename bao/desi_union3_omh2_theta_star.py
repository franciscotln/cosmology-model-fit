from numba import njit
import numpy as np
import cmb.data_early_lcdm_compression as cmb
from interpolator import interp_quad
from y2023union3.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(cov_matrix_bao)
inv_cov_cmb = np.linalg.inv(cmb.covariance[[0, 2], :][:, [0, 2]])

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000, dtype=np.float64)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0, wa):
    cubed = (1.0 + z) ** 3
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0, wa)

    return np.sqrt(radiation_term + matter_term + neutrino_term + dark_energy_term)


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0 = params[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_quad(z, z_grid, cum_dm)


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, theta):
    Obh2, Och2 = theta[2], theta[3]
    rd = cmb.r_drag(Obh2, Obh2 + Och2 + Omnuh2)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


@njit
def theory_mu(theta):
    dL = (1.0 + z_sn_vals) * DM_z(z_sn_vals, theta)
    return theta[0] + 25.0 + 5 * np.log10(dL)


def chi_squared(theta):
    delta_cmb = (
        cmb.DISTANCE_PRIORS[[0, 2]]
        - cmb.cmb_distances(Ez, theta[2], theta[3], theta)[[0, 2]]
    )
    chi_cmb = delta_cmb @ inv_cov_cmb @ delta_cmb

    delta_sn = mu_vals - theory_mu(theta)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao
    return chi_sn + chi_bao + chi_cmb


def log_likelihood(theta):
    return -0.5 * chi_squared(theta)


def main():
    from multiprocessing import Pool
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("ΔM", dist=(-1.0, +1.0))
    prior.add_parameter("H0", dist=(50.0, 90.0))
    prior.add_parameter("ωb", dist=(0.01, 0.04))
    prior.add_parameter("ωc", dist=(0.05, 0.3))
    prior.add_parameter("w0", dist=(-1.0, -1 / 3))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=8_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    one_sigma_ci = [0.159, 0.5, 0.841]

    dM_16, dM_50, dM_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    H0_16, H0_50, H0_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    Obh2_16, Obh2_50, Obh2_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    Och2_16, Och2_50, Och2_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    zd_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    rd_samples = cmb.r_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)
    zd_16, zd_50, zd_84 = quantile(zd_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, H0_50, Obh2_50, Och2_50, w0_50]
    degrees_of_freedom = 2 + len(bao_data["z"]) + len(z_sn_vals) - len(best_fit)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z_d: {zd_50:.2f} +{(zd_84 - zd_50):.2f} -{(zd_50 - zd_16):.2f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {sampler.log_z:.2f}")
    print(f"Degs of freedom: {degrees_of_freedom}")

    labels = ["$Δ_M$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    corner(
        samples,
        weights=w,
        labels=labels,
        quantiles=one_sigma_ci,
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
        range=np.repeat(0.9999, len(labels)),
    )
    plt.show()

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
*******************************

BAO DESI DR2
SNIa Union3
(θ*, ωm = Ωm x h²) CMB Early time ΛCDM

Priors:

All models:
ΔM U(-1.0, +1.0)
H0 U(50.0, 90.0)
ωb U(0.01, 0.04)
ωc U(0.05, 0.3)

wCDM:
w0 U(-1.3, -0.3)

wzCDM:
w0 U(-1.0, -1/3)

w0waCDM:
w0 U(-1.3, 0.0)
wa U(-3.5, 2.0)
w0 + wa < 0 enforced (account for that in the prior volume later)

*******************************
"""

"""
Flat ΛCDM  w(z) = -1
H0: 69.20 +0.83 -0.83 km/s/Mpc
ωb: 0.02345 +0.00095 -0.00096
ωc: 0.1174 +0.0007 -0.0007
ωm: 0.1415 +0.0012 -0.0012
Ωm: 0.296 +0.006 -0.006
w0: -1
wa: 0
z_d: 1062.31 +2.09 -2.19
r_d: 146.58 +1.07 -1.04 Mpc
Chi squared: 40.73
Log Evidence: -35.43
Degs of freedom: 33

===============================

Flat wCDM w(z) = w0
H0: 68.80 +0.78 -0.79 km/s/Mpc
ωb: 0.02679 +0.00157 -0.00153
ωc: 0.1148 +0.0012 -0.0013
ωm: 0.1422 +0.0012 -0.0012
Ωm: 0.300 +0.006 -0.006
w0: -0.880 +0.040 -0.041
wa: 0
z_d: 1069.23 +3.06 -3.13
r_d: 143.77 +1.41 -1.37 Mpc
Chi squared: 32.31
Log Evidence: -33.56 (Δ logZ = 1.87 against ΛCDM)
Degs of freedom: 32

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
H0: 67.51 +0.93 -0.92 km/s/Mpc
ωb: 0.02523 +0.00107 -0.00108
ωc: 0.1162 +0.0008 -0.0008
ωm: 0.1421 +0.0012 -0.0012
Ωm: 0.312 +0.008 -0.008
w0: -0.765 +0.067 -0.070
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
z_d: 1066.11 +2.21 -2.32
r_d: 145.00 +1.12 -1.08 Mpc
Chi squared: 29.96
Log Evidence: -31.49 (Δ logZ = 3.94 against ΛCDM)
Degs of freedom: 32

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 66.23 +1.59 -1.46 km/s/Mpc
ωb: 0.02271 +0.00246 -0.00211
ωc: 0.1185 +0.0017 -0.0021
ωm: 0.1418 +0.0012 -0.0012
Ωm: 0.323 +0.014 -0.014
w0: -0.711 +0.112 -0.107
wa: -0.855 +0.494 -0.517
z_d: 1060.79 +5.28 -4.92
r_d: 147.11 +2.05 -2.18 Mpc
Chi squared: 29.52
Log Evidence: -33.76 + 0.28 = -33.48 (Δ logZ = 1.98 against ΛCDM)
Degs of freedom: 31
"""
