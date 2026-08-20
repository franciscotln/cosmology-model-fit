from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
from interpolator import interp_hermite, interp_pchip
import cmb.data_planck_act_compression as cmb
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2026union3_1.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data
import y2018fs8.data as fs8

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

fs8_data = fs8.data
z_fs8 = fs8_data["z"]
a_fs8 = 1 / (1.0 + z_fs8)
N_fs8 = len(z_fs8)

inv_cov_fs8 = np.linalg.inv(fs8.cov_mat)
inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(cov_matrix_bao)

z_max = max(np.max(z_cmb), np.max(bao_data["z"]), np.max(z_fs8)) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def d_Omnu_dz(z):
    return cmb.Omnu_z(z) * 3 * (1.0 + cmb.w_nu_z(z)) / (1.0 + z)


@njit
def w_de_z(z, w0):
    # Thawing quintessence wzCDM
    return -1.0 + 2 * (1.0 + w0) / (1.0 + w0 + (1.0 - w0) * (1.0 + z) ** 3)


@njit
def Ode_z(z, w0):
    # Thawing quintessence wzCDM
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2


@njit
def d_Ode_dz(z, w0):
    return Ode_z(z, w0) * 3 * (1.0 + w_de_z(z, w0)) / (1.0 + z)


@njit
def Ez(z, h, Obh2, Och2):
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return np.sqrt(radiation_term + matter_term + neutrino_term + dark_energy_term)


@njit
def H_z(z, theta):
    H0 = theta[1]
    return H0 * Ez(z, h=H0 / 100, Obh2=theta[2], Och2=theta[3])


cmb.set_HZ(H_z)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


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
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnuh2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


@njit
def mu_corr(params, DM_ref):
    v_km_s = 100 * params[4] * np.where(z_cmb <= 0.2, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_ref)


@njit
def theory_mu(theta, DM):
    return theta[0] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def dH_da(z, H_vals, theta):
    H0, Obh2, Och2 = theta[1], theta[2], theta[3]
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2
    Or = Orh2 / h**2
    Onu = Omnuh2 / h**2
    numerator = 3 * Obc * (1.0 + z) ** 2 + 4 * Or * (1.0 + z) ** 3 + Onu * d_Omnu_dz(z)
    denominator = 2 * H_vals / (1.0 + z) ** 2
    return -numerator * H0**2 / denominator


@njit
def growth_ODE(a, y, theta):
    H0, Obh2, Och2 = theta[1], theta[2], theta[3]
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2

    z = 1 / a - 1.0
    H_vals = H_z(z, theta)
    dH_da_vals = dH_da(z, H_vals, theta)

    delta, d_delta_da = y

    source = 1.5 * Obc * H0**2 * delta / (H_vals**2 * a**5)
    friction = -(3 / a + dH_da_vals / H_vals) * d_delta_da
    d2_delta_da = source + friction

    return [d_delta_da, d2_delta_da]


a_span = np.logspace(np.log10(1 / 101), 0, 1000, dtype=np.float64)


def fs8_theory(a, theta):
    sol = solve_ivp(
        growth_ODE,
        t_span=(a_span[0], a_span[-1]),
        y0=(a_span[0], 1.0),
        t_eval=a_span,
        rtol=1e-6,
        atol=1e-8,
        args=(theta,),
    )

    delta, d_delta_da = sol.y
    delta0 = delta[-1]
    sig8 = theta[-1]
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    return (sig8 / delta0) * a * interp_pchip(a, a_span, d_delta_da)


Hz_DMz_fid = np.empty(N_fs8, dtype=np.float64)
for i in range(N_fs8):
    Obh2_fid = 0.022
    zi = z_fs8[i]
    Om_fid = fs8_data["omega_fid"][i]
    H0_fid = fs8_data["H0_fid"][i]
    sig8_fid = fs8_data["s8_fid"][i]
    Och2_fid = Om_fid * (H0_fid / 100) ** 2 - Obh2_fid - Omnuh2
    params_fid = [0.0, H0_fid, Obh2_fid, Och2_fid, 0.0, sig8_fid]
    DM_i = DM_z(np.array([zi]), params_fid)[0]
    Hz_DMz_fid[i] = H_z(zi, params_fid) * DM_i


def chi2_fs8(theta):
    Fap = H_z(z_fs8, theta) * DM_z(z_fs8, theta) / Hz_DMz_fid
    delta_fs8 = fs8_data["fs8"] - fs8_theory(a_fs8, theta) / Fap
    return delta_fs8 @ inv_cov_fs8 @ delta_fs8


@njit
def chi2_sn(theta):
    DM = DM_z(z_cmb, theta)
    delta_sn = mu_values - theory_mu(theta, DM) - mu_corr(theta, DM)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_bao(theta):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    return delta_bao @ inv_cov_bao @ delta_bao


@njit
def chi2_cmb(theta):
    # thetastar only
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(theta[2], theta[3], theta)
    return delta[1] ** 2 / cmb.covariance[1, 1]


def chi_squared(theta):
    return chi2_fs8(theta) + chi2_sn(theta) + chi2_bao(theta) + chi2_cmb(theta)


def log_likelihood(theta):
    return -0.5 * chi_squared(theta)


def q0(Om, w0=-1):
    """Calculate the deceleration parameter at z=0"""
    return Om / 2 + (1.0 + 3 * w0) * (1.0 - Om) / 2


def j0(Om, w0=-1, wa=0):
    """Calculate the jerk parameter at z=0"""
    return 1.0 + (3 / 2) * (1.0 - Om) * (3 * w0 * (1.0 + w0) + wa)


def main():
    from scipy.stats import norm
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions
    from fs8.plot_predictions import plot_predictions as plot_fs8_predictions

    prior = Prior()
    prior.add_parameter("ΔM", dist=(-1.0, 1.0))
    prior.add_parameter("H0", dist=(50.0, 90.0))
    prior.add_parameter("ωb", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("ωc", dist=(0.05, 0.30))
    prior.add_parameter("v", dist=(-8.5, 8.5))
    prior.add_parameter("sig8", dist=(0.5, 1.5))

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
    v_16, v_50, v_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    sig8_16, sig8_50, sig8_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    S8_samples = samples[:, 5] * (Om_samples / 0.3) ** 0.5
    rd_samples = cmb.r_drag(samples[:, 2], Omh2_samples)
    q0_samples = q0(Om_samples)
    j0_samples = j0(Om_samples)

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)
    S8_16, S8_50, S8_84 = quantile(S8_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)
    q0_16, q0_50, q0_84 = quantile(q0_samples, one_sigma_ci, weights=w)
    j0_16, j0_50, j0_84 = quantile(j0_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, H0_50, Obh2_50, Och2_50, v_50, sig8_50]
    degs_freedom = len(bao_data) + len(z_cmb) + N_fs8 - len(best_fit)
    chi2_MAP = chi_squared(samples[np.argmax(log_l)])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"σ8: {sig8_50:.3f} +{(sig8_84 - sig8_50):.3f} -{(sig8_50 - sig8_16):.3f}")
    print(f"S8: {S8_50:.3f} +{(S8_84 - S8_50):.3f} -{(S8_50 - S8_16):.3f}")
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"q0: {q0_50:.3f} +{(q0_84 - q0_50):.3f} -{(q0_50 - q0_16):.3f}")
    print(f"j0: {j0_50:.3f} +{(j0_84 - j0_50):.3f} -{(j0_50 - j0_16):.3f}")
    print(f"Chi2 (MAP): {chi2_MAP:.2f}")
    print(f"Log Evidence: {sampler.log_z:.2f}")
    print(f"Degrees of freedom: {degs_freedom}")

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
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(best_fit, DM_z(z_cmb, best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit, DM_z(z_cmb, best_fit)),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_fs8_predictions(
        lambda z: fs8_theory(1 / (1.0 + z), best_fit),
        data=fs8_data,
        q=H_z(z_fs8, best_fit) * DM_z(z_fs8, best_fit) / Hz_DMz_fid,
    )


if __name__ == "__main__":
    main()


# *********************************
# BAO: DESI DR2 2025
# SN1a: Union3.1 (2026)
# Prior on ωb from BBN (2024)
# fs8 compilation
# ---------------------------------
# Priors:
#
# all models:
# ΔM: U[-1, 1]
# H0: U[50, 90]
# ωb: N(loc=0.02218, scale=0.00055)
# ωc: U[0.05, 0.30]
# sig8: U[0.5, 1.5]
#
# wCDM:
# w0: U[-1.2, -0.6]
#
# wzCDM:
# w0: U[-1, -1/3]
#
# w0waCDM:
# w0: U[-1.5, 0]
# wa: U[-2.5, 1.5]
# w0 + wa < 0 enforced
#
# Velocity step correction:
# v: U[-8.5, 8.5] x 100 km/s
# *********************************


# ----------- Flat ΛCDM -----------
# ΔM: -0.055 +0.012 -0.012 mag
# H0: 68.34 +0.46 -0.46 km/s/Mpc
# ωb: 0.02210 +0.00053 -0.00053
# ωc: 0.1164 +0.0008 -0.0008
# ωm: 0.1391 +0.0011 -0.0011
# Ωm: 0.298 +0.004 -0.004
# σ8: 0.799 +0.017 -0.017
# S8: 0.796 +0.017 -0.017
# r_d: 148.38 +0.70 -0.70 Mpc
# q0: -0.553 +0.007 -0.007
# j0: 1
# Chi2 (MAP): 56.93
# Log Evidence: -47.25
# Degrees of freedom: 86
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.2 inflow z > 0.2 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)
#
# v: -313 +106 -104 km/s
# ΔM: -0.054 +0.012 -0.012 mag
# H0: 68.45 +0.47 -0.46 km/s/Mpc
# ωb: 0.02216 +0.00053 -0.00053
# ωc: 0.1162 +0.0008 -0.0008
# ωm: 0.1390 +0.0011 -0.0011
# Ωm: 0.297 +0.004 -0.004
# σ8: 0.799 +0.017 -0.017
# S8: 0.795 +0.017 -0.017
# r_d: 148.36 +0.70 -0.69 Mpc
# q0: -0.555 +0.007 -0.007
# j0: 1
# Chi2 (MAP): 48.33 (2.93 sigma significance)
# Log Evidence: -44.64 (Δ logZ = 2.61 in favour of v corrections)
# Degrees of freedom: 85
# ---------------------------------


# ----------- Flat wCDM -----------
# ΔM: -0.069 +0.014 -0.014 mag
# H0: 67.05 +0.78 -0.77 km/s/Mpc
# ωb: 0.02229 +0.00053 -0.00054
# ωc: 0.1143 +0.0013 -0.0014
# ωm: 0.1372 +0.0015 -0.0015
# Ωm: 0.305 +0.006 -0.006
# σ8: 0.810 +0.018 -0.018
# S8: 0.817 +0.021 -0.020
# w0: -0.932 +0.033 -0.033
# r_d: 148.73 +0.73 -0.71 Mpc
# q0: -0.472 +0.039 -0.040
# j0: 0.802
# Chi2 (MAP): 52.72 (2.13 sigma significance)
# Log Evidence: -47.32 (Δ logZ = 0.11 against ΛCDM)
# Degrees of freedom: 85
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
#
# ΔM: -0.068 +0.013 -0.013 mag
# H0: 66.52 +0.83 -0.81 km/s/Mpc
# ωb: 0.02228 +0.00053 -0.00053
# ωc: 0.1151 +0.0010 -0.0010
# ωm: 0.1381 +0.0012 -0.0012
# Ωm: 0.312 +0.007 -0.007
# σ8: 0.811 +0.018 -0.017
# S8: 0.827 +0.022 -0.021
# w0: -0.837 +0.061 -0.062
# wa: [derived wa = -1.5 * (1 - w0^2)]
# r_d: 148.51 +0.70 -0.69 Mpc
# q0: -0.364 +0.069 -0.072
# j0: 0.116 +0.302 -0.258
# Chi2 (MAP): 50.41  (2.62 sigma significance)
# Log Evidence: -45.63 (Δ logZ = 1.80 against ΛCDM)
# Degrees of freedom: 85
# ---------------------------------


# ---------- Flat w0waCDM ---------
# TODO: re-run (fs8 compilation updated)
# ΔM: -0.056 +0.014 -0.014 mag
# H0: 66.69 +0.82 -0.81 km/s/Mpc
# ωb: 0.02218 +0.00054 -0.00054
# ωc: 0.1177 +0.0017 -0.0019
# ωm: 0.1405 +0.0018 -0.0019
# Ωm: 0.316 +0.009 -0.009
# σ8: 0.786 +0.015 -0.015
# S8: 0.807 +0.019 -0.019
# w0: -0.784 +0.088 -0.086
# wa: -0.662 +0.340 -0.355
# z_d: 1059.31 +1.23 -1.25
# r_d: 147.96 +0.76 -0.76 Mpc
# z*: 1089.94 +0.76 -0.73
# r*: 145.19 Mpc
# 100 θ*: 1.04088
# q0: -0.305 +0.098 -0.097
# j0: -0.199 +0.503 -0.475
# Chi2 (MAP): 70.30 (1.9 sigma significance)
# Log Evidence: -58.33 (Δ logZ = -0.78 in favour of ΛCDM)
# Degrees of freedom: 91
# ---------------------------------
