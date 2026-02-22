from numba import njit
import numpy as np
import cmb.data_planck_act_compression as cmb
from interpolator import interp_hermite
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2026union3_1.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data
from y2024DESBAO.data import get_data as get_bao_des_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
bao_des_legend, bao_des_data, cov_matrix_bao_des = get_bao_des_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(cov_matrix_bao)
inv_cov_bao_des = np.linalg.inv(cov_matrix_bao_des)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0):
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0)

    return np.sqrt(radiation_term + matter_term + neutrino_term + dark_energy_term)


@njit
def H_z(z, theta):
    H0, Obh2, Och2, w0 = theta[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


cmb.set_HZ(H_z)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
qty_desi = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)
qty_des = np.array([qty_map[q] for q in bao_des_data["quantity"]], dtype=np.int64)


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
def theory_mu(theta):
    dL = (1.0 + z_hel) * DM_z(z_cmb, theta)
    return theta[0] + 25.0 + 5 * np.log10(dL)


@njit
def chi2_sn(theta):
    delta_sn = mu_values - theory_mu(theta)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_bao(theta):
    delta_bao_des = bao_des_data["value"] - bao_theory(
        bao_des_data["z"], qty_des, theta
    )
    chi2_bao_des = delta_bao_des @ inv_cov_bao_des @ delta_bao_des
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], qty_desi, theta)
    chi2_bao = delta_bao @ inv_cov_bao @ delta_bao
    return chi2_bao + chi2_bao_des


def chi_squared(theta):
    delta_lA = cmb.DISTANCE_PRIORS[1] - cmb.cmb_distances(theta[2], theta[3], theta)[1]
    chi2_lA = delta_lA**2 / cmb.covariance[1, 1]
    return chi2_sn(theta) + chi2_bao(theta) + chi2_lA


def log_likelihood(theta):
    return -0.5 * chi_squared(theta)


def q0(Om, w0=-1):
    """Calculate the deceleration parameter at z=0."""
    return Om / 2 + (1 + 3 * w0) * (1 - Om) / 2


def j0(Om, w0=-1, wa=0):
    """Calculate the jerk parameter at z=0."""
    return 1 + (3 / 2) * (1 - Om) * (3 * w0 * (1 + w0) + wa)


def main():
    from scipy.stats import norm
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("ΔM", dist=(-1.0, 1.0))
    prior.add_parameter("H0", dist=(50, 90))
    prior.add_parameter("ωb", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("ωc", dist=(0.05, 0.30))
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

    wa_samples = -1.5 * (1 - samples[:, 4] ** 2)
    wa_16, wa_50, wa_84 = quantile(wa_samples, one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    rd_samples = cmb.r_drag(samples[:, 2], Omh2_samples)
    zd_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    zst_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    q0_samples = q0(Om_samples, w0=samples[:, 4])
    j0_samples = j0(Om_samples, w0=samples[:, 4], wa=wa_samples)

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)
    zd_16, zd_50, zd_84 = quantile(zd_samples, one_sigma_ci, weights=w)
    zst_16, zst_50, zst_84 = quantile(zst_samples, one_sigma_ci, weights=w)
    q0_16, q0_50, q0_84 = quantile(q0_samples, one_sigma_ci, weights=w)
    j0_16, j0_50, j0_84 = quantile(j0_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, H0_50, Obh2_50, Och2_50, w0_50]
    degrees_of_freedom = (
        1 + len(bao_des_data["z"]) + len(bao_data["z"]) + len(z_cmb) - len(best_fit)
    )

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"wa: {wa_50:.3f} +{(wa_84 - wa_50):.3f} -{(wa_50 - wa_16):.3f}")
    print(f"z_d: {zd_50:.2f} +{(zd_84 - zd_50):.2f} -{(zd_50 - zd_16):.2f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"z*: {zst_50:.2f} +{(zst_84 - zst_50):.2f} -{(zst_50 - zst_16):.2f}")
    print(f"r*: {cmb.rs_z(zst_50, Obh2_50, best_fit):.2f} Mpc")
    print(
        f"100 θ*: {100 * np.pi / cmb.cmb_distances(Obh2_50, Och2_50, best_fit)[1]:.5f}"
    )
    print(f"q0: {q0_50:.3f} +{(q0_84 - q0_50):.3f} -{(q0_50 - q0_16):.3f}")
    print(f"j0: {j0_50:.3f} +{(j0_84 - j0_50):.3f} -{(j0_50 - j0_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(samples[np.argmax(log_l)]):.2f}")
    print(f"Log Evidence: {sampler.log_z:.2f}")
    print(f"Degrees of freedom: {degrees_of_freedom}")

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
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_des_data,
        errors=np.sqrt(np.diag(cov_matrix_bao_des)),
        title=bao_des_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
BAO: DESI DR2 + DES BAO 2025
Prior on ωb from BBN (Y2024)
SN1a: Union3.1 compilation

Priors:

all models:
ΔM: U(-1.0, 1.0)
H0: U(50, 90)
ωb: N(loc=0.02218, scale=0.00055)
ωc: U(0.05, 0.30)

wCDM:
w0: U(-1.2, -0.6)

wzCDM
w0: U(-1.0, -1/3)

w0waCDM:
w0: U(-1.5, 0.0)
wa: U(-3.0, 1.0)
w0 + wa < 0 enforced

Evolving absolute magnitude M(z):
M'0: U(-8.0, 4.0)
"""

"""
Flat ΛCDM  w(z) = -1
ΔM: -0.054 +0.013 -0.012 mag
H0: 68.38 +0.47 -0.46 km/s/Mpc
ωb: 0.02211 +0.00053 -0.00054
ωc: 0.1163 +0.0008 -0.0008
ωm: 0.1390 +0.0011 -0.0011
Ωm: 0.297 +0.005 -0.004
z_d: 1059.07 +1.21 -1.26
r_d: 148.39 +0.70 -0.69 Mpc
z*: 1089.90 +0.71 -0.67
r*: 145.61 Mpc
100 θ*: 1.04094
q0: -0.554 +0.007 -0.007
j0: 1
Chi2 (MAP): 41.70
Log Evidence: -36.41
Degrees of freedom: 33

===============================

Flat ΛCDM  w(z) = -1
Corrections to absolute magnitude of SNe M(z) = M0 + M0' * z / (1 + (z / z_c))
z_c = 0.0557 is the redshift of a kind of homogeneity scale
M(z << z_c) = M0 + M0' * z, low z value
M(z >> z_c) = M0 + M0' * z_c, asymptotic value

ΔM: 0.054 +0.048 -0.048 mag
M'0: -2.206 +0.941 -0.940 mag / unity redshift
H0: 68.53 +0.46 -0.47 km/s/Mpc
ωb: 0.02218 +0.00054 -0.00053
ωc: 0.1160 +0.0008 -0.0008
ωm: 0.1389 +0.0011 -0.0011
Ωm: 0.296 +0.004 -0.004
z_d: 1059.21 +1.23 -1.25
r_d: 148.37 +0.71 -0.70 Mpc
z*: 1089.79 +0.70 -0.68
r*: 145.62 Mpc
100 θ*: 1.04095
q0: -0.556 +0.007 -0.007
j0: 1
Chi2 (MAP): 36.47 (2.29 sigma away from constant M)
Log Evidence: -35.27 (Δ logZ 1.14 against constant M)
Degrees of freedom: 32

===============================

Flat wCDM w(z) = w0
ΔM: -0.067 +0.014 -0.014 mag
H0: 67.20 +0.80 -0.79 km/s/Mpc
ωb: 0.02227 +0.00054 -0.00054
ωc: 0.1144 +0.0013 -0.0014
ωm: 0.1373 +0.0015 -0.0015
Ωm: 0.304 +0.006 -0.006
w0: -0.938 +0.033 -0.034
z_d: 1059.29 +1.23 -1.25
r_d: 148.72 +0.73 -0.71 Mpc
z*: 1089.52 +0.72 -0.71
r*: 145.99 Mpc
100 θ*: 1.04093
q0: -0.480 +0.041 -0.042
j0: 0.819 +0.097 -0.087
Chi2 (MAP): 37.88 (1.95 sigma away from ΛCDM)
Log Evidence: -36.73 (Δ logZ = -0.32 in favour of ΛCDM)
Degrees of freedom: 32

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
ΔM: -0.066 +0.013 -0.013 mag
H0: 66.68 +0.84 -0.83 km/s/Mpc
ωb: 0.02227 +0.00053 -0.00053
ωc: 0.1151 +0.0010 -0.0010
ωm: 0.1381 +0.0012 -0.0012
Ωm: 0.311 +0.007 -0.007
w0: -0.849 +0.063 -0.063
wa: -0.420 +0.167 -0.153 [derived wa = -1.5 * (1 - w0^2)]
z_d: 1059.33 +1.22 -1.24
r_d: 148.52 +0.70 -0.69 Mpc
z*: 1089.60 +0.70 -0.68
r*: 145.80 Mpc
100 θ*: 1.04092
q0: -0.378 +0.072 -0.073
j0: 0.167 +0.317 -0.271
Chi2 (MAP): 36.01 (2.39 sigma away from ΛCDM)
Log Evidence: -35.18 (Δ logZ = 1.23 against ΛCDM)
Degrees of freedom: 32

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
ΔM: -0.058 +0.015 -0.015 mag
H0: 66.62 +0.84 -0.83 km/s/Mpc
ωb: 0.02215 +0.00055 -0.00054
ωc: 0.1176 +0.0018 -0.0020
ωm: 0.1404 +0.0018 -0.0020
Ωm: 0.316 +0.009 -0.009
w0: -0.785 +0.092 -0.089
wa: -0.647 +0.347 -0.372
z_d: 1059.24 +1.24 -1.25
r_d: 148.00 +0.79 -0.77 Mpc
z*: 1089.98 +0.76 -0.74
r*: 145.23 Mpc
100 θ*: 1.04092
q0: -0.305 +0.103 -0.101
j0: -0.182 +0.518 -0.493
Chi2 (MAP): 35.22 (2.06 sigma away from ΛCDM)
Log Evidence: -37.44 + 0.09 = -37.35 (Δ logZ = -1.97 in favour of ΛCDM)
Degrees of freedom: 31
"""
