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

inv_cov_fs8 = np.linalg.inv(fs8.cov_mat)
inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(cov_matrix_bao)

z_max = max(np.max(z_cmb), np.max(bao_data["z"]), np.max(fs8_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dx = np.diff(z_grid)


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
    H0, Obh2, Och2, w0, _ = theta[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


cmb.set_HZ(H_z)


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
def theory_mu(theta):
    dL = (1.0 + z_hel) * DM_z(z_cmb, theta)
    return theta[0] + 25.0 + 5 * np.log10(dL)


@njit
def dH_da(z, theta):
    dz = 1e-5
    Hz_plus = H_z(z + dz, theta)
    Hz_minus = H_z(z - dz, theta)
    dH_dz = (Hz_plus - Hz_minus) / (2 * dz)
    return -1 * (1 + z) ** 2 * dH_dz


@njit
def growth_ODE(a, y, *theta):
    H0, Obh2, Och2 = theta[1], theta[2], theta[3]
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2

    z = 1 / a - 1
    H_vals = H_z(z, theta)
    dH_da_vals = dH_da(z, theta)

    delta, d_delta_da = y

    source = 1.5 * Obc * H0**2 * delta / (H_vals**2 * a**5)
    friction = -(3 / a + dH_da_vals / H_vals) * d_delta_da
    d2_delta_da = source + friction

    return [d_delta_da, d2_delta_da]


a_vals = np.logspace(-3.037, 0, 1000, dtype=np.float64)


def fs8_theory(z, theta):
    sol = solve_ivp(
        growth_ODE,
        t_span=(a_vals[0], a_vals[-1]),
        y0=(a_vals[0], 1.0),
        t_eval=a_vals,
        rtol=1e-8,
        atol=1e-10,
        args=theta,
    )

    delta, d_delta_da = sol.y
    delta0 = interp_hermite(np.array([1.0]), a_vals, delta, d_delta_da)[0]
    sig8 = theta[-1]
    a = 1 / (1 + z)

    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)

    return sig8 * a * interp_pchip(a, a_vals, d_delta_da) / delta0


H0_fid = 67.6
Obh2_fid = 0.022
params_fid = [0.0, H0_fid, Obh2_fid, 0.31, -1.0, 0.80]
fiducial_scaling = np.empty(len(fs8_data["z"]), dtype=np.float64)

for i in range(len(fs8_data["z"])):
    zi = fs8_data["z"][i]
    Om_fid = fs8_data["omega_fid"][i]
    Och2_fid = Om_fid * (H0_fid / 100) ** 2 - Obh2_fid - Omnuh2
    params_fid[3] = Och2_fid
    DM_i = DM_z(np.array([zi]), params_fid)[0]
    fiducial_scaling[i] = H_z(zi, params_fid) * DM_i


def chi2_fs8(theta):
    Fap = H_z(fs8_data["z"], theta) * DM_z(fs8_data["z"], theta) / fiducial_scaling
    delta_fs8 = fs8_data["fs8"] - fs8_theory(fs8_data["z"], theta) / Fap
    return delta_fs8 @ inv_cov_fs8 @ delta_fs8


@njit
def chi2_sn(theta):
    delta_sn = mu_values - theory_mu(theta)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_bao(theta):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    return delta_bao @ inv_cov_bao @ delta_bao


def chi_squared(theta):
    delta_lA = cmb.DISTANCE_PRIORS[1] - cmb.cmb_distances(theta[2], theta[3], theta)[1]
    chi2_lA = delta_lA**2 / cmb.covariance[1, 1]
    return chi2_fs8(theta) + chi2_sn(theta) + chi2_bao(theta) + chi2_lA


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

    prior = Prior()
    prior.add_parameter("ΔM", dist=(-1.0, 1.0))
    prior.add_parameter("H0", dist=(50.0, 90.0))
    prior.add_parameter("ωb", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("ωc", dist=(0.05, 0.30))
    prior.add_parameter("w0", dist=(-1.0, -1 / 3))
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
    w0_16, w0_50, w0_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    sig8_16, sig8_50, sig8_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)

    wa_samples = -1.5 * (1 - samples[:, 4] ** 2)
    wa_16, wa_50, wa_84 = quantile(wa_samples, one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    S8_samples = samples[:, 5] * (Om_samples / 0.3) ** 0.5
    rd_samples = cmb.r_drag(samples[:, 2], Omh2_samples)
    zd_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    zst_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    q0_samples = q0(Om_samples, w0=samples[:, 4])
    j0_samples = j0(Om_samples, w0=samples[:, 4], wa=wa_samples)

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)
    S8_16, S8_50, S8_84 = quantile(S8_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)
    zd_16, zd_50, zd_84 = quantile(zd_samples, one_sigma_ci, weights=w)
    zst_16, zst_50, zst_84 = quantile(zst_samples, one_sigma_ci, weights=w)
    q0_16, q0_50, q0_84 = quantile(q0_samples, one_sigma_ci, weights=w)
    j0_16, j0_50, j0_84 = quantile(j0_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, H0_50, Obh2_50, Och2_50, w0_50, sig8_50]
    degs_freedom = len(bao_data["z"]) + len(z_cmb) + len(fs8_data["z"]) - len(best_fit)
    chi2_MAP = chi_squared(samples[np.argmax(log_l)])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"σ8: {sig8_50:.3f} +{(sig8_84 - sig8_50):.3f} -{(sig8_50 - sig8_16):.3f}")
    print(f"S8: {S8_50:.3f} +{(S8_84 - S8_50):.3f} -{(S8_50 - S8_16):.3f}")
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
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )

    z_fs8_smoot = np.linspace(0, np.max(fs8_data["z"]), 200)
    Fap = (
        H_z(fs8_data["z"], best_fit) * DM_z(fs8_data["z"], best_fit) / fiducial_scaling
    )

    plt.errorbar(
        fs8_data["z"],
        Fap * fs8_data["fs8"],
        yerr=Fap * fs8_data["fs8_err"],
        fmt=".",
        label="data",
    )
    plt.plot(z_fs8_smoot, fs8_theory(z_fs8_smoot, best_fit), label="best-fit")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


"""
BAO: DESI DR2 2025
Prior on ωb from BBN (Y2024)
SN1a: Union3.1 2026

Priors:

all models:
ΔM: U(-1.0, 1.0)
H0: U(50, 90)
ωb: N(loc=0.02218, scale=0.00055)
ωc: U(0.05, 0.30)
sig8: U(0.5, 1.5)

wCDM:
w0: U(-1.2, -0.6)

wzCDM
w0: U(-1.0, -1/3)

w0waCDM:
w0: U(-1.5, 0.0)
wa: U(-3.0, 1.0)
w0 + wa < 0 enforced

M(z):
p: U(-0.4, 1.0)
"""

"""
Flat ΛCDM w(z) = -1
ΔM: -0.053 +0.012 -0.012 mag
H0: 68.42 +0.46 -0.47 km/s/Mpc
ωb: 0.02214 +0.00054 -0.00053
ωc: 0.1162 +0.0008 -0.0008
ωm: 0.1390 +0.0011 -0.0011
Ωm: 0.297 +0.004 -0.004
σ8: 0.779 +0.014 -0.014
S8: 0.775 +0.014 -0.014
z_d: 1059.12 +1.25 -1.24
r_d: 148.37 +0.69 -0.71 Mpc
z*: 1089.87 +0.70 -0.68
r*: 145.60 Mpc
100 θ*: 1.04096
q0: -0.554 +0.007 -0.007
j0: 1
Chi2 (MAP): 77.45
Log Evidence: -57.55
Degrees of freedom: 93
"""

"""
Flat ΛCDM w(z) = -1, evolving absolute magnitude
M(z) = ΔM_inf + 1 - (z / (1 + z))^(0.1 * p)

ΔM: -0.083 +0.018 -0.018 mag
p: 0.270 +0.126 -0.120
H0: 68.57 +0.46 -0.46 km/s/Mpc
ωb: 0.02222 +0.00052 -0.00053
ωc: 0.1160 +0.0008 -0.0008
ωm: 0.1388 +0.0011 -0.0011
Ωm: 0.295 +0.004 -0.004
σ8: 0.780 +0.014 -0.014
S8: 0.774 +0.014 -0.014
z_d: 1059.28 +1.20 -1.24
r_d: 148.35 +0.70 -0.69 Mpc
z*: 1089.75 +0.70 -0.66
r*: 145.61 Mpc
100 θ*: 1.04095
q0: -0.557 +0.007 -0.007
j0: 1
Chi2 (MAP): 72.43 (2.24 sigma away from no evolution in magnitude)
Log Evidence: -56.51 (Δ logZ = 1.04 against no evolution in magnitude)
Degrees of freedom: 92
"""

"""
Flat wCDM w(z) = w0
ΔM: -0.066 +0.014 -0.014 mag
H0: 67.25 +0.78 -0.75 km/s/Mpc
ωb: 0.02231 +0.00053 -0.00054
ωc: 0.1144 +0.0013 -0.0014
ωm: 0.1373 +0.0014 -0.0015
Ωm: 0.304 +0.006 -0.006
σ8: 0.790 +0.015 -0.015
S8: 0.795 +0.018 -0.018
w0: -0.939 +0.032 -0.033
z_d: 1059.37 +1.21 -1.24
r_d: 148.69 +0.72 -0.72 Mpc
z*: 1089.47 +0.72 -0.69
r*: 145.97 Mpc
100 θ*: 1.04090
q0: -0.481 +0.039 -0.040
j0: 0.821 +0.093 -0.084
Chi2 (MAP): 72.95 (2.12 sigma away from ΛCDM)
Log Evidence: -57.79 (Δ logZ = -0.24 in favour of ΛCDM)
Degrees of freedom: 92
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
ΔM: -0.065 +0.013 -0.013 mag
H0: 66.75 +0.80 -0.80 km/s/Mpc
ωb: 0.02231 +0.00053 -0.00053
ωc: 0.1151 +0.0010 -0.0010
ωm: 0.1380 +0.0012 -0.0012
Ωm: 0.310 +0.007 -0.007
σ8: 0.791 +0.015 -0.015
S8: 0.804 +0.019 -0.019
w0: -0.850 +0.060 -0.061
wa: -0.416 +0.160 -0.148 [derived wa = -1.5 * (1 - w0^2)]
z_d: 1059.42 +1.22 -1.23
r_d: 148.49 +0.69 -0.69 Mpc
z*: 1089.54 +0.70 -0.67
r*: 145.78 Mpc
100 θ*: 1.04092
q0: -0.380 +0.069 -0.070
j0: 0.174 +0.304 -0.264
Chi2 (MAP): 70.96 (2.55 sigma away from ΛCDM)
Log Evidence: -56.16 (Δ logZ = 1.39 against ΛCDM)
Degrees of freedom: 92
"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
"""
