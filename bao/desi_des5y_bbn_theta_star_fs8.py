from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from scipy.integrate import solve_ivp
from interpolator import interp_hermite, interp_pchip
import cmb.data_planck_act_compression as cmb
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2025DESdovekie.data import get_data as get_sn_data, effective_sample_size
from y2025BAO.data import get_data as get_bao_data
import y2018fs8.data as fs8

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

fs8_data = fs8.data

inv_cov_fs8 = np.linalg.inv(fs8.cov_mat)
inv_cov_bao = np.linalg.inv(cov_matrix_bao)
cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao_data["z"]), np.max(fs8_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2


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
desi_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


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


pivot_mask = z_cmb <= 0.11


@njit
def mu_corr(params, DM_obs):
    z_pec = 100 * params[4] / c
    z_cosmo1 = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    z_cosmo2 = -1.0 + (1.0 + z_cmb) / (1.0 - z_pec)

    return np.where(
        pivot_mask,
        5.0 * np.log10(DM_z(z_cosmo1, params) / DM_obs),
        5.0 * np.log10(DM_z(z_cosmo2, params) / DM_obs),
    )


@njit
def theory_mu(theta, DM):
    return theta[0] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


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

    z = 1 / a - 1.0
    H_vals = H_z(z, theta)
    dH_da_vals = dH_da(z, theta)

    delta, d_delta_da = y

    source = 1.5 * Obc * H0**2 * delta / (H_vals**2 * a**5)
    friction = -(3 / a + dH_da_vals / H_vals) * d_delta_da
    d2_delta_da = source + friction

    return [d_delta_da, d2_delta_da]


a_vals = np.logspace(-3.2, 0, 1500, dtype=np.float64)


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
    a = 1 / (1.0 + z)

    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)

    return sig8 * a * interp_pchip(a, a_vals, d_delta_da) / delta0


H0_fid = 67.6
Obh2_fid = 0.022
params_fid = [0.0, H0_fid, Obh2_fid, 0.31, 0.0, 0.80]
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


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return y @ y


def chi2_sn(theta):
    DM = DM_z(z_cmb, theta)
    delta_sn = mu_values - theory_mu(theta, DM) - mu_corr(theta, DM)
    return solve_triang(cho_sn, delta_sn)


@njit
def chi2_bao(theta):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, theta)
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
    prior.add_parameter("ΔM", dist=(-0.5, 0.5))
    prior.add_parameter("H0", dist=(50.0, 90.0))
    prior.add_parameter("ωb", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("ωc", dist=(0.05, 0.30))
    prior.add_parameter("v", dist=(-6.0, 2.0))
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
    degs_freedom = len(bao_data) + effective_sample_size + len(fs8_data) - len(best_fit)
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
    plt.plot(z_fs8_smoot, fs8_theory(z_fs8_smoot, best_fit), label="Growth theory")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


"""
BAO: DESI DR2 2025
SN1a: DES5Y Dovekie (2025)
Prior on ωb from BBN (2024)
fs8 compilation

Priors:

all models:
ΔM: U(-0.5, 0.5) mag
H0: U(50, 90) km/s/Mpc
ωb: N(loc=0.02218, scale=0.00055)
ωc: U(0.05, 0.30)
sig8: U(0.5, 1.5)

wCDM:
w0: U(-1.2, -0.6)

wzCDM:
w0: U(-1.0, -1/3)

w0waCDM:
w0: U(-1.5, 0.0)
wa: U(-2.5, 1.5)
w0 + wa < 0 enforced

flow correction:
v: U(-6, 2) x 100 km/s
"""

"""
Flat ΛCDM

"""

"""
Flat ΛCDM
Isotropic velocity SNe observed redshifts (turning point z <= 0.11 inflow z > 0.11 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

v: -1.66 +0.55 -0.56 x 100 km/s
ΔM: -0.063 +0.013 -0.013 mag
H0: 68.48 +0.46 -0.45 km/s/Mpc
ωb: 0.02217 +0.00053 -0.00053
ωc: 0.1161 +0.0008 -0.0008
ωm: 0.1390 +0.0011 -0.0011
Ωm: 0.296 +0.004 -0.004
σ8: 0.779 +0.014 -0.014
S8: 0.775 +0.014 -0.014
r_d: 148.36 +0.70 -0.70 Mpc
q0: -0.555 +0.007 -0.006
j0: 1
Chi2 (MAP): 1673.65
Log Evidence: -856.51 (Δ logZ = x.xx in favour of corrections)
Degrees of freedom: 1784
"""

"""
Flat wCDM w(z) = w0

"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)

"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)

"""
