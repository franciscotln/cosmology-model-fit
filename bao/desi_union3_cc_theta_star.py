from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_planck_act_compression as cmb
from y2023union3.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = c0 / 1000  # km/s
Or_h2 = cmb.Omega_r_h2(2.044)
Omnu_h2 = cmb.Omnu_h2
z_nr = cmb.z_nr

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=2000)
dx = np.diff(z_grid)


@njit
def Omnu_z(z):
    """
    Computes the appox. evolution of one massive
    neutrino species energy density with redshift
    """
    return (
        (1 + z) ** 4
        * (1 + ((1 + z_nr) / (1 + z)) ** 2) ** 0.5
        * (1 + (1 + z_nr) ** 2) ** -0.5
    )


@njit
def Ode_z(z, w0=-1, wa=0):
    zp1 = 1 + z
    # return 1  # ΛCDM
    # return zp1 ** (3 * (1 + w0))  # wCDM
    return (4 * zp1**3 / (1 + 3 * zp1**3)) ** (4 * (1 + w0))  # wzCDM
    # return zp1 ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / zp1)  # w0waCDM


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0, wa)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, theta):
    H0, Obh2, Och2, w0 = theta[2:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


@njit
def DH_z(z, theta):
    return c / H_z(z, theta)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, theta):
    Obh2, Och2 = theta[3], theta[4]
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnu_h2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


@njit
def mu_theory(theta):
    dL = (1 + z_sn_vals) * DM_z(z_sn_vals, theta)
    return theta[1] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    delta = (cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, *theta[2:]))[1]
    thetastar_err = cmb.covariance[1, 1] ** 0.5
    chi_theta_star = (delta / thetastar_err) ** 2

    delta_sn = mu_values - mu_theory(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = solve_triang(cho_bao, delta_bao)

    delta_cc = H_cc_vals - H_z(z_cc_vals, theta)
    chi_cc = solve_triang(cho_cc, delta_cc) * theta[0] ** 2

    return chi_theta_star + chi_sn + chi_bao + chi_cc


def log_likelihood(theta):
    f_cc = theta[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(theta) - 0.5 * normalization_cc


def main():
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from cosmic_chronometers.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    # f_cc: CC error rescaling (overestimated)
    prior.add_parameter("f_cc", dist=(0.2, 3.0))
    # ΔM: magnitude offset
    prior.add_parameter("ΔM", dist=(-1.0, 1.0))
    # H0: Hubble constant at present
    prior.add_parameter("H0", dist=(50.0, 85.0))
    # Ωb x h^2: baryon density parameter
    prior.add_parameter("ωb", dist=(0.003, 0.050))
    # Ωc x h^2: cold dark matter density param today
    prior.add_parameter("ωc", dist=(0.05, 0.30))
    # w0: dark energy equation of state today
    prior.add_parameter("w0", dist=(-1.5, 0.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior,
            log_likelihood,
            n_live=10_000,
            pool=pool,
            seed=42,
            pass_dict=False,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    one_sigma_ci = [0.159, 0.5, 0.841]
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

    fcc_16, fcc_50, fcc_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    h0_16, h0_50, h0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    wb_16, wb_50, wb_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    wc_16, wc_50, wc_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)

    best_fit = [fcc_50, dM_50, h0_50, wb_50, wc_50, w0_50]

    deg_of_freedom = (
        1 + len(z_sn_vals) + len(bao_data["z"]) + len(z_cc_vals) - len(prior.keys)
    )

    Omh2_samples = samples[:, 3] + samples[:, 4] + Omnu_h2
    Om_samples = Omh2_samples / (samples[:, 2] / 100) ** 2
    r_d_samples = cmb.r_drag(samples[:, 3], Omh2_samples)
    rd_16, rd_50, rd_84 = quantile(r_d_samples, one_sigma_ci, weights=w)
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)
    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"ωb: {wb_50:.4f} +{(wb_84 - wb_50):.4f} -{(wb_50 - wb_16):.4f} Mpc")
    print(f"ωc: {wc_50:.4f} +{(wc_84 - wc_50):.4f} -{(wc_50 - wc_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / fcc_50,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=rf"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
f_cc: 1.49 +0.19 -0.18
ΔM: -0.150 +0.100 -0.098 mag
H0: 67.7 +1.5 -1.3 km/s/Mpc
ωb: 0.0212 +0.0021 -0.0019 Mpc
ωc: 0.1162 +0.0011 -0.0010
ωm: 0.1380 +0.0030 -0.0026
Ωm: 0.301 +0.008 -0.007
w0: -1
wa: 0
r_d: 149.5 +2.4 -2.6 Mpc
Chi squared: 72.40
Log evidence: -166.49
Degrees of freedom: 64

===============================

Flat wCDM: w(z) = w0
f_cc: 1.47 +0.19 -0.18
ΔM: -0.113 +0.103 -0.102 mag
H0: 68.5 +1.7 -1.5 km/s/Mpc
ωb: 0.0260 +0.0032 -0.0030 Mpc
ωc: 0.1146 +0.0017 -0.0016
ωm: 0.1412 +0.0043 -0.0037
Ωm: 0.301 +0.007 -0.007
w0: -0.888 +0.041 -0.041 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 144.6 +3.3 -3.5 Mpc
Chi squared: 65.38
Log evidence: -165.78 (Δ logZ = 0.71 against ΛCDM)
Degrees of freedom: 63

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)
f_cc: 1.47 +0.18 -0.17
ΔM: -0.140 +0.101 -0.100 mag
H0: 67.3 +1.6 -1.5 km/s/Mpc
ωb: 0.0247 +0.0027 -0.0025 Mpc
ωc: 0.1159 +0.0016 -0.0013
ωm: 0.1412 +0.0041 -0.0034
Ωm: 0.312 +0.009 -0.008
w0: -0.789 +0.070 -0.070 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
r_d: 145.6 +2.9 -3.2 Mpc
Chi squared: 63.27
Log evidence: -164.24 (Δ logZ = 2.25 against ΛCDM)
Degrees of freedom: 63

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
f_cc: 1.46 +0.18 -0.17
ΔM: -0.179 +0.107 -0.105 mag
H0: 65.9 +2.1 -1.8 km/s/Mpc
ωb: 0.0220 +0.0036 -0.0030 Mpc
ωc: 0.1179 +0.0018 -0.0021
ωm: 0.1402 +0.0037 -0.0029
Ωm: 0.323 +0.015 -0.015
w0: -0.739 +0.107 -0.100 (prior width 1.5: -1.5 to 0.0)
wa: -0.750 +0.457 -0.499 (prior width 5.5: -3.5 to 2.0)
r_d: 148.2 +3.4 -3.8 Mpc
Chi squared: 62.73
Log evidence: -166.02 (Δ logZ = 0.47 against ΛCDM)
Degrees of freedom: 62
"""
