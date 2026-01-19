from numba import njit
import numpy as np
import cmb.data_planck_act_compression as cmb
from interpolator import interp_cubic
from y2023union3.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(cov_matrix_bao)
inv_cov_cmb = np.linalg.inv(cmb.covariance[1:, 1:])

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000, dtype=np.float64)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0, wa):
    cubed = (1.0 + z) ** 3
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0=-1.0, wa=0.0):
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
def theory_mu(theta):
    dL = (1.0 + z_sn_vals) * DM_z(z_sn_vals, theta)
    return theta[0] + 25.0 + 5 * np.log10(dL)


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
    return interp_cubic(z, z_grid, cum_dm)


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
    Omh2 = Obh2 + Och2 + Omnuh2
    rd = cmb.r_drag(Obh2, Omh2)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


@njit
def chi2_sn(theta):
    delta_sn = mu_vals - theory_mu(theta)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_bao(theta):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    return delta_bao @ inv_cov_bao @ delta_bao


def chi_squared(theta):
    # Planck + ACT compressed priors for π/θ* and ωb,
    # without the shift parameter R (arXiv:1808.05724v1)
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, *theta[1:])
    chi_cmb = delta_cmb[1:] @ inv_cov_cmb @ delta_cmb[1:]
    return chi2_sn(theta) + chi2_bao(theta) + chi_cmb


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
    prior.add_parameter("ωb", dist=(0.01, 0.03))
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
DESI DR2
SNIa Union3
(θ*, ωb) CMB Planck + ACT compression

Priors:
ΔM U(-1.0, +1.0)
H0 U(50.0, 90.0)
ωb U(0.01, 0.03)
ωc U(0.05, 0.3)

wzCDM:
w0 U(-1.0, -1/3)

wCDM:
w0 U(-1.2, -0.5)

w0waCDM:
w0 U(-1.5, 0.0)
wa U(-3.0, 1.0)
"""

"""
Flat ΛCDM  w(z) = -1
H0: 68.59 +0.30 -0.30 km/s/Mpc
ωb: 0.02249 +0.00011 -0.00011
ωc: 0.1166 +0.0008 -0.0008
ωm: 0.1397 +0.0008 -0.0008
Ωm: 0.297 +0.004 -0.004
w0: -1
wa: 0
z_d: 1059.96 +0.27 -0.26
r_d: 147.87 +0.26 -0.27 Mpc
Chi squared: 39.69
Log Evidence: -36.40
Degs of freedom: 33

===============================

Flat wCDM w(z) = w0
H0: 66.87 +0.78 -0.78 km/s/Mpc
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1140 +0.0014 -0.0014
ωm: 0.1372 +0.0014 -0.0014
Ωm: 0.307 +0.006 -0.006
w0: -0.919 +0.034 -0.034
wa: 0
z_d: 1059.78 +0.28 -0.28
r_d: 148.56 +0.41 -0.41 Mpc
Chi squared: 33.96
Log Evidence: -35.71 (Δ logZ = 0.69 against ΛCDM)
Degs of freedom: 32

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
H0: 66.17 +0.86 -0.83 km/s/Mpc
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1150 +0.0010 -0.0010
ωm: 0.1382 +0.0010 -0.0010
Ωm: 0.316 +0.008 -0.008
w0: -0.799 +0.064 -0.067
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
z_d: 1059.85 +0.27 -0.27
r_d: 148.28 +0.30 -0.30 Mpc
Chi squared: 30.75
Log Evidence: -33.42 (Δ logZ = 2.98 against ΛCDM)
Degs of freedom: 32

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 66.10 +0.84 -0.83 km/s/Mpc
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1179 +0.0018 -0.0019
ωm: 0.1411 +0.0018 -0.0019
Ωm: 0.323 +0.010 -0.009
w0: -0.720 +0.099 -0.095
wa: -0.803 +0.361 -0.384
z_d: 1060.06 +0.29 -0.30
r_d: 147.51 +0.54 -0.48 Mpc
Chi squared: 29.24
Log Evidence: -35.36 (Δ logZ = 1.05 against ΛCDM)
Degs of freedom: 31
"""
