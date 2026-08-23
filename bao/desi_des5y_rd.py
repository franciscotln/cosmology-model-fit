from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2025DESdovekie.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    inv_a3 = (1.0 + z) ** 3
    # thawing quintessence
    return (2 * inv_a3 / (1.0 + w0 + (1.0 - w0) * inv_a3)) ** 2


@njit
def H_z(z, params):
    H0, Om = params[2], params[3]
    Ol = 1.0 - Om
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + Ol)


@njit
def DM_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    dm_grid = np.zeros(z_grid.size, dtype=np.float64)
    dm_grid[1:] = np.cumsum(dz * dh)
    return (dm_grid, dh_grid)


@njit
def DM_z(z, dm_interp):
    return interp_hermite(z, z_grid, *dm_interp)


@njit
def DH_z(z, dm_interp):
    return interp_pchip(z, z_grid, dm_interp[1])


@njit
def DV_z(z, DH, DM):
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params, dm_interp):
    DM = DM_z(z, dm_interp)
    DH = DH_z(z, dm_interp)
    rd = params[1]
    results = np.empty(z.size, dtype=np.float64)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2

    results[DH_mask] = DH[DH_mask] / rd
    results[DM_mask] = DM[DM_mask] / rd
    results[DV_mask] = DV_z(z[DV_mask], DH[DV_mask], DM[DV_mask]) / rd
    return results


@njit
def mu_corr(params, dm_interp):
    # Heaviside step at z = 0.10563
    v_km_s = 100 * params[4] * np.where(z_cmb <= 0.10563, 1, -1)
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + v_km_s / c)
    return 5.0 * np.log10(DM_z(z_cosmo, dm_interp) / DM_z(z_cmb, dm_interp))


@njit
def mu_theory(params, dm_interp):
    return params[0] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM_z(z_cmb, dm_interp))


@njit
def chi_squared(params):
    dm_interp = DM_grid(params)

    delta_sn = mu_vals - mu_theory(params, dm_interp) - mu_corr(params, dm_interp)
    chi_sn = solve_triangular(cho_sn, delta_sn)

    delta_bao = bao["value"] - bao_theory(bao["z"], bao_qty, params, dm_interp)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao
    return chi_sn + chi_bao


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("ΔM", dist=(-0.4, +0.4))
    prior.add_parameter("rd", dist=norm(loc=147.09, scale=0.26))  # Planck prior
    prior.add_parameter("H0", dist=(50.0, 85.0))
    prior.add_parameter("Ωm", dist=(0.1, 0.6))
    prior.add_parameter("v", dist=(-4.5, 4.5))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    one_sigma_ci = [0.159, 0.5, 0.841]

    dM_16, dM_50, dM_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    H0_16, H0_50, H0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    v_16, v_50, v_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 3] * (samples[:, 2] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, rd_50, H0_50, Om_50, v_50]

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

    degs_of_freedom = len(bao) + sn_size - len(best_fit)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"Degs of freedom: {degs_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(
            z, qty, best_fit, DM_grid(best_fit)
        ),
        data=bao,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit, DM_grid(best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit, DM_grid(best_fit)),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

# *********************************
# DESI BAO DR2 2025
# DESY5 Dovekie
# rdrag prior from Planck 2018
# ---------------------------------
#
# Priors:
# ΔM ~U[-0.4, +0.4]
# rd ~N(147.09, 0.26)
# H0 ~U[50.0, 85.0]
# Ωm ~U[0.1, 0.6]
#
# Velocity-like step correction at z = 0.10563:
# v ~U[-4.5, 4.5] x 100 km/s
#
# wCDM:
# w0 ~U[-1.5, 0.0]
#
# wzCDM thawing quintessence:
# w0 ~U[-1.0, -1/3]
#
# w0waCDM:
# w0 ~U[-1.5, 0.0]
# wa ~U[-3.5, 2.5]
#
# w0 + wa < 0 enforced
# *********************************


# ----------- Flat ΛCDM -----------
# ΔM: -0.052 +0.012 -0.012 mag
# rd: 147.09 +0.26 -0.26 Mpc
# H0: 68.57 +0.46 -0.46 km/s/Mpc
# Ωm: 0.306 +0.008 -0.008
# ωm: 0.1440 +0.0022 -0.0021
# Chi squared: 1645.3
# Log evidence: -834.4
# Degs of freedom: 1723
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity-like step correction in observed redshifts
# turning point z <= 0.11 inflow z > 0.11 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)
#
# v: -1.589 +0.576 -0.574 x 100 km/s
# ΔM: -0.047 +0.012 -0.012 mag
# rd: 147.09 +0.26 -0.26 Mpc
# H0: 68.91 +0.48 -0.47 km/s/Mpc
# Ωm: 0.300 +0.008 -0.008
# ωm: 0.1423 +0.0022 -0.0022
# Chi squared: 1637.4 (2.81 sigma significance)
# Log evidence: -832.3 (Δ logZ = 2.1 in favour of step correction)
# Degs of freedom: 1722
# ---------------------------------


# ----------- Flat wCDM -----------
# ΔM: -0.057 +0.012 -0.012 mag
# rd: 147.09 +0.26 -0.26 Mpc
# H0: 67.78 +0.56 -0.55 km/s/Mpc
# Ωm: 0.297 +0.009 -0.009
# ωm: 0.1366 +0.0038 -0.0039
# w0: -0.908 +0.037 -0.038
# Chi squared: 1639.5 (2.41 sigma significance)
# Log evidence: -834.3 (Δ logZ = 0.1 against ΛCDM)
# Degs of freedom: 1722
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
#
# ΔM: -0.055 +0.012 -0.012 mag
# rd: 147.09 +0.26 -0.26 Mpc
# H0: 67.62 +0.57 -0.56 km/s/Mpc
# Ωm: 0.305 +0.008 -0.008
# ωm: 0.1394 +0.0027 -0.0027
# w0: -0.861 +0.051 -0.052
# Chi squared: 1638.5 (2.61 sigma significance)
# Log evidence: -832.6 (Δ logZ = 1.8 against ΛCDM)
# Degs of freedom: 1722
# ---------------------------------


# ----------- Flat w0waCDM --------
# ΔM: -0.053 +0.012 -0.012 mag
# rd: 147.09 +0.26 -0.26 Mpc
# H0: 67.60 +0.58 -0.58 km/s/Mpc
# Ωm: 0.313 +0.014 -0.017
# ωm: 0.143 +0.005 -0.007
# w0: -0.85 +0.07 -0.07
# wa: -0.52 +0.48 -0.47
# Chi squared: 1638.1 (2.21 sigma away from ΛCDM)
# Log evidence: -835.3 (Δ logZ = -0.9 in favour of ΛCDM)
# Degs of freedom: 1721
# TODO: remove forbidden volume on w0-wa plane, still ΛCDM is preferred
# ---------------------------------
