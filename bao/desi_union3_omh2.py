from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2026union3_1.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_vals, sn_cov_matrix = get_sn_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

inv_cov_sn = np.linalg.inv(sn_cov_matrix)
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    cubic = (1.0 + z) ** 3
    return (2 * cubic / (1.0 + w0 + (1.0 - w0) * cubic)) ** 2


@njit
def H_z(z, params):
    H0, Omh2 = params[2], params[3]
    Om = Omh2 / (H0 / 100) ** 2
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
desi_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[1]


pivot_mask = z_cmb <= 0.2


@njit
def mu_corr(params, DM_ref):
    v_km_s = 100 * params[4] * np.where(pivot_mask, 1.0, -1.0)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)

    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_ref)


@njit
def mu_theory(params, DM):
    return params[0] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params):
    DM = DM_z(z_cmb, params)
    delta_sn = mu_vals - mu_theory(params, DM) - mu_corr(params, DM)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, params)
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
    prior.add_parameter("ΔM", dist=(-1.0, +1.0))
    prior.add_parameter("rd", dist=(120, 160))
    prior.add_parameter("H0", dist=(50.0, 85.0))
    prior.add_parameter("ωm", dist=norm(loc=0.1430, scale=0.0011))  # Planck prior
    prior.add_parameter("v", dist=(-12.0, 5.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior,
            log_likelihood,
            n_live=8_000,
            pool=pool,
            seed=42,
            pass_dict=False,
            n_networks=5,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    one_sigma_ci = [0.159, 0.5, 0.841]

    Om_samples = samples[:, 3] / (samples[:, 2] / 100) ** 2

    dM_16, dM_50, dM_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    H0_16, H0_50, H0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    Omh2_16, Omh2_50, Omh2_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    v_16, v_50, v_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, rd_50, H0_50, Omh2_50, v_50]
    MAP_params = samples[np.argmax(log_l)]

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

    degs_of_freedom = len(bao_data) + len(z_cmb) - len(best_fit)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"Chi2 (MAP): {chi_squared(MAP_params):.1f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"Degs of freedom: {degs_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit, DM_z(z_cmb, best_fit)),
        y_err=np.sqrt(np.diag(sn_cov_matrix)),
        y_model=mu_theory(best_fit, DM_z(z_cmb, best_fit)),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
DESI BAO DR2 2025
Union3.1 SNe 2026
Om x h2 prior from Planck 2018

Priors:
ΔM U(-1.0, +1.0)
rd U(120, 160)
H0 U(50.0, 85.0)
ωm N(0.1430, 0.0011)

wCDM:
w0 U(-1.3, -0.5)

Quintessence model wzCDM:
w0 U(-1.0, -0.2)

w0waCDM:
w0 U(-1.5, 0.0)
wa U(-5.0, +3.0)
w0 + wa < 0 enforced

flow correction:
v U(-12, 5) x 100 km/s
"""

"""
Flat ΛCDM: w(z) = -1

ΔM: -0.036 +0.025 -0.024 mag
rd: 147.03 +1.27 -1.27 Mpc
H0: 68.82 +0.99 -0.95 km/s/Mpc
Ωm: 0.302 +0.008 -0.008
ωm: 0.1430 +0.0011 -0.0011
Chi squared: 41.1
Log evidence: -32.2
Degs of freedom: 31
"""

"""
Flat ΛCDM
Isotropic velocity SNe observed redshifts (turning point z <= 0.2 inflow z > 0.2 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

v: -3.11 +1.06 -1.07 x 100 km/s
ΔM: -0.025 +0.025 -0.025 mag
rd: 146.43 +1.29 -1.27 Mpc
H0: 69.34 +0.99 -0.98 km/s/Mpc
Ωm: 0.297 +0.008 -0.008
ωm: 0.1430 +0.0011 -0.0011
Chi squared: 32.4 (2.95 sigma significance)
Log evidence: -29.7 (Δ logZ = 2.5 in favour of flow correction)
Degs of freedom: 30
"""

"""
Flat wCDM: w(z) = w0

w0: -0.895 +0.049 -0.050
ΔM: 0.011 +0.037 -0.035 mag
rd: 143.35 +2.27 -2.41 Mpc
H0: 69.39 +1.08 -1.04 km/s/Mpc
Ωm: 0.297 +0.009 -0.009
ωm: 0.1430 +0.0011 -0.0011
Chi squared: 36.8 (2.07 sigma significance)
Log evidence: -31.9 (Δ logZ = 0.3 against ΛCDM)
Degs of freedom: 30

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)

w0: -0.819 +0.071 -0.073
ΔM: -0.013 +0.027 -0.026 mag
rd: 144.90 +1.54 -1.56 Mpc
H0: 68.20 +0.99 -0.98 km/s/Mpc
Ωm: 0.307 +0.009 -0.009
ωm: 0.1430 +0.0011 -0.0011
Chi squared: 35.4 (2.39 sigma significance)
Log evidence: -30.9 (Δ logZ = 1.3 against ΛCDM)
Degs of freedom: 30

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
TODO
"""
