from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
from y2026union3_1.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_sn_data()
bao_legend, bao, bao_cov_matrix = get_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    inv_a3 = (1.0 + z) ** 3
    return (2 * inv_a3 / (1.0 + w0 + (1.0 - w0) * inv_a3)) ** 2


@njit
def H_z(z, params):
    H0, Om = params[2], params[3]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


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

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH[DH_mask]
    results[DM_mask] = DM[DM_mask]
    results[DV_mask] = DV_z(z[DV_mask], DH[DV_mask], DM[DV_mask])
    return results / params[1]


pivot_mask = z_cmb <= 0.2

"""
z node
0.05   v: -0.182 +1.030 -1.046 x 100 km/s   Chi squared: 41.1   Log evidence: -34.9
0.10   v: -2.285 +0.960 -0.952 x 100 km/s   Chi squared: 35.4   Log evidence: -32.2
0.15   v: -2.160 +0.969 -0.958 x 100 km/s   Chi squared: 36.1   Log evidence: -32.5
0.20   v: -3.100 +1.040 -1.060 x 100 km/s   Chi squared: 32.4   Log evidence: -30.6 <-- best node
0.25   v: -1.542 +1.135 -1.148 x 100 km/s   Chi squared: 39.3   Log evidence: -33.9
0.30   v: -1.451 +1.291 -1.288 x 100 km/s   Chi squared: 39.8   Log evidence: -34.1
0.35   v: -1.802 +1.407 -1.415 x 100 km/s   Chi squared: 39.4   Log evidence: -33.8
0.40   v: -2.914 +1.499 -1.513 x 100 km/s   Chi squared: 37.3   Log evidence: -32.7
0.45   v: -2.887 +1.591 -1.563 x 100 km/s   Chi squared: 37.7   Log evidence: -32.9
0.50   v: -3.336 +1.655 -1.704 x 100 km/s   Chi squared: 37.0   Log evidence: -32.4
0.55   v: -3.710 +1.736 -1.733 x 100 km/s   Chi squared: 36.4   Log evidence: -32.1
0.60   v: -4.145 +1.750 -1.801 x 100 km/s   Chi squared: 35.5   Log evidence: -31.6
0.65   v: -4.247 +1.770 -1.786 x 100 km/s   Chi squared: 35.3   Log evidence: -31.5
0.70   v: -3.911 +1.776 -1.787 x 100 km/s   Chi squared: 36.2   Log evidence: -32.0
0.75   v: -3.620 +1.818 -1.828 x 100 km/s   Chi squared: 36.9   Log evidence: -32.3
0.80   v: -3.644 +1.790 -1.803 x 100 km/s   Chi squared: 37.0   Log evidence: -32.3
0.90   v: -3.655 +1.819 -1.820 x 100 km/s   Chi squared: 37.0   Log evidence: -32.3
1.00   v: -3.900 +1.819 -1.810 x 100 km/s   Chi squared: 36.5   Log evidence: -32.1
1.10   v: -3.803 +1.810 -1.836 x 100 km/s   Chi squared: 36.7   Log evidence: -32.2
1.25   v: -3.932 +1.799 -1.830 x 100 km/s   Chi squared: 36.3   Log evidence: -32.0
1.40   v: -3.871 +1.790 -1.831 x 100 km/s   Chi squared: 36.5   Log evidence: -32.1
"""


@njit
def mu_corr(params, dm_interp):
    # Heaviside step at z = 0.2
    v_km_s = 100 * params[4] * np.where(pivot_mask, 1, -1)
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + v_km_s / c)
    return 5.0 * np.log10(DM_z(z_cosmo, dm_interp) / DM_z(z_cmb, dm_interp))


@njit
def mu_theory(params, dm_interp):
    return params[0] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM_z(z_cmb, dm_interp))


@njit
def chi_squared(params):
    dm_interp = DM_grid(params)

    delta_sn = mu_vals - mu_theory(params, dm_interp) - mu_corr(params, dm_interp)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    delta_bao = bao["value"] - bao_theory(bao["z"], bao_qty, params, dm_interp)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao
    return chi_sn + chi_bao


@njit
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
    prior.add_parameter("rd", dist=norm(loc=147.09, scale=0.26))  # Planck prior
    prior.add_parameter("H0", dist=(50.0, 85.0))
    prior.add_parameter("Ωm", dist=(0.1, 0.6))
    prior.add_parameter("v", dist=(-10.5, 4.5))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=8_000, pool=pool, seed=42, pass_dict=False
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

    degs_of_freedom = len(bao) + len(z_cmb) - len(best_fit)

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

"""
DESI BAO DR2 2025
Union 3.1 SNe 2026
rdrag prior from Planck 2018

Priors:
ΔM ~U(-1.0, +1.0)
rd ~N(147.09, 0.26)
H0 ~U(50.0, 85.0)
Ωm ~U(0.1, 0.6)

flow correction:
v ~U(-10.5, 4.5) x 100 km/s

wCDM:
w0 ~U(-1.5, 0.0)

wzCDM thawing quintessence:
w0 ~U(-1.0, -1/3)

w0waCDM:
w0 ~U(-1.5, 0.0)
wa ~U(-3.5, 2.5)

w0 + wa < 0 enforced
"""

"""
Flat ΛCDM
ΔM: -0.037 +0.011 -0.011 mag
rd: 147.09 +0.26 -0.26 Mpc
H0: 68.79 +0.49 -0.49 km/s/Mpc
Ωm: 0.302 +0.008 -0.008
ωm: 0.1429 +0.0023 -0.0023
Chi squared: 41.1
Log evidence: -33.3
Degs of freedom: 31
"""

"""
Flat ΛCDM
Isotropic velocity SNe observed redshifts (turning point z <= 0.2 inflow z > 0.2 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

ΔM: -0.035 +0.011 -0.011 mag
v: -3.10 +1.06 -1.06 x 100 km/s
v / (z_cut=0.2): -1550 ± 530 km/s
rd: 147.09 +0.26 -0.26 Mpc
H0: 69.03 +0.50 -0.49 km/s/Mpc
Ωm: 0.298 +0.008 -0.008
ωm: 0.1418 +0.0023 -0.0023
Chi squared: 32.4 (2.95 sigma significance)
Log evidence: -30.7 (Δ logZ = 2.6 in favour of flow corrections)
Degs of freedom: 30
"""

"""
Flat wCDM: w(z) = w0
ΔM: -0.046 +0.011 -0.011 mag
rd: 147.09 +0.26 -0.26 Mpc
H0: 67.60 +0.74 -0.74 km/s/Mpc
Ωm: 0.297 +0.009 -0.009
ωm: 0.1359 +0.0043 -0.0044
w0: -0.896 +0.049 -0.050
Chi squared: 36.8 (2.07 sigma away from ΛCDM)
Log evidence: -33.7 (Δ logZ = -0.4 in favour of ΛCDM)
Degs of freedom: 30
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
ΔM: -0.046 +0.011 -0.011 mag
rd: 147.09 +0.26 -0.26 Mpc
H0: 67.17 +0.82 -0.82 km/s/Mpc
Ωm: 0.308 +0.009 -0.009
ωm: 0.1388 +0.0029 -0.0028
w0: -0.818 +0.072 -0.074
wa: d w(z)/d z at z=0 = -1.5 * (1 - w0^2) = -0.624
Chi squared: 35.4 (2.39 sigma away from ΛCDM)
Log evidence: -31.8 (Δ logZ = 1.5 against ΛCDM)
Degs of freedom: 30
"""

"""
Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
ΔM: -0.045 +0.011 -0.011 mag
rd: 147.09 +0.26 -0.26 Mpc
H0: 66.98 +0.87 -0.86 km/s/Mpc
Ωm: 0.322 +0.016 -0.019
ωm: 0.1444 +0.0049 -0.0068
w0: -0.776 +0.106 -0.102
wa: -0.769 +0.569 -0.550
Chi squared: 34.7 (0.96 sigma away from ΛCDM)
Log evidence: -34.1 (TODO: remove forbidden volume, still ΛCDM is preferred)
Degs of freedom: 29
"""
