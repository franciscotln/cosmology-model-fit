from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2026union3_1.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def Ez(z, params):
    Om, w0 = params[3], params[4]
    zp1 = 1.0 + z
    cubic = zp1**3
    rho_de = (2 * zp1**3 / (1 + w0 + (1 - w0) * zp1**3)) ** 2
    return np.sqrt(Om * cubic + (1.0 - Om) * rho_de)


@njit
def mu_theory(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


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
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


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


@njit
def chi_squared(params):
    delta_sn = mu_vals - mu_theory(params)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
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
    prior.add_parameter("rd", dist=norm(loc=147.09, scale=0.26))  # Planck prior
    prior.add_parameter("H0", dist=(50.0, 85.0))
    prior.add_parameter("Ωm", dist=(0.1, 0.6))
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
    rd_16, rd_50, rd_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    H0_16, H0_50, H0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 3] * (samples[:, 2] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, rd_50, H0_50, Om_50, w0_50]

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

    degs_of_freedom = len(bao_data["z"]) + len(z_cmb) - len(best_fit)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
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
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
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
ΔM U(-1.0, +1.0)
rd N(147.09, 0.26)
H0 U(50.0, 85.0)
Ωm U(0.1, 0.6)

mag evolution:
p U(-1.0, 2.5)

wCDM:
w0 U(-1.5, 0.0)

wzCDM thawing quintessence:
w0 U(-1.0, -1/3)

w0waCDM:
w0 U(-1.5, 0.0)
wa U(-3.5, 2.5)

w0 + wa < 0 enforced
"""

"""
Flat ΛCDM: w(z) = -1
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
Flat ΛCDM: w(z) = -1
Evolving absolute mag of SNe M(z) = ΔM_max + 0.2 * p / (1 + (z / 0.043))

ΔM_max: -0.049 +0.012 -0.012 mag
p: 0.677 +0.299 -0.300
rd: 147.09 +0.26 -0.26 Mpc
H0: 69.05 +0.50 -0.50 km/s/Mpc
Ωm: 0.297 +0.008 -0.008
ωm: 0.1417 +0.0023 -0.0023
Chi squared: 36.0 (2.26 sigma away from no evolution in M)
Log evidence: -32.3 (Δ logZ = 1.0 against no evolution in M)
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
