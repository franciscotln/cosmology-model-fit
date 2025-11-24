from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from scipy.constants import c as c0
from y2023union3.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=2000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    h, Omh2, w0 = params[2] / 100, params[3], params[4]
    Om = Omh2 / h**2
    zp1 = 1 + z
    cubic = zp1**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Om * cubic + (1 - Om) * rho_de)


@njit
def mu_theory(params):
    dL = (1 + z_sn_vals) * DM_z(z_sn_vals, params)
    return params[0] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


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


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return y @ y


def chi_squared(params):
    delta_sn = mu_vals - mu_theory(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)
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
    # Planck prior on ωm = Ωm * h^2
    prior.add_parameter("ωm", dist=norm(loc=0.1430, scale=0.0011))
    prior.add_parameter("w0", dist=(-1.5, 0.0))
    prior.add_parameter("wa", dist=(-5.0, +3.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False
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
    w0_16, w0_50, w0_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, rd_50, H0_50, Omh2_50, w0_50]

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

    degs_of_freedom = len(bao_data["value"]) + len(z_sn_vals) - len(best_fit)

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
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
*******************************
DESI BAO DR2 2025
Union3 SNe
Omh2 prior from Planck 2018

Priors:
ΔM U(-1.0, +1.0)
rd U(120, 160)
H0 U(50.0, 85.0)
ωm N(0.1430, 0.0011)
w0 U(-1.5, 0.0)
wa U(-5.0, +3.0)

*******************************

Flat ΛCDM: w(z) = -1
rd: 147.25 +1.29 -1.29 Mpc
H0: 68.63 +0.99 -0.97 km/s/Mpc
Ωm: 0.304 +0.008 -0.008
ωm: 0.1430 +0.0011 -0.0011
w0: -1
wa: 0
Chi squared: 38.8
Log evidence: -28.0
Degs of freedom: 31

===============================

Flat wCDM: w(z) = w0
rd: 142.48 +2.37 -2.61 Mpc
H0: 69.35 +1.12 -1.08 km/s/Mpc
Ωm: 0.297 +0.009 -0.009
ωm: 0.1430 +0.0011 -0.0011
w0: -0.865 +0.051 -0.052
wa: 0
Chi squared: 32.2
Log evidence: -27.1 (Δ logZ = 0.9 against ΛCDM)
Degs of freedom: 30

===============================

Flat wzCDM: w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)
rd: 144.46 +1.61 -1.60 Mpc
H0: 67.77 +1.01 -0.99 km/s/Mpc
Ωm: 0.311 +0.009 -0.009
ωm: 0.1430 +0.0011 -0.0011
w0: -0.774 +0.074 -0.076
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
Chi squared: 30.1
Log evidence: -25.7 (Δ logZ = 2.3 against ΛCDM)
Degs of freedom: 30

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
rd: 147.99 +2.45 -3.20 Mpc
H0: 65.85 +1.95 -1.55 km/s/Mpc
Ωm: 0.330 +0.016 -0.019
ωm: 0.1430 +0.0011 -0.0011
w0: -0.702 +0.114 -0.111
wa: -0.980 +0.571 -0.565
Chi squared: 28.8
Log evidence: -27.0 (Δ logZ = 1.0 against ΛCDM)
Degs of freedom: 29
"""
