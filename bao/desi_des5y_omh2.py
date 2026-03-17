from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025DESdovekie.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(cov_matrix_bao)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence (not used now)
    cubic = (1.0 + z) ** 3
    return (2 * cubic / (1.0 + w0 + (1.0 - w0) * cubic)) ** 2


@njit
def Ez(z, params):
    h, Omh2 = params[2] / 100, params[3]
    Om = Omh2 / h**2
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dy)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    rdrag = params[1]
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rdrag


@njit
def mu_corr(params, DM_obs):
    # Heaviside step function
    v_km_s = 100 * params[4] * np.where(z_cmb <= 0.10563, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)

    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_obs)


@njit
def theory_mu(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi2_sn(params):
    DM = DM_z(z_cmb, params)
    delta_sn = mu_values - theory_mu(params[0], DM) - mu_corr(params, DM)
    return solve_triang(cho_sn, delta_sn)


@njit
def chi2_bao(params):
    delta_bao = bao["value"] - bao_theory(bao["z"], bao_qty, params)
    return delta_bao @ inv_cov_bao @ delta_bao


def chi_squared(params):
    return chi2_sn(params) + chi2_bao(params)


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
    prior.add_parameter("ΔM", dist=(-0.5, +0.5))
    prior.add_parameter("rd", dist=(120.0, 165.0))
    prior.add_parameter("H0", dist=(50.0, 90.0))
    prior.add_parameter("ωm", dist=norm(loc=0.1430, scale=0.0011))  # Planck prior
    prior.add_parameter("v", dist=(-5.5, 2.5))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=8_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    Om_samples = samples[:, 3] / (samples[:, 2] / 100) ** 2

    w = np.exp(log_w)
    one_sigma_ci = [0.159, 0.5, 0.841]

    dM_16, dM_50, dM_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    H0_16, H0_50, H0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    Omh2_16, Omh2_50, Omh2_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    v_16, v_50, v_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)

    best_fit = [dM_50, rd_50, H0_50, Omh2_50, v_50]

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"Degrees of freedom: {len(bao) + sn_size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(best_fit, DM_z(z_cmb, best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit[0], DM_z(z_cmb, best_fit)),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
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


if __name__ == "__main__":
    main()


"""
*******************************
DESI BAO DR2 2025
DES5Y Dovekie SNe
Om x h^2 prior from Planck 2018

Priors:
ΔM U(-0.5, +0.5)
rd U(120, 165)
H0 U(50.0, 90.0)
ωm N(0.1430, 0.0011)

w0 U(-1.5, 0.0)
wa U(-3.0, +2.0) (w0 + wa > 0 enforced)

v  U(-5.5, +2.5) x 100 km/s
*******************************
"""

"""
Flat ΛCDM
ΔM: -0.059 +0.024 -0.024 mag
r_d: 147.55 +1.22 -1.20 Mpc
H0: 68.37 +0.90 -0.89 km/s/Mpc
ωm: 0.1430 +0.0011 -0.0011
Ωm: 0.306 +0.008 -0.008
Chi squared: 1645.3
Log evidence: -833.8
Degrees of freedom: 1723

===============================

Flat ΛCDM +  v corrections
v: -1.59 +0.58 -0.57 x 100 km/s
ΔM: -0.041 +0.025 -0.025 mag
r_d: 146.71 +1.25 -1.25 Mpc
H0: 69.11 +0.94 -0.93 km/s/Mpc
ωm: 0.1430 +0.0011 -0.0011
Ωm: 0.299 +0.008 -0.008
Chi squared: 1637.4 (2.81 sigma significance)
Log evidence: -831.6 (Δ logZ = 2.2 in favour of v corrections)
Degrees of freedom: 1722
"""

"""
Flat wCDM w(z) = w0
r_d: 143.73 +2.07 -2.15 Mpc
H0: 69.38 +1.06 -1.02 km/s/Mpc
ωm: 0.1430 +0.0011 -0.0011
Ωm: 0.297 +0.009 -0.009
w0: -0.908 +0.037 -0.038
wa: 0
Chi squared: 1639.5
Log evidence: -833.6 (Δ logZ = 0.2 against ΛCDM)
Degrees of freedom: 1722

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
r_d: 145.20 +1.51 -1.51 Mpc
H0: 68.52 +0.90 -0.89 km/s/Mpc
ωm: 0.1430 +0.0011 -0.0011
Ωm: 0.305 +0.008 -0.007
w0: -0.860 +0.051 -0.052
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
Chi squared: 1638.5
Log evidence: -832.0 (Δ logZ = 1.8 against ΛCDM)
Degrees of freedom: 1722

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
r_d: 147.10 +2.78 -3.88 Mpc
H0: 67.64 +2.02 -1.46 km/s/Mpc
ωm: 0.1430 +0.0011 -0.0011
Ωm: 0.313 +0.014 -0.018
w0: -0.849 +0.074 -0.066
wa: -0.494 +0.490 -0.479
Chi squared: 1638.1
Log evidence: -833.1 (Δ logZ = 0.7 against ΛCDM)
Degrees of freedom: 1721
"""
