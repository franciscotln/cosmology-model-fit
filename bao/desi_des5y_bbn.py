from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2025DESdovekie.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data


c = c0 / 1000  # km/s

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def r_drag(wb, wm):
    """
    arXiv:2106.00428v2 (eq 8)
    Alternatively z_drag from the same paper can we used
    to compute the integral over c_s / H(z) yielding the same results.
    """
    a1 = 0.00257366
    a2 = 0.05032
    a3 = 0.013
    a4 = 0.7720642
    a5 = 0.24346362
    a6 = 0.00641072
    a7 = 0.5350899
    a8 = 32.7525
    a9 = 0.315473

    term_A_denominator = (a1 * (wb**a2)) + (a3 * (wb**a4) * (wm**a5)) + (a6 * (wm**a7))
    term_A = 1.0 / term_A_denominator
    term_B = a8 / (wm**a9)
    return term_A - term_B


@njit
def Ez(z, params):
    Om, w0 = params[1], params[3]
    Ode = 1.0 - Om
    one_plus_z = 1.0 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2
    return np.sqrt(Om * cubed + Ode * rho_de)


def theory_mu(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[-1] + 25.0 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


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
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    rd = r_drag(wb=Obh2, wm=Omh2)
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi_bao + chi_sn


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from scipy.stats import norm
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(55, 80))
    prior.add_parameter("Om", dist=(0.10, 0.65))
    prior.add_parameter("ωb", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("w0", dist=(-1.0, -1 / 3))
    prior.add_parameter("dM", dist=(-0.5, 0.5))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False
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

    H0_16, H0_50, H0_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    Obh2_16, Obh2_50, Obh2_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    best_fit = [H0_50, Om_50, Obh2_50, w0_50, dM_50]

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    rd_samples = r_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {sampler.log_z:.2f}")
    print(f"Degrees of freedom: {len(bao_data['z']) + sn_size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
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


if __name__ == "__main__":
    main()

"""
*******************************
DESI DR2 + DES5Y + BBN Schöngerg+2024

Priors:
H0 U(55, 80)
Om U(0.10, 0.65)
ωb N(bbn.Obh2, bbn.Obh2_sigma)
dM U(-0.5, 0.5)

wCDM:
w0 U(-1.5, -0.5)

wzCDM:
w0 U(-1.0, -1/3)

w0waCDM:
w0 U(-1.5, 0.0)
wa U(-4.0, 2.5)

*******************************

Flat ΛCDM w(z) = -1
H0: 68.8 +0.6 -0.6 km/s/Mpc
Ωm: 0.3064 +0.0078 -0.0076
ωb: 0.02219 +0.00055 -0.00055
ωm: 0.14506 +0.00479 -0.00464
w0: -1
wa: 0
r_d: 146.56 +1.46 -1.47 Mpc
Chi squared: 1645.28
Log Evidence: -833.65
Degrees of freedom: 1723

===============================

Flat wCDM w(z) = w0
H0: 66.3 +1.2 -1.2 km/s/Mpc
Ωm: 0.2975 +0.0086 -0.0086
ωb: 0.02218 +0.00055 -0.00055
ωm: 0.13065 +0.00757 -0.00734
w0: -0.910 +0.037 -0.038
wa: 0
ΔM: -0.106 +0.034 -0.034
r_d: 150.49 +2.30 -2.26 Mpc
Chi squared: 1639.50
Log Evidence: -833.20 (Δ logZ = 0.45 against ΛCDM)
Degrees of freedom: 1722

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
H0: 66.8 +1.0 -1.0 km/s/Mpc
Ωm: 0.3048 +0.0078 -0.0075
ωb: 0.02218 +0.00055 -0.00054
ωm: 0.13582 +0.00571 -0.00543
w0: -0.861 +0.051 -0.052
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
ΔM: -0.083 +0.025 -0.025
r_d: 149.03 +1.75 -1.74 Mpc
Chi squared: 1638.46
Log Evidence: -831.95 (Δ logZ = 1.70 against ΛCDM)
Degrees of freedom: 1722

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.9 +1.4 -1.5 km/s/Mpc
Ωm: 0.3157 +0.0125 -0.0136
ωb: 0.02218 +0.00055 -0.00055
ωm: 0.14555 +0.01045 -0.01132
w0: -0.836 +0.070 -0.061
wa: -0.590 +0.399 -0.443
r_d: 146.46 +3.15 -2.76 Mpc
Chi squared: 1638.14
Log Evidence: -834.70 (Δ logZ = -1.05 in favour of ΛCDM)
Degrees of freedom: 1721
"""
