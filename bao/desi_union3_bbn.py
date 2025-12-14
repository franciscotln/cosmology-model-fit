from numba import njit
import numpy as np
from scipy.constants import c as c0
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2023union3.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data


c = c0 / 1000  # km/s

sn_legend, z_cmb, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=2300)
dx = np.diff(z_grid)


@njit
def r_drag(wb, wm):
    # arXiv:2106.00428v2 (eq 8)
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
    zp1 = 1 + z
    cubed = zp1**3
    rho_de = (2 * cubed / (1 + w0 + (1 - w0) * cubed)) ** 2
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


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
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


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

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / r_drag(Obh2, Omh2)


@njit
def theory_mu(params):
    dL = (1 + z_cmb) * DM_z(z_cmb, params)
    return params[-1] + 25 + 5 * np.log10(dL)


@njit
def chi_squared(params):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_sn = mu_values - theory_mu(params)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    return chi_bao + chi_sn


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def q0(Om, w0=-1):
    """Calculate the deceleration parameter at z=0."""
    return Om / 2 + (1 + 3 * w0) * (1 - Om) / 2


def j0(Om, w0=-1, wa=0):
    """Calculate the jerk parameter at z=0."""
    return 1 + (3 / 2) * (1 - Om) * (3 * w0 * (1 + w0) + wa)


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
    prior.add_parameter("Ωm", dist=(0.10, 0.65))
    prior.add_parameter("ωb", dist=norm(loc=bbn.Obh2, scale=bbn.Obh2_sigma))
    prior.add_parameter("w0", dist=(-1.0, -1 / 3))
    prior.add_parameter("dM", dist=(-1.0, 1.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=9_000, pool=pool, seed=42, pass_dict=False
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

    wa_samples = -1.5 * (1 - samples[:, 3] ** 2)
    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    rd_samples = r_drag(samples[:, 2], Omh2_samples)
    q0_samples = q0(samples[:, 1], w0=samples[:, 3])
    j0_samples = j0(samples[:, 1], w0=samples[:, 3], wa=wa_samples)

    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(rd_samples, one_sigma_ci, weights=w)
    q0_16, q0_50, q0_84 = quantile(q0_samples, one_sigma_ci, weights=w)
    j0_16, j0_50, j0_84 = quantile(j0_samples, one_sigma_ci, weights=w)
    wa_16, wa_50, wa_84 = quantile(wa_samples, one_sigma_ci, weights=w)

    best_fit = [H0_50, Om_50, Obh2_50, w0_50, dM_50]

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"wa: {wa_50:.3f} +{(wa_84 - wa_50):.3f} -{(wa_50 - wa_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"q0: {q0_50:.3f} +{(q0_84 - q0_50):.3f} -{(q0_50 - q0_16):.3f}")
    print(f"j0: {j0_50:.3f} +{(j0_84 - j0_50):.3f} -{(j0_50 - j0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {sampler.log_z:.2f}")
    print(f"Degrees of freedom: {len(bao_data['z']) + len(z_cmb) - len(best_fit)}")

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
DESI DR2 + Union3 + BBN Schöngerg2024

Priors:

All models:
H0 U(55, 80)
Om U(0.10, 0.65)
ωb N(0.02218, 0.00055)
dM U(-1.0, 1.0)

wCDM:
w0 U(-1.3, -0.3)

wzCDM:
w0 U(-1.0, -1/3)

w0waCDM:
w0 U(-1.3, 0.0)
wa U(-4.0, 2.0)
"""

"""
Flat ΛCDM w(z) = -1
H0: 68.8 +0.6 -0.6 km/s/Mpc
Ωm: 0.3040 +0.0084 -0.0081
ωb: 0.02219 +0.00055 -0.00055
ωm: 0.14388 +0.00506 -0.00487
w0: -1
wa: 0
r_d: 146.87 +1.54 -1.52 Mpc
q0: -0.544 +0.013 -0.012
j0: 1
Chi squared: 38.82
Log Evidence: -28.11
Degrees of freedom: 31

===============================

Flat wCDM w(z) = w0
H0: 65.1 +1.6 -1.6 km/s/Mpc
Ωm: 0.2980 +0.0091 -0.0091
ωb: 0.02218 +0.00055 -0.00055
ωm: 0.12646 +0.00845 -0.00851
w0: -0.868 +0.051 -0.052
wa: 0
r_d: 151.70 +2.71 -2.53 Mpc
q0: -0.413 +0.051 -0.052
j0: 0.324 +0.252 -0.236
Chi squared: 32.15
Log Evidence: -26.91 (Δ logZ = 1.20 against ΛCDM)
Degrees of freedom: 30

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
H0: 65.4 +1.3 -1.3 km/s/Mpc
Ωm: 0.3122 +0.0090 -0.0088
ωb: 0.02219 +0.00055 -0.00055
ωm: 0.13346 +0.00598 -0.00568
w0: -0.765 +0.074 -0.077
wa: -0.622 +0.185 -0.161 [derived wa = -1.5 * (1 - w0**2)]
r_d: 149.68 +1.84 -1.83 Mpc
q0: -0.289 +0.080 -0.084
j0: -0.198 +0.332 -0.266
Chi squared: 29.96
Log Evidence: -25.01 (Δ logZ = 3.10 against ΛCDM)
Degrees of freedom: 30

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 66.6 +1.4 -1.5 km/s/Mpc
Ωm: 0.3313 +0.0156 -0.0177
ωb: 0.02219 +0.00055 -0.00055
ωm: 0.14753 +0.01020 -0.01183
w0: -0.695 +0.116 -0.111
wa: -1.023 +0.556 -0.565
ΔM: -0.154 +0.099 -0.103
r_d: 145.96 +3.24 -2.67 Mpc
q0: -0.198 +0.128 -0.129
Chi squared: 28.84
Log Evidence: -26.92 + 0.25 = -26.67 (Δ logZ = 1.44 against ΛCDM)
Degrees of freedom: 29
"""
