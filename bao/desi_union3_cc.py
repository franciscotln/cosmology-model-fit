from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2026union3_1.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, sn_mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def Ez(z, params):
    Om, w0 = params[4], params[5]
    zp1 = 1.0 + z
    cubed = zp1**3
    rho_de = (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2
    return np.sqrt(Om * cubed + (1.0 - Om) * rho_de)


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
    return results / params[3]


@njit
def mu_theory(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[1] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_sn = sn_mu_vals - mu_theory(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    f_cc = params[0]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = solve_triang(cho_cc, delta_cc) * f_cc**2

    return chi_sn + chi_bao + chi_cc


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def main():
    from corner import quantile
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from cosmic_chronometers.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    # f_cc: CC error rescaling (overestimated)
    prior.add_parameter("fcc", dist=(0.01, 3.0))
    # ΔM: magnitude offset
    prior.add_parameter("ΔM", dist=(-1.0, 1.0))
    # H0: Hubble constant at present
    prior.add_parameter("H0", dist=(50.0, 85.0))
    # rd: sound horizon at drag epoch
    prior.add_parameter("rd", dist=(120.0, 180.0))
    # Ωm: matter density parameter today
    prior.add_parameter("Ωm", dist=(0.01, 0.70))
    # w0: dark energy equation of state today
    prior.add_parameter("w0", dist=(-1.0, -1 / 3))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

    labels = ["f_{cc}", "ΔM", "H_0", "rd", "Ω_m", "w_0"]
    gd_samples = MCSamples(
        samples=samples, weights=w, names=prior.keys, labels=labels, loglikes=log_l
    )
    plots.get_subplot_plotter().triangle_plot(
        gd_samples, filled=True, title_limit=1, color=["blue"], contour_colors=["blue"]
    )
    plt.show()

    one_sigma_ci = [0.159, 0.5, 0.841]

    fcc_16, fcc_50, fcc_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    h0_16, h0_50, h0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)

    best_fit = [fcc_50, dM_50, h0_50, rd_50, Om_50, w0_50]

    Omh2_samples = samples[:, 4] * samples[:, 2] ** 2 / 100**2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    deg_of_freedom = len(z_cmb) + len(bao_data["z"]) + len(z_cc_vals) - len(best_fit)

    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=sn_mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"$H_0$={h0_50:.2f}, $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / fcc_50,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )


if __name__ == "__main__":
    main()


"""
BAO DESI DR2 + SN1a Union3.1 + Cosmic Chronometers

Priors:
f_cc: U(0.01, 3.0)
ΔM:   U(-1.0, 1.0)
H0:   U(50.0, 85.0)
rd:   U(120.0, 180.0)
Ωm:   U(0.01, 0.70)

wzCDM:
w0:   U(-1.0, -1/3)

wCDM:
w0:   U(-1.2, -0.5)

w0waCDM:
w0:   U(-1.5, 0.0)
wa:   U(-5.0, 3.0)
Enforced w0 + wa > 0

Varying absolute magnitude SNe within ΛCDM:
p:    U(-0.4, 1.0)
"""

"""
Flat ΛCDM: w(z) = -1
f_cc: 1.48 +0.18 -0.17
ΔM: -0.040 +0.070 -0.072 mag
H0: 68.7 +2.3 -2.3 km/s/Mpc
r_d: 147.2 +4.9 -4.6 Mpc
Ωm: 0.303 +0.008 -0.008
ωm: 0.1437 +0.0330 -0.0233
Chi squared: 76.36
Log evidence: -179.01
Degrees of freedom: 66
"""

"""
Flat ΛCDM: w(z) = -1, evolving absolute magnitude SNe
M(z) = ΔM_inf + 1 - (z / (1 + z))^(0.1 * p)

f_cc: 1.48 +0.17 -0.17
ΔM: -0.066 +0.073 -0.073 mag
p: 0.26 +0.13 -0.13
H0: 69.0 +2.3 -2.3 km/s/Mpc
r_d: 147.2 +4.5 -5.1 Mpc
Ωm: 0.298 +0.008 -0.008
ωm: 0.1426 +0.0309 -0.0219
Chi squared: 71.79 (2.14 sigma away from no mag evolution)
Log evidence: -178.26 (Δ logZ = 0.75 against no mag evolution)
Degrees of freedom: 65
"""

"""
Flat wCDM: w(z) = w0
f_cc: 1.48 +0.18 -0.17
ΔM: -0.049 +0.070 -0.072 mag
H0: 67.5 +2.3 -2.3 km/s/Mpc
r_d: 147.3 +4.9 -4.6 Mpc
Ωm: 0.298 +0.009 -0.009
ωm: 0.1359 +0.0299 -0.0248
w0: -0.898 +0.049 -0.050
Chi squared: 72.09 (2.07 sigma away from ΛCDM)
Log evidence: -178.66 (Δ logZ = 0.35 against ΛCDM)
Degrees of freedom: 65
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
f_cc: 1.47 +0.18 -0.17
ΔM: -0.049 +0.070 -0.072 mag
H0: 67.1 +2.4 -2.3 km/s/Mpc
r_d: 147.3 +4.9 -4.6 Mpc
Ωm: 0.308 +0.009 -0.008
ωm: 0.1391 +0.0295 -0.0234
w0: -0.825 +0.071 -0.074
wa: d w(z)/d z at z=0 = -1.5 * (1 - w0^2)
Chi squared: 70.70 (2.38 sigma away from ΛCDM)
Log evidence: -177.67 (Δ logZ = 1.34 against ΛCDM)
Degrees of freedom: 65
"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
f_cc: 1.47 +0.18 -0.17
ΔM: -0.047 +0.071 -0.072 mag
H0: 67.0 +2.4 -2.3 km/s/Mpc
r_d: 147.2 +4.9 -4.7 Mpc
Ωm: 0.319 +0.016 -0.020
ωm: 0.1444 +0.0368 -0.0254
w0: -0.801 +0.105 -0.097
wa: -0.633 +0.564 -0.555
Chi squared: 70.15 (2.01 sigma away from ΛCDM)
Log evidence: -180.47 (Δ logZ = -1.46 in favour of ΛCDM. TODO: remove forbidden region but still not preferred)
Degrees of freedom: 64
"""
