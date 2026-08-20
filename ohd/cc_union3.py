from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2026union3_1.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data

legend_sn, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_sn_data()
legend_cc, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_cc = np.linalg.inv(cov_matrix_cc)

c = c0 / 1000  # Speed of light in km/s

z_grid = np.linspace(0, np.max(z_cmb), num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2


@njit
def H_z(z, params):
    H0, Om = params[2], params[3]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    dm_grid = np.zeros(z_grid.size, dtype=np.float64)
    dm_grid[1:] = np.cumsum(dh * dz)
    return (dm_grid, dh_grid)


@njit
def DM_z(z, DM_interp):
    return interp_hermite(z, z_grid, *DM_interp)


@njit
def mu_corr(v, DM_interp):
    # Heaviside step at z = 0.2
    v_km_s = v * np.where(z_cmb <= 0.2, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)

    return 5 * np.log10(DM_z(z_cosmo, DM_interp) / DM_z(z_cmb, DM_interp))


@njit
def mu_theory(offset, DM_interp):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM_z(z_cmb, DM_interp))


@njit
def chi_squared(params):
    f_cc, offset, v = params[0], params[1], params[4]

    DM_interp = DM_grid(params)
    delta_sn = mu_vals - mu_theory(offset, DM_interp) - mu_corr(v, DM_interp)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = f_cc**2 * cc_delta @ inv_cov_cc @ cc_delta

    return chi_sn + chi_cc


@njit
def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def main():
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from corner import quantile
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from ohd.plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("f_cc", dist=(0.05, 3.35))
    prior.add_parameter("dM", dist=(-1.0, 1.0))
    prior.add_parameter("H0", dist=(40.0, 95.0))
    prior.add_parameter("Om", dist=(0.1, 0.7))
    prior.add_parameter("v", dist=(-900, 900))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    log_evd = sampler.log_z
    labels = ["f_{cc}", "ΔM", "H_0", "Ω_m", "v"]

    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )

    plots.get_subplot_plotter().triangle_plot(
        gd_samples, prior.keys, title_limit=1, contour_colors=["C0"]
    )
    plt.show()

    one_sigma_ci = [0.159, 0.5, 0.841]

    fcc_16, fcc_50, fcc_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    h0_16, h0_50, h0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    v_16, v_50, v_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    best_fit = [fcc_50, dM_50, h0_50, Om_50, v_50]
    DOF = len(z_cmb) + N_cc - len(prior.keys)

    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"v: {v_50:.2f} +{(v_84 - v_50):.2f} -{(v_50 - v_16):.2f} km/s")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"DOF: {DOF}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / fcc_50,
        label=legend_cc,
    )
    plot_sn_predictions(
        legend=legend_sn,
        x=z_cmb,
        y=mu_vals - mu_corr(v_50, DM_grid(best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(dM_50, DM_grid(best_fit)),
        label=f"$Ω_m$={Om_50:.4f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ---------------- Flat ΛCDM ----------------
# ΔM: -0.079 +- 0.075 mag
# H0: 66.8 +- 2.5 km/s/Mpc
# Ωm: 0.333 +- 0.022
# f_cc: 1.51 +- 0.17
# Chi squared: 65.89
# Log evidence: -175.6
# DOF: 56
# -------------------------------------------


# ---------------- Flat ΛCDM ----------------
# velocity step correction SNe observed redshifts (turning point z <= 0.2 inflow z > 0.2 outflow)
# z_cosmo = -1 + (1 + z) / (1 + v/c)
# v: -291 +- 120 km/s (prior U[-900, 900])
# ΔM: -0.047 +- 0.075 mag
# H0: 68.5 +- 2.7 km/s/Mpc
# Ωm: 0.306 +- 0.023
# f_cc: 1.50 +- 0.17
# Chi squared: 59.37 (2.55 sigma significance)
# Log evidence: -174.3 (ΔlogZ = 1.3 in favour of v corrections)
# DOF: 55
# -------------------------------------------


# ---------------- Flat wCDM ----------------
# ΔM: -0.062 +- 0.081 mag
# H0: 67.0 +- 2.6 km/s/Mpc
# Ωm: 0.304 +0.049 -0.038
# w0: -0.92 +0.13 -0.11 (prior U[-1.5, -0.5])
# f_cc: 1.49 +- 0.17
# Chi squared: 64.47 (1.19 sigma significance)
# Log evidence: -176.5 (ΔlogZ = -0.9 in favour of ΛCDM)
# DOF: 55
# -------------------------------------------


# --------------- Flat w0waCDM --------------
# w0 + wa < 0 enforced in the likelihood
# ΔM: -0.085 +- 0.081 mag
# H0: 65.9 +- 2.6 km/s/Mpc
# Ωm: 0.365 +0.052 -0.024
# w0: -0.81 +0.14 -0.12 (prior U[-3, 1])
# wa: < -1.65 (prior U[-3, 2] truncated posterior)
# f_cc: 1.50 +- 0.17
# Chi squared: 62.90 (1.22 sigma significance)
# Log evidence: -177.4 + 0.3 (ΔlogZ = -1.5 in favour of ΛCDM)
# DOF: 54
# -------------------------------------------
