from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2026union3_1.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data

legend_sn, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_sn_data()
legend_cc, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

L_sn = cho_factor(cov_matrix_sn, lower=True)[0]
L_cc = cho_factor(cov_matrix_cc, lower=True)[0]

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
    H0, Om = params[3], params[4]
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
def chi_squared(params, f_array):
    offset, v = params[2], params[5]

    DM_interp = DM_grid(params)
    delta_sn = mu_vals - mu_theory(offset, DM_interp) - mu_corr(v, DM_interp)
    chi_sn = solve_triangular(L_sn, delta_sn)

    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = solve_triangular(L_cc, f_array * cc_delta)

    return chi_sn + chi_cc


@njit
def log_likelihood(params):
    f_array = params[0] + params[1] * z_cc_vals
    if np.any(f_array <= 1e-4):
        return -np.inf

    logdet = logdet_cc - 2.0 * np.log(f_array).sum()
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet

    return -0.5 * (chi_squared(params, f_array) + normalization_cc)


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
    prior.add_parameter("f0", dist=(0.05, 4.0))
    prior.add_parameter("fa", dist=(-2.0, 2.0))
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
    labels = ["f_0", "f_a", "ΔM", "H_0", "Ω_m", "v"]

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

    f0_16, f0_50, f0_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    fa_16, fa_50, fa_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    h0_16, h0_50, h0_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    v_16, v_50, v_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)

    best_fit = samples[np.argmax(log_l)]
    DOF = len(z_cmb) + N_cc - len(prior.keys)

    f_array = best_fit[0] + best_fit[1] * z_cc_vals

    print(f"f0: {f0_50:.2f} +{(f0_84 - f0_50):.2f} -{(f0_50 - f0_16):.2f}")
    print(f"fa: {fa_50:.2f} +{(fa_84 - fa_50):.2f} -{(fa_50 - fa_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"v: {v_50:.2f} +{(v_84 - v_50):.2f} -{(v_50 - v_16):.2f} km/s")
    print(f"Chi squared: {chi_squared(best_fit, f_array):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"DOF: {DOF}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_array,
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
# ΔM: -0.069 +- 0.066 mag
# H0: 67.2 +- 2.2 km/s/Mpc
# Ωm: 0.329 +- 0.021
# f0: 2.25 +- 0.35
# fa: -0.79 +0.27 -0.32
# Chi squared: 68.65
# Log evidence: -179.0
# DOF: 56
# -------------------------------------------


# ---------------- Flat ΛCDM ----------------
# velocity step correction SNe observed redshifts (turning point z <= 0.2 inflow z > 0.2 outflow)
# z_cosmo = -1 + (1 + z) / (1 + v/c)
#
# v: -298 +- 120 km/s (prior U[-900, 900])
# ΔM: -0.060 +- 0.065 mag
# H0: 68.1 +- 2.2 km/s/Mpc
# Ωm: 0.303 +- 0.022
# f0: 2.27 +- 0.35
# fa: -0.82 +0.26 -0.32
# Chi squared: 62.05 (2.57 sigma significance)
# Log evidence: -177.5 (ΔlogZ = 1.5 in favour of v corrections)
# DOF: 55
# -------------------------------------------


# ---------------- Flat wCDM ----------------
# ΔM: -0.069 +- 0.067 mag
# H0: 66.9 +- 2.2 km/s/Mpc
# Ωm: 0.298 +0.052 -0.038
# w0: -0.91 +- 0.13 (prior U[-1.5, -0.5])
# f0: 2.23 +- 0.35
# fa: -0.79 +0.27 -0.31
# Chi squared: 66.67 (1.41 sigma significance)
# Log evidence: -179.9 (ΔlogZ = -0.9 in favour of ΛCDM)
# DOF: 55
# -------------------------------------------


# --------------- Flat w0waCDM --------------
# w0 + wa < 0 enforced in the likelihood
# ΔM: -0.078 +- 0.065 mag
# H0: 66.1 +- 2.1 km/s/Mpc
# Ωm: 0.372 +0.044 -0.022
# w0: -0.80 +0.14 -0.12 (prior U[-3, 1])
# wa: < -1.9 (prior U[-3, 2] truncated posterior)
# f0: 2.31 +- 0.35
# fa: -0.85 +0.26 -0.31
# Chi squared: 65.37 (1.22 sigma significance)
# Log evidence: -180.3 + 0.3 (ΔlogZ = -1.0 in favour of ΛCDM)
# DOF: 54
# -------------------------------------------
