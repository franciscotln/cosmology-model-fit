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
def chi_squared(params, f_cc_arr):
    offset, v = params[2], params[5]

    DM_interp = DM_grid(params)
    delta_sn = mu_vals - mu_theory(offset, DM_interp) - mu_corr(v, DM_interp)
    y_sn = solve_triangular(L_sn, delta_sn)
    chi_sn = np.dot(y_sn, y_sn)

    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    y_cc = solve_triangular(L_cc, f_cc_arr * cc_delta)
    chi_cc = np.dot(y_cc, y_cc)

    return chi_sn + chi_cc


@njit
def log_likelihood(params):
    f_cc_arr = np.exp(params[0]) * (1.0 + z_cc_vals)**params[1]
    if np.any(f_cc_arr <= 1e-4):
        return -np.inf

    logdet = logdet_cc - 2.0 * np.log(f_cc_arr).sum()
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet

    return -0.5 * (chi_squared(params, f_cc_arr) + normalization_cc)


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
    prior.add_parameter("ln_f0", dist=(-0.5, 2.5))
    prior.add_parameter("n", dist=(-4.0, 4.0))
    prior.add_parameter("dM", dist=(-1.0, 1.0))
    prior.add_parameter("H0", dist=(40.0, 95.0))
    prior.add_parameter("Om", dist=(0.1, 0.7))
    prior.add_parameter("v", dist=(-900, 900))

    with Pool(5) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    log_evd = sampler.log_z
    labels = ["ln(f_0)", "n", "ΔM", "H_0", "Ω_m", "v"]

    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    plots.get_subplot_plotter().triangle_plot(
        gd_samples, prior.keys, title_limit=1, contour_colors=["C0"]
    )
    plt.show()

    best_fit = samples[np.argmax(log_l)]
    DOF = len(z_cmb) + N_cc - len(prior.keys)

    f_array = np.exp(best_fit[0]) * (1.0 + z_cc_vals)**best_fit[1]

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
        y=mu_vals - mu_corr(best_fit[5], DM_grid(best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit[2], DM_grid(best_fit)),
        label=f"$Ω_m$={best_fit[4]:.4f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ---------------- Flat ΛCDM ----------------
# H0 = 67.7 +- 1.9 km/s/Mpc
# Ωm = 0.326 +- 0.021
# ΔM = -0.053 +0.061 -0.055 mag
# ln(f0) = 1.11 +0.26 -0.23
# n = -1.29 +- 0.46
# Chi squared: 67.90
# Log evidence: -178.6
# DOF: 56
# -------------------------------------------


# ---------------- Flat ΛCDM ----------------
# velocity step correction SNe observed redshifts
# turning point z <= 0.2 inflow z > 0.2 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)
#
# v = -308 +- 120 km/s
# H0 = 68.4 +- 1.9 km/s/Mpc
# Ωm = 0.299 +- 0.022
# ΔM = -0.055 +- 0.058
# ln(f0) = 1.14 +0.26 -0.23
# n = -1.35 +- 0.46
# Chi squared: 61.97 (2.4 sigma significance)
# Log evidence: -176.9 (ΔlogZ = 1.7 in favour of v corrections)
# DOF: 55
# -------------------------------------------


# ---------------- Flat wCDM ----------------
# w0 = -0.88 +0.13 -0.12 (prior U[-1.5, -0.5])
# H0 = 67.3 +- 1.9 km/s/Mpc
# Ωm = 0.283 +0.059 -0.041
# ΔM = -0.058 +- 0.059
# ln(f0) = 1.13 +0.27 -0.24
# n = -1.36 +- 0.48
# Chi squared: 66.82 (1.0 sigma significance)
# Log evidence: -179.3 (ΔlogZ = -0.7 in favour of ΛCDM)
# DOF: 55
# -------------------------------------------


# --------------- Flat w0waCDM --------------
# w0 + wa < 0 enforced in the likelihood
# H0: 66.4 +- 1.9 km/s/Mpc
# Ωm: 0.363 +0.048 -0.023
# w0: -0.78 +0.14 -0.12 (prior U[-3, 1])
# wa: < -1.87 (prior U[-3, 2] truncated posterior)
# ΔM: -0.070 +- 0.059 mag
# ln(f0) = 1.16 +0.26 -0.23
# n = -1.38 +- 0.46
# Chi squared: 64.28 (1.39 sigma significance)
# Log evidence: -179.7 + 0.3 (ΔlogZ = -0.8 in favour of ΛCDM)
# DOF: 54
# -------------------------------------------
