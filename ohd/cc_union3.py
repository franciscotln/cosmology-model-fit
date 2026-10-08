from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2026union3_1.data import get_data as get_sn_data
from y2005cc.data import method, get_data as get_cc_data

legend_sn, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_sn_data()
legend_cc, z_cc, H_cc_vals, diag_stat, cov_mat_cc_sys = get_cc_data(split_sys=True)

N_cc = len(z_cc)
L_sn = cho_factor(cov_matrix_sn, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s

z_grid = np.linspace(0, np.max(z_cmb), num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0):
    return (1. + z)**(3 * (1. + w0))  # wCDM


@njit
def H_z(z, params):
    h, om = params[3], params[4]
    return 100 * h * np.sqrt(om * (1.0 + z) ** 3 + (1 - om))


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
def get_z_cosmo(dz1000):
    # Heaviside step at z = 0.2
    offset = 1e-3 * dz1000 * np.where(z_cmb <= 0.2, 1, -1)
    return z_cmb + offset


def mu_corr(dz1000, DM_interp):
    #  For plotting purposes only
    z_cosmo = get_z_cosmo(dz1000)
    return 5 * np.log10(DM_z(z_cosmo, DM_interp) / DM_z(z_cmb, DM_interp))


@njit
def mu_theory(mag_offset, DM):
    return mag_offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params, L_cc):
    z_cosmo = get_z_cosmo(dz1000=params[5])
    DM = DM_z(z_cosmo, DM_grid(params))

    delta_sn = mu_vals - mu_theory(mag_offset=params[2], DM=DM)
    y_sn = solve_triangular(L_sn, delta_sn)
    chi_sn = np.dot(y_sn, y_sn)

    cc_delta = H_cc_vals - H_z(z_cc, params)
    y_cc = solve_triangular(L_cc, cc_delta)
    chi_cc = np.dot(y_cc, y_cc)

    return chi_sn + chi_cc


method_f = method == "F"
z_pivot = 1.198
shape = np.ones_like(z_cc, dtype=np.float64)
shape[method_f] = ((1 + z_cc[method_f]) / (1 + z_pivot))**4


@njit
def get_fz(params):
    fp, n = np.exp(params[0]), params[1]
    return fp * shape**n


@njit
def log_likelihood_jit(params):
    cov_mat_cc = cov_mat_cc_sys + np.diag(diag_stat**2 * get_fz(params)**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2 * np.sum(np.log(np.diag(L_cc)))
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc

    return -0.5 * (chi_squared(params, L_cc) + normalization_cc)


def log_likelihood(params):
    return log_likelihood_jit(params)


def main():
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from ohd.plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("ln_fp", dist=(-2.0, 1.0))
    prior.add_parameter("n", dist=(-2.5, 2.5))
    prior.add_parameter("dM", dist=(-1.0, 1.0))
    prior.add_parameter("h", dist=(0.5, 1.0))
    prior.add_parameter("om", dist=(0.1, 0.7))
    prior.add_parameter("dz1000", dist=(-3.5, 3.5)) # 1e-03 x Δz

    with Pool(5) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    log_evd = sampler.log_z
    labels = ["ln(f_{pivot})", "n", "ΔM", "h", "Ω_m", "1000 Δz"]

    gd_samples = MCSamples(samples=samples, weights=np.exp(log_w), names=prior.keys, labels=labels)
    gd_samples.addDerived(gd_samples["om"] * gd_samples["h"]**2, name="omh2", label="Ω_m h^2")
    gd_samples.addDerived(np.exp(gd_samples["ln_fp"]), name="fp", label="f_{pivot}")

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    DOF = len(z_cmb) + N_cc - len(prior.keys)

    fz_cc = get_fz(best_fit)
    cov_mat_cc = cov_mat_cc_sys + np.diag(diag_stat**2 * fz_cc**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)

    print(f"χ² (MAP): {chi_squared(best_fit, L_cc):.2f}")
    print(f"Log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"DOF: {DOF}")

    plots.get_subplot_plotter().triangle_plot(
        gd_samples,
        params=["omh2"] + prior.keys,
        filled=True,
        title_limit=1,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_mat_cc_sys) + diag_stat**2),
        label=legend_cc,
        method=method,
        err_scaling=1 / fz_cc,
    )
    plot_sn_predictions(
        legend=legend_sn,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit[5], DM_grid(best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit[2], DM_z(z_cmb, DM_grid(best_fit))),
        label=f"$Ω_m$={best_fit[4]:.4f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ---------------- Flat ΛCDM ----------------
# h = 0.688 ± 0.015
# Ωm = 0.331 ± 0.021
# Ωm h^2 = 0.1564 ± 0.0090
#
# ΔM = -0.015 ± 0.043 mag
# n = 1.02 +0.31 -0.51
# ln(fp) = -0.40 ± 0.25
# fp = 0.69 +0.13 -0.19
#
# χ² (MAP): 67.05
# Log likelihood (MAP): -164.86
# Log evidence: -178.5
# DOF: 56
# -------------------------------------------


# ---------------- Flat ΛCDM ----------------
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.2 positive z > 0.2 negative
# z_cosmo = z_cmb ± Δz
#
# h = 0.696 ± 0.016
# Ωm = 0.307 ± 0.022
# Ωm h^2 = 0.1487 ± 0.0093
# 1000 Δz = 1.06 ± 0.43 (prior ~ U[-3.5, 3.5])
#
# ΔM = -0.010 ± 0.043 mag
# n = 1.03 +0.31 -0.54
# ln(fp) = -0.40 ± 0.25
# fp = 0.69 +0.13 -0.19
#
# χ² (MAP): 60.90
# Log likelihood (MAP): -161.90
# Log evidence: -177.4
# DOF: 55
# -------------------------------------------


# ---------------- Flat wCDM ----------------
# h = 0.686 ± 0.015
# Ωm = 0.289 +0.057 -0.041
# Ωm h^2 = 0.136 +0.027 -0.020
# w = -0.88 +0.14 -0.12 (prior ~ U[-2, 0])
#
# ΔM = -0.011 ± 0.043 mag
# n = 1.03 +0.31 -0.52
# ln(fp) = -0.38 ± 0.25
# fp = 0.70 +0.13 -0.20
#
# χ² (MAP): 64.17
# Log likelihood (MAP): -164.43
# Log evidence: -179.9
# DOF: 55
# -------------------------------------------


# --------------- Flat w0waCDM --------------
# w0 + wa < 0 enforced in the likelihood
#
# h = 0.678 ± 0.016
# Ωm = 0.361 +0.066 -0.030
# Ωm h^2 = 0.165 +0.028 -0.013
# w0 = -0.78 ± 0.15 (prior ~ U[-2, 0])
# wa = -1.8 +1.6 -1.3 (prior ~ N(0, 2^2))
#
# ΔM = -0.022 ± 0.043 mag
# n = 1.07 +0.33 -0.54
# ln(fp) = -0.39 ± 0.25
# fp = 0.70 +0.13 -0.19
#
# χ² (MAP): 63.35
# Log likelihood (MAP): -162.78
# Log evidence: -179.6
# DOF: 54
# -------------------------------------------
