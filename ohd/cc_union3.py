from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2026union3_1.data import get_data as get_sn_data
from y2005cc.data_no_loubser import get_data as get_cc_data

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
    Omh2, Om = params[3], params[4]
    h2 = Omh2 / Om
    return 100 * np.sqrt(Omh2 * (1.0 + z) ** 3 + (h2 - Omh2))


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


@njit
def get_fz(params):
    z_pivot = 1.035
    fp, n = np.exp(params[0]), params[1]
    return fp * ((1.0 + z_cc) / (1.0 + z_pivot)) ** n


@njit
def log_likelihood(params):
    cov_mat_cc = cov_mat_cc_sys + np.diag(diag_stat**2 * get_fz(params)**2)
    L_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2 * np.sum(np.log(np.diag(L_cc)))
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc

    return -0.5 * (chi_squared(params, L_cc) + normalization_cc)


def main():
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from ohd.plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("ln_fp", dist=(-2, 1))
    prior.add_parameter("n", dist=(-3, 7))
    prior.add_parameter("dM", dist=(-1.0, 1.0))
    prior.add_parameter("Omh2", dist=(0.01, 0.25))
    prior.add_parameter("Om", dist=(0.1, 0.7))
    prior.add_parameter("dz1000", dist=(-3.5, 3.5)) # 1000 x Δz

    with Pool(5) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    log_evd = sampler.log_z
    labels = ["ln(f_{pivot})", "n", "ΔM", "Ω_m h^2", "Ω_m", "1000 Δz"]

    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=-log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(100 * np.sqrt(gd_samples["Omh2"] / gd_samples["Om"]), name="H0", label="H_0")
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
        params=["H0"] + prior.keys,
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
# H0 = 66.9 ± 2.8 km/s/Mpc
# Ωm = 0.332 ± 0.021
# Ωm h^2 = 0.149 +0.012 -0.013
#
# ΔM = -0.076 ± 0.088 mag
# n = 3.06 +0.86 -1.20
# ln(fp) = -0.49 ± 0.27
# fp = 0.64 +0.12 -0.19
#
# χ² (MAP): 63.55
# Log likelihood (MAP): -151.41
# Log evidence: -164.2
# DOF: 53
# -------------------------------------------


# ---------------- Flat ΛCDM ----------------
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.2 positive z > 0.2 negative
# z_cosmo = z_cmb ± Δz
#
# 1000 Δz = 1.03 ± 0.43 (prior ~ U[-3.5, 3.5])
# H0 = 68.0 ± 2.9 km/s/Mpc
# Ωm = 0.310 ± 0.022
# Ωm h^2 = 0.143 ± 0.013
#
# ΔM = -0.059 ± 0.087 mag
# n = 3.05 +0.87 -1.20
# ln(fp) = -0.48 ± 0.27
# fp = 0.64 +0.12 -0.20
#
# χ² (MAP): 57.56
# Log likelihood (MAP): -148.47
# Log evidence: -163.2
# DOF: 52
# -------------------------------------------


# ---------------- Flat wCDM ----------------
# w0 = -0.92 +0.13 -0.12 (prior ~ U[-2, 0])
# H0 = 66.9 ± 2.8 km/s/Mpc
# Ωm h^2 = 0.135 +0.024 -0.019
# Ωm = 0.303 +0.053 -0.039
#
# ΔM = -0.069 ± 0.089 mag
# n = 3.06 +0.87 -1.20
# ln(fp) = -0.47 ± 0.27
# fp = 0.65 +0.13 -0.20
#
# χ² (MAP): 63.32
# Log likelihood (MAP): -151.12
# Log evidence: -165.8
# DOF: 52
# -------------------------------------------


# --------------- Flat w0waCDM --------------
# w0 + wa < 0 enforced in the likelihood
#
# H0 = 65.9 ± 2.9 km/s/Mpc
# Ωm = 0.366 +0.057 -0.030
# Ωm h^2 = 0.159 +0.024 -0.017
# w0 = -0.84 ± 0.14 (prior ~ N(-1.0, 0.5^2))
# wa = -1.5 ± 1.1 (prior ~ N(0.0, 1.5^2))
#
# ΔM = -0.088 ± 0.089 mag
# n = 3.13 +0.86 -1.20
# ln(fp) = -0.47 ± 0.27
# fp = 0.65 +0.13 -0.20
#
# χ² (MAP): 56.35
# Log likelihood (MAP): -148.88
# Log evidence: -164.9
# DOF: 51
# -------------------------------------------
