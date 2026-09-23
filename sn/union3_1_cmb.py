from numba import njit
import numpy as np
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2026union3_1.data import get_data
import cmb.data_spt_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
L_sn = np.linalg.cholesky(cov_matrix_sn)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=2000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0):
    # wCDM
    return (1. + z)**(3 * (1. + w0))


@njit
def Hz(z, params):
    H0, Obh2, Och2 = params[1], params[2], params[3]
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode
    return H0 * np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


cmb.set_HZ(Hz)


@njit
def DM_z(z, dm_dh_grid):
    return interp_hermite(z, z_grid, dm_dh_grid[0], dm_dh_grid[1])


@njit
def DM_DH_grid(params):
    dh_grid = c / Hz(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return cum_dm, dh_grid


@njit
def get_z_cosmo(params):
    # Heaviside step at z = 0.2
    offset = 1e-3 * params[4] * np.where(z_cmb <= 0.2, 1, -1)
    return z_cmb + offset


def mu_corr(params):
    # For plotting purposes only (delta_z model, return 0 otherwise)
    dm_dh_grid = DM_DH_grid(params)
    z_cosmo = get_z_cosmo(params)
    DM_cosmo = DM_z(z_cosmo, dm_dh_grid)
    DM_obs = DM_z(z_cmb, dm_dh_grid)
    return 5 * np.log10(DM_cosmo / DM_obs)


@njit
def mu_theory(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(params):
    z_cosmo = get_z_cosmo(params)
    dm_cosmo = DM_z(z=z_cosmo, dm_dh_grid=DM_DH_grid(params))
    delta = mu_vals - mu_theory(offset=params[0], DM=dm_cosmo)
    y = solve_triangular(L_sn, delta)
    return np.dot(y, y)


@njit
def chi2_cmb(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    return delta_cmb @ cmb.inv_cov_mat @ delta_cmb


@njit
def chi_squared(params):
    return chi2_cmb(params) + chi2_sn(params)


@njit
def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from getdist import plots, MCSamples
    from matplotlib import pyplot as plt
    from sn.plotting import plot_predictions

    prior = Prior()
    prior.add_parameter("dM", dist=(-1.0, +1.0))
    prior.add_parameter("H0", dist=(60.0, 75.0))
    prior.add_parameter("obh2", dist=(0.01, 0.03))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("dz_1000", dist=(-3.5, 3.5)) # 1000 x Δz

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()

    labels=["ΔM", "H_0", "ω_b", "ω_c", "1000 Δz"]
    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        names=prior.keys,
        labels=labels,
        label="Union3.1 + CMB(θ*, ωb, ωm)",
    )
    gd_samples.addDerived(
        Omnuh2 + gd_samples["obh2"] + gd_samples["och2"], name="omh2", label="ω_m"
    )
    gd_samples.addDerived(
        gd_samples["omh2"] / (gd_samples["H0"] / 100) ** 2, name="om", label="Ω_m"
    )

    MAP_index = np.argmax(log_l)
    best_fit = samples[MAP_index]
    DOF = len(mu_vals) + len(cmb.DISTANCE_PRIORS) - len(best_fit)

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    print(f"Chi2 (MAP): {chi_squared(best_fit):.1f}")
    print(f"Log Evidence: {sampler.log_z:.1f}")
    print(f"DOF: {DOF}")

    g = plots.get_subplot_plotter()
    g.triangle_plot(
        gd_samples,
        params=["dM", "H0", "om", "dz_1000"],
        title_limit=1,
        filled=True,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    plot_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit[0], DM_z(z_cmb, DM_DH_grid(best_fit))),
        label=f"Ωm: {gd_samples['om'].mean():.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# *********************************
# Data sets:
# Union 3.1 (2026 - 22 bins)
# CMB(θ*, ωb, ωm) SPT + Planck + ACT compressed
# *********************************


# ----------- Flat ΛCDM -----------
# H0: 67.13 +- 0.37 km/s/Mpc
# Ωm: 0.3184 +- 0.0054
# ΔM: -0.0766 +- 0.0087 mag
# Chi2 (MAP): 28.7
# Log Evidence: -33.1
# DOF: 21
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.2 positive z > 0.2 negative
# z_cosmo = z_cmb ± Δz

# 1000 Δz = 0.97 ± 0.39 (prior ~ U[-3.5, 3.5])
# H0: 67.23 ± 0.38 km/s/Mpc
# Ωm: 0.3169 ± 0.0054
# ΔM: -0.0768 ± 0.0087 mag
# Chi2 (MAP): 22.6 (2.55 sigma significance)
# Log Evidence: -32.1 (Δ logZ = 1.0 in favour of z offset step correction)
# DOF: 20
# ---------------------------------


# ----------- Flat wCDM -----------
# w0: -0.965 ± 0.040 (prior ~ U[-1.5, -0.5])

# H0: 66.2 ± 1.1 km/s/Mpc
# Ωm: 0.327 ± 0.012
# ΔM: -0.090 ± 0.018 mag
# Chi2 (MAP): 27.7
# Log Evidence: -35.1
# DOF: 20
# ---------------------------------


# --------- Flat w0waCDM ----------
# w0: -0.74 ± 0.16 (prior ~ U[-3, 1])
# wa: -1.10 ± 0.74 (prior ~ U[-5, 5])

# H0: 67.5 +1.4 -1.2 km/s/Mpc
# Ωm: 0.316 +0.011 -0.014
# ΔM: -0.037 +0.041 -0.030 mag
# Chi2 (MAP): 26.0
# Log Evidence: -37.0
# DOF: 19

# -- at z_pivot = 0.25 (corr(wp, wa) = -0.05) --
# w_piv: -0.959 ± 0.041 (prior ~ U[-3, 1])
# wa: -1.10 ± 0.74 (prior ~ U[-5, 5])

# H0: 67.4 +1.4 -1.2 km/s/Mpc
# Ωm: 0.316 +0.011 -0.014
# ΔM: -0.037 +0.041 -0.030 mag
# Chi2 (MAP): 26.0
# Log Evidence: -37.0
# DOF: 19
# ---------------------------------
