from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite, interp_pchip
from y2025DESdovekie.data import get_data as get_sn_data, effective_sample_size
from y2025BAO.data_fs_lya import get_data as get_bao_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1 + w0 + (1 - w0) * zp1**3)) ** 2  # wzCDM
    # return 1  # ΛCDM
    # return zp1 ** (3 * (1 + w0))  # wCDM
    # return zp1 ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / zp1)  # w0waCDM


@njit
def Ez(z, h, Obh2, Och2):
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0 = params[1]
    return H0 * Ez(z, h=H0 / 100, Obh2=params[2], Och2=params[3])


cmb.set_HZ(H_z)


@njit
def DM_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return (cum_dm, dh_grid)


dv_rs = 0
dm_rs = 1
dh_rs = 2
f_ap = 3
qty_map = {
    "DV_over_rs": dv_rs,
    "DM_over_rs": dm_rs,
    "DH_over_rs": dh_rs,
    "F_AP": f_ap,
}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params, DM_interp):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnuh2
    RD = cmb.r_drag(Obh2, Omh2)
    DM = interp_hermite(z, z_grid, *DM_interp)
    DH = interp_pchip(z, z_grid, DM_interp[1])

    DV_MASK = qty == dv_rs
    DM_MASK = qty == dm_rs
    DH_MASK = qty == dh_rs
    FAP_MASK = qty == f_ap
    theory = np.empty(z.size, dtype=np.float64)

    theory[DH_MASK] = DH[DH_MASK] / RD
    theory[DM_MASK] = DM[DM_MASK] / RD
    theory[DV_MASK] = (z[DV_MASK] * DH[DV_MASK] * DM[DV_MASK] ** 2) ** (1 / 3) / RD
    theory[FAP_MASK] = DM[FAP_MASK] / DH[FAP_MASK]
    return theory


@njit
def mu_corr(v_100, DM_interp):
    # Heaviside step at z = 0.10563
    v_km_s = 100 * v_100 * np.where(z_cmb <= 0.10563, 1, -1)
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + v_km_s / c)

    DM_cosmo = interp_hermite(z_cosmo, z_grid, *DM_interp)
    DM_obs = interp_hermite(z_cmb, z_grid, *DM_interp)
    return 5 * np.log10(DM_cosmo / DM_obs)


@njit
def theory_mu(offset, DM_interp):
    DM = interp_hermite(z_cmb, z_grid, *DM_interp)
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi2_sn(params, DM_interp):
    delta = mu_values - theory_mu(params[0], DM_interp) - mu_corr(params[4], DM_interp)
    return solve_triang(cho_sn, delta)


@njit
def chi2_cmb(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    return delta @ cmb.inv_cov_mat @ delta


@njit
def chi2_bao(params, DM_interp):
    delta_bao = bao["value"] - bao_theory(bao["z"], bao_qty, params, DM_interp)
    return delta_bao @ inv_cov_bao @ delta_bao


def chi_squared(params):
    DM_interp = DM_grid(params)
    return chi2_cmb(params) + chi2_bao(params, DM_interp) + chi2_sn(params, DM_interp)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("dM", dist=(-0.5, +0.5))
    prior.add_parameter("H0", dist=(60.0, 75.0))
    prior.add_parameter("obh2", dist=(0.010, 0.030))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("v", dist=(-5.5, 2.5))  # x100 km/s

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=["ΔM", "H_0", "ω_b", "ω_c", "v_{100}"],
    )
    gd_samples.addDerived(
        gd_samples["obh2"] + gd_samples["och2"] + Omnuh2, name="omh2", label="ω_m"
    )
    gd_samples.addDerived(
        gd_samples["omh2"] / (gd_samples["H0"] / 100) ** 2, name="om", label="Ω_m"
    )
    gd_samples.addDerived(
        cmb.z_star(gd_samples["obh2"], gd_samples["omh2"]), name="zstar", label="z_*"
    )
    gd_samples.addDerived(
        cmb.z_drag(gd_samples["obh2"], gd_samples["omh2"]),
        name="zdrag",
        label="z_{drag}",
    )
    gd_samples.addDerived(
        cmb.r_drag(gd_samples["obh2"], gd_samples["omh2"]),
        name="rdrag",
        label="r_{drag}",
    )

    plot_params = ["H0", "rdrag", "om", "v", "dM"]
    plots.get_subplot_plotter().triangle_plot(
        gd_samples, params=plot_params, title_limit=1, contour_colors=["C0"]
    )
    plt.show()

    best_fit = gd_samples.mean(prior.keys)
    DOF = effective_sample_size + len(bao) + len(cmb.DISTANCE_PRIORS) - len(prior.keys)

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    map_index = np.argmax(log_l)
    map_params = samples[map_index]
    print(f"χ2 (MAP): {chi_squared(map_params):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {DOF}")

    best_fit_dm = DM_grid(best_fit)

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(
            z, qty, best_fit, best_fit_dm
        ),
        data=bao,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(best_fit[4], best_fit_dm),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit[0], best_fit_dm),
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

# """
# *********************************
# BAO: DESI DR2 + FS Lya
# SNe1A: DES5Y Dovekie
# CMB: ACT DR6 + Planck compression (R, π/θ*, ωb) 
# *********************************


# ----------- Flat ΛCDM -----------
# H0: 68.26 ± 0.26 km/s/Mpc
# r_d: 147.49 ± 0.19 Mpc
# Ωm: 0.3025 ± 0.0035
# ΔM: -0.0644 ± 0.0078

# ωb: 0.02255 ± 0.00010
# ωc: 0.11776 ± 0.00063
# ωm: 0.14096 ± 0.00062
# z*: 1089.47 ± 0.15
# z_d: 1060.19 ± 0.23
# χ2 (MAP): 1651.56
# Log evidence: -843.9
# Degrees of freedom: 1727
# ---------------------------------


# ----------- Flat ΛCDM -----------
# velocity step correction in SNe observed redshifts
# (turning point z <= 0.10563 inflow z > 0.10563 outflow)
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# H0: 68.36 ± 0.26 km/s/Mpc
# r_d: 147.54 ± 0.19 Mpc
# Ωm: 0.3012 ± 0.0035
# v: -1.56 ± 0.55 (prior ~ U(-5.5, 2.5)) x 100 km/s
# v / (z_turn=0.10563): -1477 ± 521 km/s
# ΔM: -0.0637 ± 0.0078 mag

# ωb: 0.02257 ± 0.00010
# ωc: 0.11753 ± 0.00064
# ωm: 0.14074 ± 0.00063
# z*: 1089.43 ± 0.15
# z_d: 1060.20 ± 0.23
# χ2 (MAP): 1643.55 (2.87 sigma significance)
# Log evidence: -841.7 (Δ logZ = 2.83 in favour of velocity step correction)
# Degrees of freedom: 1726
# ---------------------------------


# ----------- Flat wCDM -----------
# H0: 67.66 ± 0.53 km/s/Mpc
# r_d: 147.63 ± 0.22 Mpc
# Ωm: 0.3066 ± 0.0048
# w0: -0.972 ± 0.022 (prior U(-4/3, -2/3))
# ΔM: -0.073 ± 0.010 mag

# ωb: 0.02258 ± 0.00011
# ωc: 0.11712 ± 0.00082
# ωm: 0.14034 ± 0.00079
# z*: 1089.37 ± 0.17
# z_d: 1060.20 ± 0.23
# χ2 (MAP): 1649.86 (1.30 sigma away from ΛCDM)
# Log evidence: -845.6 (Δ logZ = -1.7 in favour of ΛCDM)
# Degrees of freedom: 1726
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
# H0: 67.20 ± 0.53 km/s/Mpc
# r_d: 147.62 ± 0.20 Mpc
# Ωm: 0.3108 ± 0.0051
# w0: -0.908 ± 0.040 (prior U(-1, -1/3))
# wa: computed from w0
# ΔM: -0.0749 ± 0.0091 mag

# ωb: 0.02259 ± 0.00010
# ωc: 0.11712 ± 0.00069
# ωm: 0.14035 ± 0.00068
# z*: 1089.36 ± 0.16
# z_d: 1060.21 ± 0.23
# χ2 (MAP): 1646.96 (2.14 sigma away from ΛCDM)
# Log evidence: -843.5 (Δ logZ = 0.4 in favour of wzCDM)
# Degrees of freedom: 1726
# ---------------------------------


# ----------- Flat w0waCDM --------
# H0: 67.38 ± 0.55 km/s/Mpc
# r_d: 147.26 ± 0.25 Mpc
# Ωm: 0.3127 ± 0.0054
# w0: -0.834 ± 0.056 (prior U(-1.5, 0.0))
# wa: -0.57 +0.24 -0.20 (prior U(-2.5, 1.0))
# ΔM: -0.059 ± 0.012 mag

# ωb: 0.02251 ± 0.00011
# ωc: 0.11879 ± 0.00098
# ωm: 0.14195 ± 0.00095
# z*: 1089.61 ± 0.19
# z_d: 1060.17 ± 0.23
# χ2 (MAP): 1642.56 (2.54 sigma away from ΛCDM)
# Log evidence: -844.5 + 0.1 (Δ logZ = -0.5 in favour of ΛCDM)
# Degrees of freedom: 1725
# ---------------------------------
