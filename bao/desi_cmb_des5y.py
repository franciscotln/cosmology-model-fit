from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2025DESdovekie.data import get_data as get_sn_data, effective_sample_size
from y2025BAO.data_fs_lya import get_data as get_bao_data
import cmb.data_spt_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]
cho_cmb = cho_factor(cmb.covariance, lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0, wa):
    # w1w2CDM
    zp1 = 1. + z
    return zp1**(3 * (1. + w0 + wa)) * ((zp1**2 + 1) / (2 * zp1**2))**(3 * wa)


@njit
def H_z(z, params):
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
    dark_energy_term = Ode * Ode_z(z, params[4], params[5])

    return H0 * np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


cmb.set_HZ(H_z)


@njit
def DM_DH_grid(params):
    dh_grid = c/ H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    dm_grid = np.zeros(z_grid.size, dtype=np.float64)
    dm_grid[1:] = np.cumsum(dh * dz)
    return (dm_grid, dh_grid)


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
    inv_rd = 1 / cmb.r_drag(Obh2, Omh2)

    DM = interp_hermite(z, z_grid, y=DM_interp[0], y_prime=DM_interp[1])
    DH = interp_pchip(z, z_grid, y=DM_interp[1])

    DV_MASK = qty == dv_rs
    DM_MASK = qty == dm_rs
    DH_MASK = qty == dh_rs
    FAP_MASK = qty == f_ap
    result = np.empty(z.size, dtype=np.float64)

    result[DH_MASK] = DH[DH_MASK] * inv_rd
    result[DM_MASK] = DM[DM_MASK] * inv_rd
    result[DV_MASK] = (z[DV_MASK] * DH[DV_MASK] * DM[DV_MASK] ** 2) ** (1 / 3) * inv_rd
    result[FAP_MASK] = DM[FAP_MASK] / DH[FAP_MASK]
    return result


@njit
def get_z_cosmo(dz_1000):
    return z_cmb
    # Heaviside step at z = 0.10563
    # z_offset = 1e-03 * dz_1000 * np.where(z_cmb <= 0.10563, 1, -1)
    # return z_cmb + z_offset


def mu_corr(dz_1000, dm_interp):
    return 0.0
    # For plotting purposes only
    z_cosmo = get_z_cosmo(dz_1000)
    DM_cosmo = interp_hermite(z_cosmo, z_grid, *dm_interp)
    DM_obs = interp_hermite(z_cmb, z_grid, *dm_interp)
    return 5 * np.log10(DM_cosmo / DM_obs)


@njit
def theory_mu(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(params, dm_interp):
    z_cosmo = get_z_cosmo(dz_1000=params[4])
    DM_cosmo = interp_hermite(z_cosmo, z_grid, y=dm_interp[0], y_prime=dm_interp[1])
    delta = mu_values - theory_mu(params[0], DM_cosmo)
    y = solve_triangular(cho_sn, delta)
    return np.dot(y, y)


@njit
def chi2_cmb(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    y = solve_triangular(cho_cmb, delta)
    return np.dot(y, y)


@njit
def chi2_bao(params, dm_interp):
    delta_bao = bao["value"] - bao_theory(bao["z"], bao_qty, params, dm_interp)
    y = solve_triangular(cho_bao, delta_bao)
    return np.dot(y, y)


@njit
def chi_squared(params):
    dm_dh_grid = DM_DH_grid(params)
    return chi2_cmb(params) + chi2_bao(params, dm_dh_grid) + chi2_sn(params, dm_dh_grid)


def log_likelihood(params):
    if params[4] + params[5] >= 0:
        return -1e10
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
    prior.add_parameter("dz_1000", dist=(-1.5, 0.0))
    prior.add_parameter("wa", dist=(-2.5, 1.5))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    labels=["ΔM", "H_0", "ω_b", "ω_c", "1000 Δz", "w_a"]
    gd_samples = MCSamples(samples=samples, weights=np.exp(log_w), names=prior.keys, labels=labels)
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
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    plot_params = ["H0", "om", "rdrag", "dz_1000", "wa"]
    plots.get_subplot_plotter().triangle_plot(
        gd_samples, params=plot_params, title_limit=1, contour_colors=["C0"]
    )
    plt.show()

    best_fit = samples[np.argmax(log_l)]
    DOF = effective_sample_size + len(bao) + len(cmb.DISTANCE_PRIORS) - len(prior.keys)

    map_index = np.argmax(log_l)
    map_params = samples[map_index]
    print(f"χ2 (MAP): {chi_squared(map_params):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {DOF}")

    best_dm_dh_grid = DM_DH_grid(best_fit)

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit, best_dm_dh_grid),
        data=bao,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(best_fit[4], best_dm_dh_grid),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit[0], interp_hermite(z_cmb, z_grid, *best_dm_dh_grid)),
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# *********************************
# BAO: DESI DR2 + FS Lya
# SNe1A: DES5Y Dovekie
# CMB: ACT DR6 + Planck compression (R, π/θ*, ωb) 
# *********************************


# ----------- Flat ΛCDM -----------
# H0 = 67.98 ± 0.24 km/s/Mpc
# Ωm = 0.3061 ± 0.0033
# ωb = 0.022459 ± 0.000091
# ωc = 0.11835 ± 0.00058
# ωm = 0.14146 ± 0.00058
# z* = 1088.58 ± 0.11
# z_d = 1059.94 ± 0.21
# r_d = 147.45 ± 0.18 Mpc
# ΔM = -0.0712 ± 0.0074 mag
# χ2 (MAP): 1655.99
# Log evidence: -846.4
# Degrees of freedom: 1727
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.10563 positive z > 0.10563 negative
# z_cosmo = z_cmb ± Δz

# 1000 Δz = 0.52 ± 0.20 (prior ~ U[-1.5, 1.5])
# H0 = 68.06 ± 0.24 km/s/Mpc
# Ωm = 0.3050 ± 0.0033
# ωb = 0.022469 ± 0.000091
# ωc = 0.11816 ± 0.00059
# ωm = 0.14127 ± 0.00058
# z* = 1088.55 ± 0.11
# z_d = 1059.95 ± 0.20
# r_d = 147.48 ± 0.17 Mpc
# ΔM = -0.0716 ± 0.0074 mag
# χ2 (MAP): 1648.88 (2.7 sigma significance)
# Log evidence: -844.7 (Δ logZ = 1.7 in favour of z offset step correction)
# Degrees of freedom: 1726
# ---------------------------------


# ----------- Flat wCDM -----------
# H0 = 67.77 ± 0.53 km/s/Mpc
# Ωm = 0.3077 ± 0.0048
# ωb = 0.022467 ± 0.000093
# ωc = 0.11817 ± 0.00072
# w = -0.991 ± 0.021 (prior ~ U[-4/3, -2/3])
# ωm = 0.14128 ± 0.00070
# z* = 1088.56 ± 0.12
# z_d = 1059.94 ± 0.20
# r_d = 147.49 ± 0.20 Mpc
# ΔM = -0.074 ± 0.010 mag
# χ2 (MAP): 1655.78
# Log evidence: -848.8 (Δ logZ = -2.4 in favour of ΛCDM)
# DOF: 1726
# ---------------------------------


# ----------- Flat w0waCDM --------
# w0 + wa > 0 enforced in the likelihood
# Correction in prior volume: +0.2 to the evidence
# log((1.5 + 2.5)*1.5 / ((1.5 + 2.5)*1.5 - 0.5*1.5**2)) = 0.2
#
# H0 = 67.36 ± 0.55 km/s/Mpc
# Ωm = 0.3146 ± 0.0053
# ωb = 0.022412 ± 0.000094
# ωc = 0.11968 ± 0.00080
# w0 = -0.815 ± 0.056 (prior ~ U[-1.5, 0.0])
# wa = -0.71 +0.24 -0.20 (prior ~ U[-2.5, 1.5])
# ωm = 0.14273 ± 0.00078
# z* = 1088.72 ± 0.13
# z_d = 1059.94 ± 0.20
# r_d = 147.15 ± 0.21 Mpc
# ΔM = -0.058 ± 0.012 mag
# χ2 (MAP): 1643.76 (3.1 sigma significance)
# Log evidence: -845.5 + 0.2 (Δ logZ = 1.1 in favour of w0waCDM)
# Degrees of freedom: 1725
# ---------------------------------


# ----------- Flat w1w2CDM --------
# Enforced w1 + w2 < 0 in the likelihood
# (+0.2 to evidence from excluded volume)
#
# w(z) = w1 + w2 * ((1 + z)^2 - 1) / ((1 + z)^2 + 1)
#
# H0 = 67.39 ± 0.54 km/s/Mpc
# ωb = 0.022415 ± 0.000093
# ωc = 0.11969 ± 0.00080
# w1 = -0.826 ± 0.053 (prior ~ U[-1.5, 0.0])
# w2 = -0.58 +0.19 -0.17 (prior ~ U[-2.5, 1.5])
# ωm = 0.14275 ± 0.00078
# Ωm = 0.3144 ± 0.0053
# z* = 1088.72 ± 0.13
# z_d = 1059.94 ± 0.20
# r_d = 147.13 ± 0.21 Mpc
# ΔM = -0.057 ± 0.012 mag
# χ2 (MAP): 1643.80
# Log evidence: -845.7 + 0.2 (Δ logZ = 0.9 in favour of w1w2CDM)
# DOF: 1725
# ---------------------------------