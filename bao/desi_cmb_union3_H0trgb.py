from numba import njit
import numpy as np
from interpolator import interp_hermite
from y2026union3_1.data import get_data
from y2025BAO.data import get_data as get_bao_data
from y20116dFBAO.data import get_data as get_6dF_bao_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
sixdF_bao_legend, sixdF_bao_data, sixdF_bao_cov_matrix = get_6dF_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(bao_cov_matrix)
inv_cov_6dF_bao = np.linalg.inv(sixdF_bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, 4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence with w(z) ranging from -1 to 1
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


@njit
def Ez(z, H0, Obh2, Och2):
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

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0 = params[1]
    return H0 * Ez(z, H0, Obh2=params[2], Och2=params[3])


cmb.set_HZ(H_z)


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
qty_desi = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)
qty_6dF = np.array([qty_map[q] for q in sixdF_bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnuh2
    rd = cmb.r_drag(Obh2, Omh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


pivot_mask = z_cmb <= 0.2


@njit
def mu_corr(params):
    z_pec = 100 * params[4] / c
    z_cosmo1 = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    z_cosmo2 = -1.0 + (1.0 + z_cmb) / (1.0 - z_pec)

    DM_ref = DM_z(z_cmb, params)

    return np.where(
        pivot_mask,
        5.0 * np.log10(DM_z(z_cosmo1, params) / DM_ref),
        5.0 * np.log10(DM_z(z_cosmo2, params) / DM_ref),
    )


@njit
def mu_theory(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


@njit
def chi2_sn(params):
    delta_sn = mu_vals - mu_theory(params) - mu_corr(params)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_bao(params):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], qty_desi, params)
    chi2_desi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_bao_6dF = sixdF_bao_data["value"] - bao_theory(
        sixdF_bao_data["z"], qty_6dF, params
    )
    chi2_6dF_bao = delta_bao_6dF @ inv_cov_6dF_bao @ delta_bao_6dF
    return chi2_desi_bao + chi2_6dF_bao


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb
    chi2_H0 = ((params[1] - 70.39) / 1.80) ** 2  # TRGB arXiv:2408.06153v3

    return chi2_H0 + chi2_cmb + chi2_bao(params) + chi2_sn(params)


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
    prior.add_parameter("dM", dist=(-1.0, +1.0))
    prior.add_parameter("H0", dist=(60.0, 75.0))
    prior.add_parameter("obh2", dist=(0.010, 0.030))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("v", dist=(-9.5, 3.5))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    labels = ["ΔM", "H_0", "ω_b", "ω_c", "v"]
    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
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

    plots.get_subplot_plotter().triangle_plot(
        gd_samples, params=prior.keys, title_limit=1, contour_colors=["C0"]
    )
    plt.show()

    best_fit = gd_samples.mean(prior.keys)
    degs_of_freedom = (
        1
        + len(z_cmb)
        + len(sixdF_bao_data)
        + len(bao_data)
        + len(cmb.DISTANCE_PRIORS)
        - len(best_fit)
    )

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    index_MAP = np.argmax(log_l)
    print(f"χ2 (MAP): {chi_squared(samples[index_MAP]):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"Degrees of freedom: {degs_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=sixdF_bao_data,
        errors=np.sqrt(np.diag(sixdF_bao_cov_matrix)),
        title=sixdF_bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
Union 3.1 SNe 2026
H0_TRGB prior: ~ N(70.39, 1.80) km/s/Mpc
Compressed Planck + ACT
DESI BAO DR2 2025
6dF BAO 2011
"""


"""
Flat ΛCDM w(z) = -1
ΔM: -0.049 ± 0.007 mag
H0: 68.47 ± 0.27 km/s/Mpc
ωb: 0.02258 ± 0.00010
ωc: 0.11730 ± 0.00064
ωm: 0.14053 ± 0.00063
Ωm: 0.2998 ± 0.0035
z*: 1089.39 ± 0.15
z_d: 1060.21 ± 0.23
r_d: 147.58 ± 0.19 Mpc
χ2 (MAP): 46.56
Log evidence: -42.3
Degrees of freedom: 36
"""


"""
Flat ΛCDM w(z) = -1
Isotropic velocity SNe observed redshifts (turning point z <= 0.2 inflow z > 0.2 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

ΔM: -0.0497 ± 0.0069 mag
v: -3.1 ± 1.0 (prior U(-9.5, 3.5)) x 100 km/s
v / (z_cut=0.2): -1550 ± 500 km/s
H0: 68.53 ± 0.27 km/s/Mpc
ωb: 0.02259 ± 0.00010
ωc: 0.11716 ± 0.00065
ωm: 0.14040 ± 0.00063
Ωm: 0.2990 ± 0.0035
z*: 1089.36 ± 0.15
z_d: 1060.22 ± 0.23
r_d: 147.61 ± 0.19 Mpc
χ2 (MAP): 37.87 (2.95 sigma significance)
Log evidence: -39.5 (ΔlogZ = 2.8 in favour of in/outflow corrections)
Degrees of freedom: 35
"""


"""
Flat wCDM w(z) = w0
ΔM: -0.050 ± 0.010 mag
H0: 68.42 ± 0.64 km/s/Mpc
ωb: 0.02258 ± 0.00011
ωc: 0.1173 ± 0.0008
w0: -0.998 ± 0.026 (prior U(-1.3, -0.5))
ωm: 0.1405 ± 0.0008
Ωm: 0.300 ± 0.005
z*: 1089.38 ± 0.17
z_d: 1060.21 ± 0.23
r_d: 147.59 ± 0.22
χ2 (MAP): 46.56 (same as ΛCDM since w0 is consistent with -1)
Log evidence: -44.8 (ΔlogZ = -2.5 in favour of ΛCDM)
Degrees of freedom: 35
"""


"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
ΔM: -0.057 ± 0.008 mag
H0: 67.57 +0.70 -0.51 km/s/Mpc
ωb: 0.02260 ± 0.00010
ωc: 0.1169 ± 0.0007
w0: -0.926 +0.030 -0.061 (prior U(-1.0, -1/3))
ωm: 0.1401 ± 0.0007
Ωm: 0.307 ± 0.006
z*: 1089.32 ± 0.16
z_d: 1060.23 ± 0.23
r_d: 147.67 ± 0.20 Mpc
χ2 (MAP): 45.46 (1.05 sigma away from ΛCDM)
Log evidence: -43.4 (ΔlogZ = -1.1 in favour of ΛCDM)
Degrees of freedom: 35
"""


"""
Flat w(z) = w0 + wa * z / (1 + z)
ΔM: -0.042 ± 0.010 mag
H0: 67.47 ± 0.73 km/s/Mpc
ωb: 0.02252 ± 0.00011
ωc: 0.11877 ± 0.00097
w0: -0.804 ± 0.077 (prior U(-1.5, 0.0))
wa: -0.70 +0.29 -0.25 (prior U(-2.5, 1.0))
ωm: 0.1419 ± 0.0009
Ωm: 0.3119 ± 0.0071
z*: 1089.60 ± 0.19
z_d: 1060.19 ± 0.23
r_d: 147.26 ± 0.25 Mpc
χ2 (MAP): 39.33 (2.21 sigmas away from ΛCDM)
Log evidence: -43.3 (ΔlogZ = -1.0 in favour of ΛCDM)
Degrees of freedom: 34
"""
