from numba import njit
import numpy as np
from scipy.linalg import block_diag
from interpolator import interp_hermite
from y2026union3_1.data import get_data
from y2025BAO.data import get_data as get_bao_data
from y2024DESBAO.data import get_data as get_des_bao_data
from y20116dFBAO.data import get_data as get_6dF_bao_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
desi_legend, desi_bao_data, desi_bao_cov_matrix = get_bao_data()
des_legend, des_bao_data, des_bao_cov_matrix = get_des_bao_data()
sixdF_legend, sixdF_bao_data, sixdF_bao_cov_matrix = get_6dF_bao_data()

bao_data = np.concatenate((desi_bao_data, des_bao_data, sixdF_bao_data))
bao_cov_mat = block_diag(desi_bao_cov_matrix, des_bao_cov_matrix, sixdF_bao_cov_matrix)

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(bao_cov_mat)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, 4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
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
def DM_grid(params):
    dh_grid = DH_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return (cum_dm, dh_grid)


@njit
def DV_z(z, DM, params):
    DH = DH_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
bao_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params, DM):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnuh2
    rd = cmb.r_drag(Obh2, Omh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DM_mask] = DM[DM_mask]
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], DM[DV_mask], params)
    return results / rd


@njit
def chi2_bao(params, DM_interp):
    DM = interp_hermite(bao_data["z"], z_grid, *DM_interp)
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], bao_qty, params, DM)
    return delta_bao @ inv_cov_bao @ delta_bao


@njit
def mu_corr(params, DM_interp):
    # Heaviside step at z = 0.2
    v_km_s = 100 * params[4] * np.where(z_cmb <= 0.2, 1.0, -1.0)
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + v_km_s / c)

    DM_obs = interp_hermite(z_cmb, z_grid, *DM_interp)
    DM_cosmo = interp_hermite(z_cosmo, z_grid, *DM_interp)
    return 5.0 * np.log10(DM_cosmo / DM_obs)


@njit
def mu_theory(params, DM):
    return params[0] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(params, DM_interp):
    DM = interp_hermite(z_cmb, z_grid, *DM_interp)
    delta_sn = mu_vals - mu_theory(params, DM) - mu_corr(params, DM_interp)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_cmb(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    return delta_cmb @ cmb.inv_cov_mat @ delta_cmb


@njit
def chi_squared(params):
    DM, DM_prime = DM_grid(params)

    return (
        chi2_cmb(params)
        + chi2_bao(params, (DM, DM_prime))
        + chi2_sn(params, (DM, DM_prime))
    )


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
    prior.add_parameter("dM", dist=(-1, +1))  # mag
    prior.add_parameter("H0", dist=(60, 75))  # km/s/Mpc
    prior.add_parameter("obh2", dist=(0.01, 0.03))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("v", dist=(-10, 4))  # x 100 km/s

    with Pool(6) as pool:
        sampler = Sampler(
            prior,
            log_likelihood,
            n_live=10_000,
            pool=pool,
            seed=42,
            pass_dict=False,
            n_networks=5,
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
        len(z_cmb) + len(bao_data) + len(cmb.DISTANCE_PRIORS) - len(best_fit)
    )

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    index_MAP = np.argmax(log_l)
    print(f"χ2 (MAP): {chi_squared(samples[index_MAP]):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"Degrees of freedom: {degs_of_freedom}")

    DM_grid_best = DM_grid(best_fit)

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(
            z, qty, best_fit, interp_hermite(z, z_grid, *DM_grid_best)
        ),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_mat)),
        title="DESI + DES + 6dF BAO",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit, DM_grid_best),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit, interp_hermite(z_cmb, z_grid, *DM_grid_best)),
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
Union 3.1 SNe 2026
Compressed Planck + ACT
DESI BAO DR2 2025
DES BAO 2025
6dF BAO 2011
"""


"""
Flat ΛCDM w(z) = -1
ΔM: -0.050 ± 0.007 mag
H0: 68.44 ± 0.27 km/s/Mpc
Ωm: 0.3001 +0.0036 -0.0035
ωb: 0.02257 ± 0.00010
ωc: 0.1173 ± 0.0006
ωm: 0.1406 ± 0.0006
z*: 1089.40 ± 0.15
r*: 144.98 Mpc
z_d: 1060.20 ± 0.23
r_d: 147.58 ± 0.19 Mpc
Chi2 (MAP): 46.12
Log evidence: -42.0
Degs of freedom: 36
"""


"""
Flat ΛCDM w(z) = -1
Isotropic velocity SNe observed redshifts (turning point z <= 0.2 inflow z > 0.2 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

ΔM: -0.0502 ± 0.0070 mag
v: -3.1 ± 1.0 (prior U(-10, 4)) x 100 km/s
v / (z_cut=0.2): -1590 ± 500 km/s
H0: 68.51 ± 0.27 km/s/Mpc
Ωm: 0.2992 ± 0.0036
ωb: 0.02258 ± 0.00010
ωc: 0.11719 ± 0.00065
ωm: 0.1404 ± 0.0006
z*: 1089.38 ± 0.15
z_d: 1060.21 ± 0.23
r_d: 147.61 ± 0.19 Mpc
Chi2 (MAP): 37.51 (2.93 sigma significance)
Log evidence: -39.4 (Δ logZ = 2.6 in favour of flow corrections)
Degs of freedom: 35
"""


"""
Flat wCDM w(z) = w0
ΔM: -0.053 ± 0.010 mag
H0: 68.18 ± 0.68 km/s/Mpc
Ωm: 0.3020 ± 0.0057
ωb: 0.02259 ± 0.00011
ωc: 0.1171 ± 0.0009
ωm: 0.1404 ± 0.0008
w0: -0.989 ± 0.027 (prior U(-1.3, -0.5))
z*: 1089.37 ± 0.17
z_d: 1060.21 ± 0.23
r_d: 147.63 ± 0.22 Mpc
Chi2 (MAP): 45.92 (0.45 sigma away from ΛCDM)
Log evidence: -44.4 (Δ logZ = -2.4 in favour of ΛCDM)
Degs of freedom: 35
"""


"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
ΔM: -0.061 ± 0.009 mag
H0: 67.18 +0.78 -0.65 km/s/Mpc
Ωm: 0.3101 +0.0070 -0.0064
ωb: 0.02260 ± 0.00010
ωc: 0.1168 ± 0.0007
ωm: 0.1401 ± 0.0007
w0: -0.898 +0.048 -0.061 (prior U(-1.0, -1/3)
z*: 1089.32 ± 0.16
z_d: 1060.22 ± 0.23
r_d: 147.69 ± 0.20 Mpc
Chi2 (MAP): 43.60 (1.59 sigma away from ΛCDM)
Log evidence: -42.4 (Δ logZ = -0.4 in favour of ΛCDM)
Degs of freedom: 35
"""


"""
Flat w(z) = w0 + wa * z / (1 + z)
ΔM: -0.046 ± 0.011 mag
H0: 66.92 ± 0.78 km/s/Mpc
Ωm: 0.3171 ± 0.0078
ωb: 0.02252 ± 0.00011
ωc: 0.1188 ± 0.0010
ωm: 0.1420 ± 0.0009
w0: -0.760 ± 0.081 (prior U(-1.5, 0.0))
wa: -0.80 +0.30 -0.26 (prior U(-2.5, 1.0))
z*: 1089.61 ± 0.19
z_d: 1060.17 ± 0.23
r_d: 147.26 ± 0.25 Mpc
Chi2 (MAP): 36.70 (2.61 sigma away from ΛCDM)
Log evidence: -41.8 (Δ logZ = 0.2 | enforced wa + w0 < 0)
Degs of freedom: 34
"""
