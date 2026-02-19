from numba import njit
import numpy as np
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
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
des_bao_legend, des_bao_data, des_bao_cov_matrix = get_des_bao_data()
sixdF_bao_legend, sixdF_bao_data, sixdF_bao_cov_matrix = get_6dF_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(bao_cov_matrix)
inv_cov_des_bao = np.linalg.inv(des_bao_cov_matrix)
inv_cov_6dF_bao = np.linalg.inv(sixdF_bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, 3000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0):
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0 = params[1]
    return H0 * Ez(z, H0, Obh2=params[2], Och2=params[3], w0=params[4])


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
qty_des = np.array([qty_map[q] for q in des_bao_data["quantity"]], dtype=np.int32)
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


@njit
def mu_theory(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


@njit
def chi2_sn(params):
    delta_sn = mu_vals - mu_theory(params)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_bao(params):
    delta_bao_des = des_bao_data["value"] - bao_theory(
        des_bao_data["z"], qty_des, params
    )
    chi2_des_bao = delta_bao_des @ inv_cov_des_bao @ delta_bao_des
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], qty_desi, params)
    chi2_desi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_bao_6dF = sixdF_bao_data["value"] - bao_theory(
        sixdF_bao_data["z"], qty_6dF, params
    )
    chi2_6dF_bao = delta_bao_6dF @ inv_cov_6dF_bao @ delta_bao_6dF
    return chi2_desi_bao + chi2_des_bao + chi2_6dF_bao


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    return chi2_cmb + chi2_bao(params) + chi2_sn(params)


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
    prior.add_parameter("w0", dist=(-1.0, -1 / 3))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    labels = ["ΔM", "H_0", "ω_b", "ω_c", "w_0"]
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
        len(z_cmb)
        + len(sixdF_bao_data["z"])
        + len(des_bao_data["z"])
        + len(bao_data["z"])
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
        data=des_bao_data,
        errors=np.sqrt(np.diag(des_bao_cov_matrix)),
        title=des_bao_legend,
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
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
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
Evolving absolute mag of SNe M(z) = M_max + 0.2 * p / (1 + (z / z_c))
where z_c = 0.043
Equivalently M(z) = M0 + M'0 * z / (1 + (z / z_c))

ΔM_max: -0.064 ± 0.009 mag
p: 0.66 ± 0.29 (prior U(-1.0, 2.5))
H0: 68.51 ± 0.27 km/s/Mpc
Ωm: 0.2992 ± 0.0036
ωb: 0.02258 ± 0.00010
ωc: 0.1172 ± 0.0007
ωm: 0.1404 ± 0.0006
z*: 1089.37 ± 0.15
z_d: 1060.21 ± 0.23
r_d: 147.61 ± 0.19 Mpc
Chi2 (MAP): 41.05 (2.25 sigma away from constant M)
Log evidence: -41.1 (Δ logZ = 0.9 against constant M)
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
