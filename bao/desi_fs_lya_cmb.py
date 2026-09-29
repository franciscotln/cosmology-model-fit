from numba import njit
import numpy as np
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2025BAO.data_fs_lya import get_data
import cmb.data_early_lcdm_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

legend, bao, cov_mat = get_data()

L_bao = np.linalg.cholesky(cov_mat)
L_cmb = np.linalg.cholesky(cmb.covariance)

N_bao = len(bao)
N_cmb = len(cmb.DISTANCE_PRIORS)

logdet_bao = 2 * np.sum(np.log(np.diag(L_bao)))
logdet_cmb = 2 * np.sum(np.log(np.diag(L_cmb)))

norm_bao = N_bao * np.log(2 * np.pi) + logdet_bao
norm_cmb = N_cmb * np.log(2 * np.pi) + logdet_cmb

z_grid = np.linspace(0, np.max(bao["z"]) + 0.1, 4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0, wa):
    zp1 = 1.0 + z
    return zp1 ** (3 * (1.0 + w0 + wa)) * np.exp(-3 * wa * z / zp1)  # CPL


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0, wa = params
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0, wa)

    return H0 * np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


cmb.set_HZ(H_z)


@njit
def DM_DH_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    dm_grid = np.zeros(z_grid.size, dtype=np.float64)
    dm_grid[1:] = np.cumsum(dh * dz)
    return (dm_grid, dh_grid)


dv_rs = 0
dm_rs = 1
dh_rs = 2
f_ap = 3
qty_map = {"DV_over_rs": dv_rs, "DM_over_rs": dm_rs, "DH_over_rs": dh_rs, "F_AP": f_ap}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    dm_dh_grid = DM_DH_grid(params)

    Obh2, Och2 = params[1], params[2]
    Omh2 = Obh2 + Och2 + Omnuh2
    inv_rd = 1.0 / cmb.r_drag(Obh2, Omh2)

    DM = interp_hermite(z, x=z_grid, y=dm_dh_grid[0], y_prime=dm_dh_grid[1])
    DH = interp_pchip(z, x=z_grid, y=dm_dh_grid[1])

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == dv_rs
    DM_mask = qty == dm_rs
    DH_mask = qty == dh_rs
    FAP_mask = qty == f_ap
    results[DM_mask] = DM[DM_mask] * inv_rd
    results[DH_mask] = DH[DH_mask] * inv_rd
    results[DV_mask] = (z[DV_mask] * DH[DV_mask] * DM[DV_mask] ** 2) ** (1 / 3) * inv_rd
    results[FAP_mask] = DM[FAP_mask] / DH[FAP_mask]
    return results


@njit
def chi2_bao(params):
    delta = bao["value"] - bao_theory(bao["z"], bao_qty, params)
    y = solve_triangular(L_bao, delta)
    return np.dot(y, y)


@njit
def chi2_cmb(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[1], params[2], params)
    y = solve_triangular(L_cmb, delta)
    return np.dot(y, y)


@njit
def chi_squared(params):
    return chi2_cmb(params) + chi2_bao(params)


@njit
def log_likelihood(params):
    if params[3] + params[4] >= 0:
        return -1e10
    return -0.5 * (chi_squared(params) + norm_bao + norm_cmb)


def main():
    from multiprocessing import Pool
    from nautilus import Sampler, Prior
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(60.0, 75.0))  # km/s/Mpc
    prior.add_parameter("obh2", dist=(0.01, 0.03))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("w0", dist=(-3.0, 1.0))
    prior.add_parameter("wa", dist=(-4.0, 3.0))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()

    labels=["H_0", "ω_b", "ω_c", "w_0", "w_a"]
    gd_samples = MCSamples(samples=samples, weights=np.exp(log_w), names=prior.keys, labels=labels)
    gd_samples.addDerived(gd_samples["obh2"] + gd_samples["och2"] + Omnuh2, name="omh2", label="ω_m")
    gd_samples.addDerived(gd_samples["omh2"] / (gd_samples["H0"] / 100) ** 2, name="om", label="Ω_m")
    gd_samples.addDerived(cmb.r_drag(gd_samples["obh2"], gd_samples["omh2"]), name="rdrag", label="r_{drag}")

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=["H0", "om", "omh2", "rdrag", "w0", "wa"],
        title_limit=1,
        contour_colors=["C0"],
    )
    plt.show()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    map_params = samples[np.argmax(log_l)]
    DOF = N_bao + N_cmb - len(map_params)
    chi2_map = chi_squared(map_params)

    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"χ2 (MAP): {chi2_map:.2f}")
    print(f"χ2 / DOF (MAP): {chi2_map / DOF:.2f}")
    print(f"DOF: {DOF}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, map_params),
        data=bao,
        errors=np.sqrt(np.diag(cov_mat)),
        title=legend,
    )


if __name__ == "__main__":
    main()

# *******************************************
# Compressed Early Time ΛCDM 
# DESI BAO DR2 2025 + FS Lya
# *******************************************


# --------------- Flat ΛCDM -----------------
# H0 = 68.30 ± 0.29 km/s/Mpc
# Ωm = 0.3010 ± 0.0037
# r_d = 147.80 ± 0.19 Mpc
# Log evidence: 14.0
# χ2 (MAP): 15.65
# χ2 / DOF (MAP): 1.12
# DOF: 14
# -------------------------------------------


# --------------- Flat wCDM -----------------
# H0 = 68.77 ± 0.95 km/s/Mpc
# Ωm = 0.2977 ± 0.0075
# r_d = 147.74 ± 0.22 Mpc
# w0 = -1.020 ± 0.039 (prior ~ U[-1.5, -0.5])
# Log evidence: 11.8 (Δ logZ = 2.2 in favour of ΛCDM)
# χ2 (MAP): 15.47
# χ2 / DOF (MAP): 1.19
# DOF: 13
# -------------------------------------------


# -------------- Flat w0waCDM ---------------
# H0: 64.7 ± 2.0 km/s/Mpc
# Ωm: 0.340 +0.021 -0.024
# r_d: 147.47 ± 0.25 Mpc
# w0: -0.57 +0.21 -0.24 (prior ~ U[-3, 1])
# wa: -1.28 +0.74 -0.57 (prior ~ U[-4, 3])
# Log evidence: 11.0 + 0.3 = 11.3 (Δ logZ = 2.7 in favour of ΛCDM)
# χ2 (MAP): 11.34
# χ2 / DOF (MAP): 0.95
# DOF: 12
#
# w0 + wa < 0 enforced in the likelihood
# Correction in prior volume: ln(4 * 7 / (4 * 7 - (0.5 * (4+1) * 3))) ~ 0.31
# -------------------------------------------
