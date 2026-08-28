from numba import njit
import numpy as np
from interpolator import interp_hermite, interp_pchip
from y2025BAO.data_fs_lya import get_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

legend, bao, cov_mat = get_data()
inv_cov = np.linalg.inv(cov_mat)

z_grid = np.linspace(0, np.max(bao["z"]) + 0.1, 4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0, wa):
    # w0waCDM
    zp1 = 1 + z
    return zp1 ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / zp1)
    # thawing quintessence
    # a3 = zp1 ** -3
    # return 4 / ((1 + w0) * a3 + (1 - w0)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0, wa):
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

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0, wa = params
    return H0 * Ez(z, H0, Obh2, Och2, w0, wa)


cmb.set_HZ(H_z)


@njit
def DM_grid(params):
    dh_grid = c / H_z(z_grid, params)
    n = z_grid.size
    cum_dm = np.zeros(n, dtype=np.float64)

    # Compute local derivatives d(dh)/dz using central differences
    d_dh = np.empty(n, dtype=np.float64)

    # Central difference for internal points
    d_dh[1:-1] = (dh_grid[2:] - dh_grid[:-2]) / (2 * dz)
    # Forward/Backward difference at boundaries
    d_dh[0] = (dh_grid[1] - dh_grid[0]) / dz
    d_dh[-1] = (dh_grid[-1] - dh_grid[-2]) / dz

    # Integrate with 4th-order cubic correction per interval
    dz_sq_over_12 = (dz ** 2) / 12
    acc = 0.0

    for i in range(n - 1):
        # trapezoidal area + 1st-derivative endpoint correction
        trap = 0.5 * dz * (dh_grid[i] + dh_grid[i + 1])
        corr = dz_sq_over_12 * (d_dh[i] - d_dh[i + 1])

        acc += trap + corr
        cum_dm[i + 1] = acc

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
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    dm_interp = DM_grid(params)

    Obh2, Och2 = params[1], params[2]
    Omh2 = Obh2 + Och2 + Omnuh2
    inv_rd = 1.0 / cmb.r_drag(Obh2, Omh2)

    DM = interp_hermite(z, z_grid, y=dm_interp[0], y_prime=dm_interp[1])
    DH = interp_pchip(z, z_grid, y=dm_interp[1])

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
    return delta @ inv_cov @ delta


@njit
def chi2_cmb(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[1], params[2], params)
    return delta @ cmb.inv_cov_mat @ delta


@njit
def chi_squared(params):
    return chi2_cmb(params) + chi2_bao(params)


@njit
def log_likelihood(params):
    if params[3] + params[4] >= 0.0:
        return -1e8
    return -0.5 * chi_squared(params)


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
    prior.add_parameter("wa", dist=(-3.0, 2.0))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()

    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=["H_0", "ω_b", "ω_c", "w_0", "w_a"],
    )
    gd_samples.addDerived(
        gd_samples["obh2"] + gd_samples["och2"] + Omnuh2, name="omh2", label="ω_m"
    )
    gd_samples.addDerived(
        gd_samples["omh2"] / (gd_samples["H0"] / 100) ** 2, name="om", label="Ω_m"
    )
    gd_samples.addDerived(
        cmb.r_drag(gd_samples["obh2"], gd_samples["omh2"]),
        name="rdrag",
        label="r_{drag}",
    )

    plots.get_subplot_plotter().triangle_plot(
        gd_samples,
        ["H0", "om", "w0", "wa", "omh2", "rdrag"],
        title_limit=1,
        contour_colors=["C0"],
    )
    plt.show()

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    best_fit = gd_samples.mean(prior.keys)
    DOF = len(bao) + len(cmb.DISTANCE_PRIORS) - len(best_fit)

    print(f"χ2 (MAP): {chi_squared(samples[np.argmax(log_l)]):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {DOF}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao,
        errors=np.sqrt(np.diag(cov_mat)),
        title=legend,
    )


if __name__ == "__main__":
    main()

# *******************************************
# Compressed Planck + ACT
# DESI BAO DR2 2025 + FS Lya
# *******************************************


# --------------- Flat ΛCDM -----------------
# H0: 68.38 ± 0.27 km/s/Mpc
# Ωm: 0.3009 ± 0.0036
# r_d: 147.55 ± 0.19 Mpc
# χ2 (MAP): 16.49
# Log evidence: -21.9
# DOF: 14
# -------------------------------------------


# --------------- Flat wCDM -----------------
# H0: 68.96 ± 0.94 km/s/Mpc
# Ωm: 0.2968 ± 0.0074
# w0: -1.024 ± 0.038 (prior U[-1.5, -0.5])
# r_d: 147.46 ± 0.23 Mpc
# χ2 (MAP): 16.20 (0.54 sigma away from ΛCDM)
# Log evidence: -24.1 (Δ logZ = -2.2 in favour of ΛCDM)
# DOF: 13
# -------------------------------------------


# --------------- Flat wzCDM ----------------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
# H0: 67.2 +1.1 -0.57 km/s/Mpc
# Ωm: 0.3111 +0.0058 -0.010
# w0: < -0.906 +0.028 -0.090 (prior U[-1, 0] - truncated posterior)
# r_d: 147.63 ± 0.20 Mpc
# χ2 (MAP): 16.48 (0.1 sigma away from ΛCDM)
# Log evidence: -23.80 (Δ logZ = -1.9 in favour of ΛCDM)
# DOF: 13
# -------------------------------------------


# -------------- Flat w0waCDM ---------------
# H0: 64.9 ± 2.0 km/s/Mpc
# Ωm: 0.339 +0.020 -0.023
# w0: -0.58 +0.20 -0.23 (prior U[-3.0, 1.0])
# wa: -1.25 +0.69 -0.55 (prior U[-3.0, 2.0])
# r_d: 147.17 ± 0.26 Mpc
# χ2 (MAP): 11.71 (1.68 sigma away from ΛCDM)
# Log evidence: -24.4 + 0.3 = -24.1 (Δ logZ = -2.2 in favour of ΛCDM)
# DOF: 12
# w0 + wa < 0 enforced in the likelihood
# Correction in prior volume: ln(4 * 5 / (4 * 5 - (0.5 * 3 ** 2))) ~ 0.25
#
#
# imposing the constraint -1 <= w0 + wa < 0 then:
# H0: 69.3 ± 1.3 km/s/Mpc
# Ωm: 0.293 ± 0.010
# w0: -1.075 ± 0.077 (prior U[-3.0, 1.0])
# wa: 0.165 ± 0.142 (prior U[-3.0, 2.0])
# r_d: 147.55 ± 0.23
# χ2 (MAP): 16.46
# Log evidence: -29.0
# DOF: 12
# -------------------------------------------
