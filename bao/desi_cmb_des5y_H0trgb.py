from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025DESdovekie.data import get_data as get_sn_data, effective_sample_size
from y2025BAO.data import get_data as get_bao_data
from y20116dFBAO.data import get_data as get_6dF_bao_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
sixdF_bao_legend, sixdF_bao_data, sixdF_bao_cov_matrix = get_6dF_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)
inv_cov_6dF_bao = np.linalg.inv(sixdF_bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2  # wzCDM thawing quint.
    # return 1.0  # ΛCDM
    # return zp1 ** (3 * (1.0 + w0))  # wCDM
    # return zp1 ** (3 * (1.0 + w0 + wa)) * np.exp(-3 * wa * z / zp1)  # w0waCDM


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
    return H0 * Ez(z, H0=H0, Obh2=params[2], Och2=params[3], w0=params[4])


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
desi_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)
sixdF_qty = np.array([qty_map[q] for q in sixdF_bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnuh2
    rd = cmb.r_drag(wb=Obh2, wm=Omh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


@njit
def theory_mu(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta @ cmb.inv_cov_mat @ delta

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_bao_6dF = sixdF_bao_data["value"] - bao_theory(
        sixdF_bao_data["z"], sixdF_qty, params
    )
    chi_6dF_bao = delta_bao_6dF @ inv_cov_6dF_bao @ delta_bao_6dF

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    chi2_H0 = ((params[1] - 70.39) / 1.80) ** 2  # TRGB arXiv:2408.06153v3

    return chi2_cmb + chi_bao + chi_6dF_bao + chi_sn + chi2_H0


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
    prior.add_parameter("w0", dist=(-1.0, -1 / 3))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False
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

    best_fit_mean = gd_samples.mean(prior.keys)
    degs_of_freedom = (
        1
        + effective_sample_size
        + len(bao_data["z"])
        + len(sixdF_bao_data["z"])
        + len(cmb.DISTANCE_PRIORS)
        - len(prior.keys)
    )

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    map_index = np.argmax(log_l)
    map_params = samples[map_index]
    print(f"χ2 (MAP): {chi_squared(map_params):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"Degrees of freedom: {degs_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit_mean),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit_mean),
        data=sixdF_bao_data,
        errors=np.sqrt(np.diag(sixdF_bao_cov_matrix)),
        title=sixdF_bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit_mean),
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
*******************************
DESI DR2 + DES5Y + (R, π/θ*, ωb)CMB ACT DR6 + Planck
*******************************
"""

"""
Flat ΛCDM: w(z) = -1
ΔM: -0.0613 ± 0.0078 mag
H0: 68.39 ± 0.26 km/s/Mpc
ωb: 0.02257 ± 0.00010
ωc: 0.11747 ± 0.00063
ωm: 0.1407 ± 0.0006
Ωm: 0.301 ± 0.004
z*: 1089.42 ± 0.15
zd: 1060.20 ± 0.23
rd: 147.55 ± 0.19 Mpc
χ2 (MAP): 1651.26
Log evidence: -843.8
Degrees of freedom: 1728
"""


"""
Flat ΛCDM: w(z) = -1
Evolving absolute mag of SNe M(z) = ΔM_max + 0.2 * p / (1 + (z / z_c))
where z_c = 0.043 and p = -20 * z_c * M'(z_c)

p: 0.40 ± 0.15 (prior ~ U(-0.5, +1.5))
ΔM_max: -0.0706 ± 0.0086 mag
H0: 68.53 ± 0.27 km/s/Mpc
ωb: 0.02259 ± 0.00010
ωc: 0.11715 ± 0.00064
ωm: 0.1404 ± 0.0006
Ωm: 0.299 ± 0.004
z*: 1089.36 ± 0.15
zd: 1060.22 ± 0.23
rd: 147.61 ± 0.19 Mpc
χ2 (MAP): 1643.72 (2.75 sigmas away from constant M)
Log evidence: -841.7 (ΔlogZ = 2.1 against constant M)
Degrees of freedom: 1727
"""


"""
Flat wCDM: w(z) = w0
ΔM: -0.067 ± 0.010 mag
H0: 67.99 ± 0.52 km/s/Mpc
ωb: 0.02260 ± 0.00011
ωc: 0.11700 ± 0.00081
w0: -0.981 ± 0.021 (prior U(-4/3, -2/3))
ωm: 0.1402 ± 0.0008
Ωm: 0.303 ± 0.005
z*: 1089.34 ± 0.17
zd: 1060.22 ± 0.23
rd: 147.65 ± 0.22 Mpc
χ2 (MAP): 1650.39 (0.93 sigmas away from ΛCDM)
Log evidence: -845.9 (ΔlogZ = -2.1 in favour of ΛCDM)
Degrees of freedom: 1727
"""


"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
ΔM: -0.0694 ± 0.088 mag
H0: 67.54 +0.52 -0.47 km/s/Mpc
ωb: 0.02260 ± 0.00010
ωc: 0.11689 ± 0.00069
w0: -0.924 +0.035 -0.041 (prior U(-1, -1/3))
ωm: 0.1401 ± 0.0007
Ωm: 0.307 ± 0.005
z*: 1089.32 ± 0.16
zd: 1060.23 ± 0.23
rd: 147.67 ± 0.20 Mpc
χ2 (MAP): 1648.00 (1.81 sigmas away from ΛCDM)
Log evidence: -844.1 (ΔlogZ = -0.3 in favour of ΛCDM)
Degrees of freedom: 1727
"""


"""
Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
ΔM: -0.051 ± 0.011 mag
H0: 67.72 ± 0.52 km/s/Mpc
Ωm: 0.3094 ± 0.0051
ωb: 0.02252 ± 0.00011
ωc: 0.1187 ± 0.0010
ωm: 0.1419 ± 0.0009
w0: -0.834 ± 0.056 (prior U(-1.5, 0.0))
wa: -0.62 +0.24 -0.21 (prior U(-2.5, 1.0))
z*: 1089.59 ± 0.19
z_d: 1060.19 ± 0.24
r_d: 147.28 ± 0.24 Mpc
χ2 (MAP): 1642.06 (2.57 sigmas away from ΛCDM)
Log evidence: -844.3 (ΔlogZ = -0.5 in favour of ΛCDM)
Degrees of freedom: 1726
"""
