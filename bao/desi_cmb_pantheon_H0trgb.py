from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_planck_act_compression as cmb
from interpolator import interp_hermite
from y2022pantheonSHOES.data import get_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # km/s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mb_values, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return zp1 ** (3 * (1.0 + w0))  # wCDM
    # return 1  # ΛCDM
    # return (2 * zp1**3 / ((1.0 + w0) + (1.0 - w0) * zp1**3)) ** 2  # wzCDM
    # return zp1 ** (3 * (1.0 + w0 + wa)) * np.exp(-3 * wa * z / zp1)  # w0waCDM


@njit
def Ez(z, H0, Obh2, Och2):
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
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
    return H0 * Ez(z, H0=H0, Obh2=params[2], Och2=params[3])


cmb.set_HZ(H_z)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnu_h2
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
def apparent_mag(params):
    v_100 = params[4]
    v_flow_corr = 100 * v_100 * (5 / np.log(10)) / (c * z_cmb)
    M = params[0] + v_flow_corr
    return M + 25.0 + 5 * np.log10((1.0 + z_hel) * DM_z(z_cmb, params))


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_sn = mb_values - apparent_mag(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    chi2_H0 = ((params[1] - 70.39) / 1.80) ** 2  # TRGB arXiv:2408.06153v3

    return chi2_cmb + chi_bao + chi_sn + chi2_H0


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
    prior.add_parameter("M", dist=(-20.0, -19.0))
    prior.add_parameter("H0", dist=(60.0, 75.0))
    prior.add_parameter("obh2", dist=(0.019, 0.025))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("v_flow", dist=(-1.2, 3.2))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    labels = ["M", "H_0", "ω_b", "ω_c", "v_{flow}"]
    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(
        gd_samples["obh2"] + gd_samples["och2"] + Omnu_h2, name="omh2", label="ω_m"
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
        1 + len(z_cmb) + len(bao_data) + len(cmb.DISTANCE_PRIORS) - len(prior.keys)
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
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mb_values - gd_samples.mean("M"),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag(best_fit_mean) - gd_samples.mean("M"),
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
Priors:

M  ~U(-20.0, -19.0)
H0 ~U(60.0, 75.0)
ωb ~U(0.019, 0.025)
ωc ~U(0.01, 0.25)

wCDM:
w0 ~U(-1.5, -0.5)

w0waCDM:
w0 ~U(-1.5, 0.0)
wa ~U(-2.0, 1.0)
with w0 + wa < 0 enforced

v_flow corrections of SNe:
v_flow ~U(-1.2, 3.2) [x 100 km/s]
"""

"""
Flat ΛCDM
M0: -19.411 ± 0.008 mag
H0: 68.42 ± 0.27 km/s/Mpc
ωb: 0.02258 ± 0.00010
ωc: 0.1174 ± 0.0006
ωm: 0.1406 ± 0.0006
Ωm: 0.301 ± 0.004
z*: 1089.41 ± 0.15
zd: 1060.21 ± 0.23
rd: 147.56 ± 0.19 Mpc
MAP chi^2: 1421.56
Log evidence: -727.9
Degrees of freedom: 1603
"""

"""
Flat ΛCDM
Outflow v_flow correction of SNe M(z) = M_inf + v_flow_corr
v_flow_corr = 100 * v_flow * (5 / np.log(10)) / (c * z_cmb) with v_flow in units 100 km/s

v_flow: 96 ± 42 km/s
M_inf: -19.422 ± 0.010 mag
H0: 68.48 ± 0.27 km/s/Mpc
ωb: 0.02258 ± 0.00010
ωc: 0.1173 ± 0.0006
ωm: 0.1405 ± 0.0006
Ωm: 0.300 ± 0.004
z*: 1089.39 ± 0.15
zd: 1060.21 ± 0.23
rd: 147.59 ± 0.19 Mpc
MAP chi^2: 1416.51 (2.25 sigma away from no v_flow)
Log evidence: -726.8 (Δ logZ = 1.1 in favour of v_flow)
Degrees of freedom: 1602
"""

"""
Flat wCDM w(z) = w0
w0: -0.986 ± 0.022
M0: -19.418 ± 0.014 mag
H0: 68.12 ± 0.56 km/s/Mpc
ωb: 0.02259 ± 0.00011
ωc: 0.1171 ± 0.0008
ωm: 0.1404 ± 0.0008
Ωm: 0.303 ± 0.005
z*: 1089.36 ± 0.17
zd: 1060.21 ± 0.23
rd: 147.62 ± 0.22 Mpc
MAP chi^2: 1421.16 (0.63 sigma away from ΛCDM)
Log evidence: -730.6 (Δ logZ = -2.70 in favour of ΛCDM)
Degrees of freedom: 1602
"""


"""
Flat w(z) = w0 + wa * z / (1 + z)
TODO
"""
