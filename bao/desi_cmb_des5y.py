from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025DESdovekie.data import get_data as get_sn_data, effective_sample_size
from y2025BAO.data import get_data as get_bao_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1 + w0 + (1 - w0) * zp1**3)) ** 2  # wzCDM
    # return 1  # ΛCDM
    # return zp1 ** (3 * (1 + w0))  # wCDM
    # return zp1 ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / zp1)  # w0waCDM


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
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
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

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_bao + chi_sn


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

    best_fit_mean = gd_samples.mean(prior.keys)
    degs_of_freedom = (
        effective_sample_size
        + len(bao_data["z"])
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

ΔM: -0.0626 ± 0.0079
H0: 68.34 ± 0.27 km/s/Mpc
Ωm: 0.3015 ± 0.0035
ωb: 0.02256 ± 0.00010
ωc: 0.11760 ± 0.00064
ωm: 0.14081 ± 0.00063
z*: 1089.44 ± 0.15
z_d: 1060.19 ± 0.23
r_d: 147.52 ± 0.19 Mpc
χ2 (MAP): 1649.72
Log evidence: -843.0
Degrees of freedom: 1726
"""


"""
Flat wCDM: w(z) = w0

H0: 67.73 +0.54 -0.53 km/s/Mpc
Ωm: 0.3056 +0.0048 -0.0047
ωb: 0.02259 +0.00011 -0.00011
ωc: 0.1170 +0.0008 -0.0008
ωm: 0.1402 +0.0008 -0.0008
w0: -0.972 +0.021 -0.022 (prior width 2/3: -4/3 to -2/3)
wa: -0.083 +0.064 -0.062
r*: 145.06 Mpc
z*: 1089.34 +0.17 -0.17
r_d: 147.66 Mpc
z_d: 1060.21 +0.23 -0.23
Chi squared: 1647.95
Log evidence: -844.6 (Δ logZ = -1.6 in favour of ΛCDM)
Degrees of freedom: 1725
"""


"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)

H0: 67.26 +0.54 -0.54 km/s/Mpc
Ωm: 0.3099 +0.0053 -0.0051
ωb: 0.02260 +0.00010 -0.00010
ωc: 0.1169 +0.0007 -0.0007
ωm: 0.1402 +0.0007 -0.0007
w0: -0.906 +0.041 -0.041 (prior width 2/3: -1 to -1/3)
wa: -0.268 +0.113 -0.110
r*: 145.07 Mpc
z*: 1089.33 +0.16 -0.16
r_d: 147.67 Mpc
z_d: 1060.22 +0.23 -0.23
Chi squared: 1644.75
Log evidence: -842.4 (Δ logZ = 0.6 against ΛCDM)
Degrees of freedom: 1725
"""


"""
Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
H0: 67.43 +0.55 -0.54 km/s/Mpc
Ωm: 0.3120 +0.0054 -0.0053
ωb: 0.02252 +0.00011 -0.00011
ωc: 0.1187 +0.0010 -0.0010
ωm: 0.1419 +0.0009 -0.0009
w0: -0.822 +0.057 -0.055 (prior width 1.5: -1.5 to 0.0)
wa: -0.620 +0.216 -0.231 (prior width 3.5: -2.5 to 1.0)
r*: 144.66 Mpc
z*: 1089.60 +0.19 -0.19
r_d: 147.28 Mpc
z_d: 1060.18 +0.23 -0.24
Chi squared: 1639.18
Log evidence: -842.7 (Δ logZ = 0.3 compared to ΛCDM)
Degrees of freedom: 1724
"""


"""
Flat ΛCDM, varying absolute magnitude M(z) of SNe
M(z) = ΔM + tanh(1 - z^(0.1 * p))

p: 0.132 ± 0.052 (prior ~ U(-0.4, +0.8))
ΔM: -0.0738 ± 0.0091 mag
H0: 68.49 ± 0.27 km/s/Mpc
Ωm: 0.2994 ± 0.0036
ωb: 0.02258 ± 0.00010
ωc: 0.11723 ± 0.00065
ωm: 0.1405 ± 0.0006
z*: 1089.38 ± 0.15
z_d: 1060.21 ± 0.23
r_d: 147.60 ± 0.19 Mpc
χ2 (MAP): 1643.28
Log evidence: -842.0 (Δ logZ = 1.0 compared to no evolution in M(z))
Degrees of freedom: 1725
"""
