from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite, interp_pchip
from y2025DESdovekie.data import get_data as get_sn_data, effective_sample_size
from y2025BAO.data import get_data as get_bao_data
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
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_grid(params):
    dh_grid = DH_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return (cum_dm, dh_grid)


@njit
def DV_z(z, DM, DH):
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params, DM_interp):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnuh2
    rd = cmb.r_drag(Obh2, Omh2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)

    DM, DH = DM_interp

    results[DH_mask] = interp_pchip(z[DH_mask], z_grid, DH)
    results[DM_mask] = interp_hermite(z[DM_mask], z_grid, DM, DH)
    results[DV_mask] = DV_z(
        z[DV_mask],
        interp_hermite(z[DV_mask], z_grid, y=DM, y_prime=DH),
        interp_pchip(z[DV_mask], z_grid, y=DH),
    )
    return results / rd


@njit
def mu_corr(v_100, DM_interp):
    # Heaviside step at z = 0.10563
    v_km_s = 100 * v_100 * np.where(z_cmb <= 0.10563, 1, -1)
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + v_km_s / c)

    DM_cosmo = interp_hermite(z_cosmo, z_grid, *DM_interp)
    DM_obs = interp_hermite(z_cmb, z_grid, *DM_interp)
    return 5.0 * np.log10(DM_cosmo / DM_obs)


@njit
def theory_mu(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi2_sn(params, DM_interp):
    DM_sn = interp_hermite(z_cmb, z_grid, *DM_interp)
    delta_sn = mu_values - theory_mu(params[0], DM_sn) - mu_corr(params[4], DM_interp)
    return solve_triang(cho_sn, delta_sn)


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
    prior.add_parameter("v", dist=(-6.0, 2.0))

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
        effective_sample_size + len(bao) + len(cmb.DISTANCE_PRIORS) - len(prior.keys)
    )

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    map_index = np.argmax(log_l)
    map_params = samples[map_index]
    print(f"χ2 (MAP): {chi_squared(map_params):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"Degrees of freedom: {degs_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(
            z, qty, best_fit, DM_grid(best_fit)
        ),
        data=bao,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(best_fit[4], DM_grid(best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(
            best_fit[0], interp_hermite(z_cmb, z_grid, *DM_grid(best_fit))
        ),
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
Flat ΛCDM

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
Flat ΛCDM
Isotropic velocity SNe observed redshifts (turning point z <= 0.10563 inflow z > 0.10563 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

ΔM: -0.0619 ± 0.0079 mag
v: -1.58 ± 0.55 (prior ~ U(-6, 2)) x 100 km/s
v / (z_cut=0.10563): -1496 ± 521 km/s
H0: 68.44 ± 0.27 km/s/Mpc
Ωm: 0.3001 ± 0.0036
ωb: 0.02258 ± 0.00010
ωc: 0.11734 ± 0.00065
ωm: 0.1406 ± 0.0006
z*: 1089.40 ± 0.15
z_d: 1060.20 ± 0.23
r_d: 147.58 ± 0.19 Mpc
χ2 (MAP): 1641.50 (2.87 sigma significance)
Log evidence: -840.6 (Δ logZ = 2.4 in favour of in/outflow)
Degrees of freedom: 1725
"""


"""
Flat wCDM: w(z) = w0

H0: 67.73 +0.54 -0.53 km/s/Mpc
Ωm: 0.3056 +0.0048 -0.0047
ωb: 0.02259 +0.00011 -0.00011
ωc: 0.1170 +0.0008 -0.0008
ωm: 0.1402 +0.0008 -0.0008
w0: -0.972 +0.021 -0.022 (prior U(-4/3, -2/3))
wa: -0.083 +0.064 -0.062
r*: 145.06 Mpc
z*: 1089.34 +0.17 -0.17
r_d: 147.66 Mpc
z_d: 1060.21 +0.23 -0.23
Chi squared: 1647.95 (1.33 sigma away from ΛCDM)
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
w0: -0.906 +0.041 -0.041 (prior U(-1, -1/3))
wa: -0.268 +0.113 -0.110
r*: 145.07 Mpc
z*: 1089.33 +0.16 -0.16
r_d: 147.67 Mpc
z_d: 1060.22 +0.23 -0.23
Chi squared: 1644.75 (2.23 sigma away from ΛCDM)
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
w0: -0.822 +0.057 -0.055 (prior U(-1.5, 0.0))
wa: -0.620 +0.216 -0.231 (prior U(-2.5, 1.0))
r*: 144.66 Mpc
z*: 1089.60 +0.19 -0.19
r_d: 147.28 Mpc
z_d: 1060.18 +0.23 -0.24
Chi squared: 1639.18 (2.80 sigma away from ΛCDM)
Log evidence: -842.7 (Δ logZ = 0.3 against ΛCDM)
Degrees of freedom: 1724
"""
