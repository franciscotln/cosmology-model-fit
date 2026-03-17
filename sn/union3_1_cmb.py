from numba import njit
import numpy as np
from interpolator import interp_hermite
from y2026union3_1.data import get_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
inv_cov_sn = np.linalg.inv(cov_matrix_sn)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence with w(z) ranging from -1 to 1
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


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
def Hz(z, params):
    H0 = params[1]
    return H0 * Ez(z, h=H0 / 100, Obh2=params[2], Och2=params[3])


cmb.set_HZ(Hz)


@njit
def DM_z(z, params):
    dh_grid = c / Hz(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DM_grid(params):
    dh_grid = c / Hz(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return (cum_dm, dh_grid)


@njit
def mu_corr(params, DM_interp):
    # Heaviside step at z = 0.2
    v_km_s = 100 * params[4] * np.where(z_cmb <= 0.2, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)

    DM_cosmo = interp_hermite(z_cosmo, z_grid, *DM_interp)
    DM_obs = interp_hermite(z_cmb, z_grid, *DM_interp)
    return 5 * np.log10(DM_cosmo / DM_obs)


@njit
def mu_theory(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    DM_interp = DM_grid(params)
    DM = interp_hermite(z_cmb, z_grid, *DM_interp)
    delta_sn = mu_vals - mu_theory(params[0], DM) - mu_corr(params, DM_interp)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    return chi2_cmb + chi_sn


@njit
def log_likelihood_single(params):
    return -0.5 * chi_squared(params)


def log_likelihood(batch):
    N = batch.shape[0]
    log_likes = np.empty(N, dtype=np.float32)
    for i in range(N):
        log_likes[i] = log_likelihood_single(batch[i])
    return log_likes


def main():
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from getdist import plots, MCSamples
    from matplotlib import pyplot as plt
    from sn.plotting import plot_predictions

    prior = Prior()
    prior.add_parameter("dM", dist=(-1.0, +1.0))
    prior.add_parameter("H0", dist=(60.0, 75.0))
    prior.add_parameter("obh2", dist=(0.01, 0.03))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("v", dist=(-10.0, 4.0))

    with Pool(6) as pool:
        sampler = Sampler(
            prior,
            log_likelihood,
            n_live=8_000,
            pool=pool,
            seed=42,
            pass_dict=False,
            vectorized=True,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()

    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=["ΔM", "H_0", "ω_b", "ω_c", "v_{flow}"],
        label="Union3.1 + CMB(R, lA, ωb)",
    )
    gd_samples.addDerived(
        Omnuh2 + gd_samples["obh2"] + gd_samples["och2"], name="omh2", label="ω_m"
    )
    gd_samples.addDerived(
        gd_samples["omh2"] / (gd_samples["H0"] / 100) ** 2, name="om", label="Ω_m"
    )

    g = plots.get_subplot_plotter()
    g.triangle_plot(
        gd_samples,
        params=["dM", "v", "H0", "om"],
        title_limit=1,
        filled=True,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    best_fit = np.percentile(samples, 50, axis=0)
    degs_of_freedom = len(mu_vals) + len(cmb.DISTANCE_PRIORS) - len(best_fit)

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    MAP_index = np.argmax(log_l)
    print(f"Chi2 (MAP): {chi_squared(samples[MAP_index]):.1f}")
    print(f"Log Evidence: {sampler.log_z:.1f}")
    print(f"Degrees of freedom: {degs_of_freedom}")

    plot_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit, DM_grid(best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(
            best_fit[0], interp_hermite(z_cmb, z_grid, *DM_grid(best_fit))
        ),
        label=f"ΛCDM",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: Union 3.1 (22 bins)
CMB(R, lA = π / θ*, ωb) ACT+Planck compressed
z range: 0.050 - 2.262
*******************************
"""

"""
Flat ΛCDM w(z) = -1
ΔM: -0.069 +- 0.011
H0: 67.50 +- 0.48 km/s/Mpc
Ωm: 0.3134 +- 0.0069
Chi2 (MAP): 29.6
Log Evidence: -33.1
Degrees of freedom: 21
"""

"""
Flat ΛCDM w(z) = -1
Isotropic velocity SNe observed redshifts (turning point z <= 0.2 inflow z > 0.2 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

ΔM: -0.067 ± 0.011 mag
v: -2.8 ± 1.1 (prior ~ U(-10, 4)) x 100 km/s
v / (z_cut=0.2): -1400 ± 550 km/s
H0: 67.68 ± 0.49 km/s/Mpc
Ωm: 0.3107 ± 0.0069
Chi2 (MAP): 22.4 (2.66 sigma significance)
Log Evidence: -31.2 (delta logZ = 1.9 in favour of flow)
Degrees of freedom: 20
"""

"""
Flat wCDM w(z) = w0
ΔM: -0.086 +0.018 -0.019
H0: 66.35 +1.15 -1.14 km/s/Mpc
Ωm: 0.324 +0.012 -0.012
ωm: 0.14241 +0.00118 -0.00117
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1193 +0.0012 -0.0012
w0: -0.957 +0.040 -0.039 (prior ~ U(-1.5, -0.5))
z*: 1089.68 +0.21 -0.21
z_drag: 1060.17 +0.23 -0.23
r*: 144.53 Mpc
r_d: 147.15 Mpc
Chi squared: 28.3
Log Evidence: -34.9
Degrees of freedom: 20
"""

"""
Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
ΔM: -0.081 +0.013 -0.013
H0: 66.15 +0.96 -0.80 km/s/Mpc
Ωm: 0.325 +0.009 -0.011
ωm: 0.14230 +0.00116 -0.00116
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1192 +0.0012 -0.0012
w0: -0.890 +0.051 -0.073 (prior ~ U(-1, -1/3))
wa: d w(z)/d z at z=0 = -1.5 * (1 - w0^2) = -0.312
z*: 1089.66 +0.21 -0.21
z_drag: 1060.18 +0.23 -0.23
r*: 144.56 Mpc
r_d: 147.17 Mpc
Chi2 (MAP): 27.6
Log Evidence: -33.6
Degrees of freedom: 20
"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
w0: (prior ~ U(-1.5, 0.0))
wa: (prior ~ U(-5.5, 3.0))
"""
