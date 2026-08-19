from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025DESdovekie.data import get_data, effective_sample_size
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Onuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence with w(z) ranging from -1 to 1
    a3 = 1 / (1 + z) ** 3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


@njit
def Ez(z, h, Obh2, Och2):
    Obc = (Obh2 + Och2) / h**2
    Onu = Onuh2 / h**2
    Or = Orh2 / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, theta):
    H0 = theta[1]
    return H0 * Ez(z, h=H0 / 100, Obh2=theta[2], Och2=theta[3])


cmb.set_HZ(H_z)


@njit
def DM_grid(theta):
    dh_grid = c / H_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return (cum_dm, dh_grid)


@njit
def DM_z(z, DM_interp):
    return interp_hermite(z, z_grid, *DM_interp)


@njit
def mu_corr(params, DM_inter):
    # z_turn = 0.10563
    v_km_s = 100 * params[4] * np.where(z_cmb <= 0.11, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)

    return 5.0 * np.log10(DM_z(z_cosmo, DM_inter) / DM_z(z_cmb, DM_inter))


@njit
def theory_mu(params, DM_inter):
    return params[0] + 25 + 5 * np.log10((1.0 + z_hel) * DM_z(z_cmb, DM_inter))


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi2_sn(params):
    DM_inter = DM_grid(params)
    delta_sn = mu_vals - theory_mu(params, DM_inter) - mu_corr(params, DM_inter)
    return solve_triang(cho_sn, delta_sn)


@njit
def chi2_cmb(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    return delta @ cmb.inv_cov_mat @ delta


def chi_squared(params):
    return chi2_cmb(params) + chi2_sn(params)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions, plot_residuals

    prior = Prior()
    prior.add_parameter("dM", dist=(-0.7, +0.7))  # mag
    prior.add_parameter("H0", dist=(55, 75))  # km/s/Mpc
    prior.add_parameter("obh2", dist=(0.01, 0.03))
    prior.add_parameter("och2", dist=(0.01, 0.25))
    prior.add_parameter("v", dist=(-4.5, 4.5)) # x 100 km/s

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()

    labels = ["ΔM", "H_0", "Ω_b h^2", "Ω_c h^2", "v_{100}"]
    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(
        100 * gd_samples["v"], name="v_km_s", label="v_{km/s}"
    )
    gd_samples.addDerived(
        gd_samples["obh2"] + gd_samples["och2"] + Onuh2, name="omh2", label="ω_m"
    )
    gd_samples.addDerived(
        gd_samples["omh2"] / (gd_samples["H0"] / 100) ** 2, name="om", label="Ω_m"
    )
    gd_samples.updateBaseStatistics()

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    DOF = effective_sample_size + len(cmb.DISTANCE_PRIORS) - len(prior.keys)
    index_MAP = np.argmax(log_l)
    print(f"χ2 (MAP): {chi_squared(samples[index_MAP]):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {DOF}")

    best_fit = gd_samples.mean(prior.keys)
    dm_inter = DM_grid(best_fit)
    mu_pred = theory_mu(best_fit, dm_inter)
    mu_corrected = mu_vals - mu_corr(best_fit, dm_inter)
    residuals = mu_corrected - mu_pred
    mu_std = np.sqrt(np.diag(cov_matrix_sn))

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=["dM", "H0", "om", "v_km_s"],
        title_limit=1,
        contour_colors=["C0"],
    )
    plt.show()

    plot_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_corrected,
        y_err=mu_std,
        y_model=mu_pred,
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=mu_std, bins=60)


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# ΔM: -0.085 ± 0.012 mag
# H0: 67.38 ± 0.45 km/s/Mpc
# Ωb h^2: 0.02247 ± 0.00011
# Ωc h^2: 0.1199 ± 0.0011
# ωm: 0.1430 ± 0.0011
# Ωm: 0.3150 ± 0.0065
# χ2 (MAP): 1632.67
# Log evidence: -834.5
# DOF: 1713
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.10563 inflow z > 0.10563 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# v: -135 ± 56 km/s (prior ~U[-450, 450])
# v / z_turn: -1278 ± 530 km/s

# ΔM: -0.080 ± 0.012 mag
# H0: 67.65 ± 0.47 km/s/Mpc
# Ωb h^2: 0.02250 ± 0.00011
# Ωc h^2: 0.1192 ± 0.0011
# ωm: 0.14238 ± 0.00109
# Ωm: 0.3112 ± 0.0066
# χ2 (MAP): 1626.95 (2.39 sigma significance)
# Log evidence: -833.5 (Δ logZ = 1.0 in favour of velocity step correction)
# DOF: 1712
# ---------------------------------


# ----------- Flat wCDM -----------
# w0: 0.967 ± 0.026 (prior ~U[-2, 0])

# ΔM: -0.096 ± 0.015 mag
# H0: 66.66 ± 0.72 km/s/Mpc
# Ωb h^2: 0.02250 ± 0.00011
# Ωc h^2: 0.1193 ± 0.0012
# ωm: 0.14241 ± 0.00117
# Ωm: 0.3207 ± 0.0080
# χ2 (MAP): 1631.03 (1.28 sigma significance)
# Log evidence: -837.1 (ΛCDM is preferred)
# DOF: 1712
# ---------------------------------


# ----------- Flat wzCDM ----------
# Thawing quintessence with w(z) >= -1
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)

# w0: 0.925 +0.035 -0.048 (prior ~U[-1, 0])

# ΔM: -0.091 ± 0.012 mag
# H0: 66.64 ± 0.59 km/s/Mpc
# Ωb h^2: 0.02250 ± 0.00011
# Ωc h^2: 0.1191 ± 0.0012
# ωm: 0.1423 ± 0.0011
# Ωm: 0.3204 ± 0.0072
# χ2 (MAP): 1630.43 (1.50 sigma significance)
# Log evidence: -835.6 (ΛCDM is preferred)
# DOF: 1712
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < 0 enforced in the likelihood and corrected in the evidence calculation

# w0: -0.81 ± 0.11 (prior ~U[-2, 0])
# wa: -0.78 ± 0.55 (prior ~U[-4, 2])

# ΔM: -0.049 +0.038 -0.030 mag
# H0: 67.7 +1.1 -0.92 km/s/Mpc
# Ωb h^2: 0.02249 ± 0.00011
# Ωc h^2: 0.1194 ± 0.0012
# ωm: 0.1426 ± 0.0012
# Ωm: 0.3109 +0.009 -0.011
# χ2 (MAP): 1629.28 (1.33 sigma significance)
# Log evidence: -837.61 + 0.18 (ΛCDM is preferred)
# DOF: 1711
# ---------------------------------
