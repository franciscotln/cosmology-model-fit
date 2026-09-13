from numba import njit
import numpy as np
import cmb.data_spt_planck_act_compression as cmb
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2026union3_1.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, H_err_cc, cov_sys_cc = get_cc_data(split_sys=True)
sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

L_sn = np.linalg.cholesky(cov_matrix_sn)
L_bao = np.linalg.cholesky(cov_matrix_bao)

N_cc = len(z_cc_vals)

c = cmb.c  # km/s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]

z_piv_cc = 0.614


@njit
def Ode_z(z, w0, wa):
    zp1 = 1.0 + z
    cubed = zp1**3
    # return 1.0  # ΛCDM
    # return cubed ** (1.0 + w0)  # wCDM
    z_piv_w0 = 0.23
    return cubed ** (1.0 + w0 + wa / (1. + z_piv_w0)) * np.exp(-3 * wa * z / zp1)  # w0waCDM


@njit
def Ez(z, h, Obh2, Och2):
    Omnu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
    Ombc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Ombc - Or - Omnu

    radiation_term = Or * (1.0 + z) ** 4
    matter_term = Ombc * (1.0 + z) ** 3
    neutrino_term = Omnu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, theta):
    H0 = theta[3]
    return H0 * Ez(z, h=H0 / 100, Obh2=theta[4], Och2=theta[5])


cmb.set_HZ(H_z)


@njit
def DM_DH_grid(theta):
    dh_grid = c / H_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return (cum_dm, dh_grid)


@njit
def DH_z(z, dm_dh_grid):
    return interp_pchip(z, z_grid, dm_dh_grid[1])


@njit
def DM_z(z, dm_dh_grid):
    return interp_hermite(z, z_grid, dm_dh_grid[0], dm_dh_grid[1])


@njit
def DV_z(z, DM, DH):
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
desi_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, theta, dm_dh_grid):
    Obh2, Och2 = theta[4], theta[5]
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnu_h2)

    DH = DH_z(z, dm_dh_grid)
    DM = DM_z(z, dm_dh_grid)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    FAP_mask = qty == 3
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH[DH_mask] / rd
    results[DM_mask] = DM[DM_mask] / rd
    results[DV_mask] = DV_z(z[DV_mask], DM[DV_mask], DH[DV_mask]) / rd
    results[FAP_mask] = DM[FAP_mask] / DH[FAP_mask]
    return results


@njit
def get_z_cosmo(params):
    # Heaviside step at z = 0.2
    z_offset = 1e-3 * params[6] * np.where(z_cmb <= 0.2, 1, -1)
    return z_cmb + z_offset


def mu_corr(params, dm_dh_grid):
    # For plotting purposes only
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, dm_dh_grid) / DM_z(z_cmb, dm_dh_grid))


@njit
def mu_theory(theta, DM):
    return theta[2] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(theta, L_cc):
    dm_dh_grid = DM_DH_grid(theta)
    delta_thetastar = (cmb.DISTANCE_PRIORS - cmb.cmb_distances(theta[4], theta[5], theta))[0]
    chi2_theta_star = delta_thetastar**2 / cmb.covariance[0, 0]

    z_cosmo = get_z_cosmo(theta)
    delta_sn = mu_values - mu_theory(theta, DM_z(z_cosmo, dm_dh_grid))
    y_sn = solve_triangular(L_sn, delta_sn)
    chi2_sn = y_sn @ y_sn

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, theta, dm_dh_grid)
    y_bao = solve_triangular(L_bao, delta_bao)
    chi2_bao = y_bao @ y_bao

    delta_cc = H_cc_vals - H_z(z_cc_vals, theta)
    y_cc = solve_triangular(L_cc, delta_cc)
    chi2_cc = y_cc @ y_cc

    return chi2_theta_star + chi2_sn + chi2_bao + chi2_cc


@njit
def log_likelihood(theta):
    fp_cc, n_cc = np.exp(theta[0]), theta[1]
    fz_cc = fp_cc * ((1 + z_cc_vals) / (1 + z_piv_cc)) ** n_cc
    cov_cc = cov_sys_cc + np.diag(H_err_cc**2 * fz_cc**2)
    L_cc = np.linalg.cholesky(cov_cc)
    logdet_cc = 2.0 * np.sum(np.log(np.diag(L_cc)))
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc
    return -0.5 * chi_squared(theta, L_cc) - 0.5 * normalization_cc


def main():
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from ohd.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    # ---- ln(fp): CC error rescaling (overestimated) ----
    prior.add_parameter("ln_fp", dist=(np.log(0.3), np.log(1.2)))
    prior.add_parameter("n_cc", dist=(-4.0, +4.0))
    # ----------------------------------------------------

    # ΔM: Supernova magnitude offset (zero point)
    prior.add_parameter("ΔM", dist=(-1.0, 1.0))
    # ----------------------------------------------------

    # ---- cosmological params ----
    # H0: Hubble constant at present
    prior.add_parameter("H0", dist=(50.0, 85.0))
    # Ωb x h^2: baryon density parameter
    prior.add_parameter("obh2", dist=(0.003, 0.050))
    # Ωc x h^2: cold dark matter density param today
    prior.add_parameter("och2", dist=(0.05, 0.30))
    # ----------------------------------------------------

    # Sne1a redshift offset: 1000 Δz
    prior.add_parameter("dz_1000", dist=(-3.5, 3.5))
    # ----------------------------------------------------

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    best_fit = samples[np.argmax(log_l)]
    DOF = 1 + len(z_cmb) + len(bao_data) + N_cc - len(prior.keys)
    labels = ["ln(f_p)", "n_{cc}", "ΔM", "H_0", "ω_b", "ω_c", "1000 Δz"]
    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=-log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(gd_samples["obh2"] + gd_samples["och2"] + Omnu_h2, name="omh2", label="Ω_m h^2")
    gd_samples.addDerived(gd_samples["omh2"] * (100 / gd_samples["H0"])**2, name="om", label="Ω_m")
    gd_samples.addDerived(cmb.r_drag(gd_samples["obh2"], gd_samples["omh2"]), name="rd", label="r_{drag}")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    fz_cc = np.exp(best_fit[0]) * ((1. + z_cc_vals) / (1. + z_piv_cc))**best_fit[1]
    cov_cc = np.diag(H_err_cc**2 * fz_cc**2) + cov_sys_cc
    L_cc = np.linalg.cholesky(cov_cc)

    print(f"Chi squared (MAP): {chi_squared(best_fit, L_cc):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"Degrees of freedom: {DOF}")

    plots.get_subplot_plotter().triangle_plot(
        gd_samples,
        params=["H0", "om", "omh2", "dz_1000", "ln_fp", "n_cc"],
        title_limit=1,
        contour_colors=["C0"],
    )
    plt.show()

    dm_dh_grid = DM_DH_grid(best_fit)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit, dm_dh_grid),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=H_err_cc,
        label=cc_legend,
        err_scaling=1.0 / fz_cc,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(best_fit, dm_dh_grid),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit, DM_z(z_cmb, dm_dh_grid)),
        label=rf"$Ω_m$={gd_samples['om'].mean():.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# *********************************
# Priors:
# ln(fp_cc) ~U[ln(0.3), ln(1.2)]
# n_cc      ~U[-4, +4]
# ΔM        ~U[-1, +1]
# H0        ~U[50, 85]
# ωb        ~U[0.003, 0.050]
# ωc        ~U[0.05, 0.30]
#
# wCDM:
# w0        ~U[-1.5, -0.5]
#
# w0waCDM:
# w0        ~U[-1.5, 0]
# wa        ~U[-3.5, +2]
# w0 + wa <= -1/3 enforced
#
# Z offset step correction:
# 1000 Δz   ~U[-3.5, +3.5]
# *********************************


# Flat ΛCDM: w(z) = -1
# H0 = 67.4 +1.2 -1.5 km/s/Mpc
# ωb = 0.0208 +0.0017 -0.0021
# ωc = 0.11603 +0.00099 -0.0014
# Ωm h^2 = 0.1375 +0.0024 -0.0034
# Ωm = 0.3032 ± 0.0070
# rd = 150.0 +2.7 -2.3 Mpc
# ln(fp) = -0.51 +0.11 -0.12
# n_cc = 1.39 ± 0.47
# ΔM = -0.083 +0.035 -0.042 mag
# Chi squared (MAP): 81.06
# Log evidence: -192.68
# Degrees of freedom: 70
# ---------------------------------


# Flat ΛCDM: w(z) = -1
# redshift offset step correction in SNe observed redshifts
# turning point z <= 0.2 positive z > 0.2 negative
# z_cosmo = z_cmb +- Δz
#
# H0 = 67.9 +1.3 -1.5 km/s/Mpc
# ωb = 0.0215 +0.0018 -0.0022
# ωc = 0.1162 +0.0011 -0.0015
# 1000 Δz = 1.14 ± 0.40
# Ωm h^2 = 0.1384 +0.0026 -0.0036
# Ωm = 0.3001 ± 0.0069
# rd = 149.1 +2.8 -2.4 Mpc
# ln(fp) = -0.51 +0.11 -0.12
# n_cc = 1.40 ± 0.47
# ΔM = -0.070 +0.036 -0.043 mag
# Chi squared (MAP): 73.75
# Log evidence: -190.46
# Degrees of freedom: 69
# ---------------------------------


# Flat wCDM: w(z) = w0
# H0 = 67.7 +1.3 -1.6 km/s/Mpc
# ωb = 0.0233 +0.0024 -0.0031
# ωc = 0.1155 +0.0014 -0.0017
# w0 = -0.940 ± 0.042
# Ωm h^2 = 0.1394 +0.0030 -0.0045
# Ωm = 0.3039 ± 0.0069
# rd = 147.4 +3.6 -3.0 Mpc
# ln(fp) = -0.49 +0.11 -0.13
# n_cc = 1.36 ± 0.47
# ΔM = -0.051 +0.043 -0.053 mag
# Chi squared (MAP): 78.55
# Log evidence: -193.89
# Degrees of freedom: 69
# ---------------------------------


# Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
# H0 = 65.2 +1.4 -2.0 km/s/Mpc
# ωb = 0.0193 +0.0019 -0.0033
# ωc = 0.1180 +0.0018 -0.0013
# w0 = -0.78 ± 0.10
# wa = -0.87 ± 0.50
# Ωm h^2 = 0.1380 +0.0020 -0.0033
# Ωm = 0.325 ± 0.013
# rd = 151.3 +3.7 -2.5 Mpc
# ln(fp) = -0.50 +0.11 -0.13
# n_cc = 1.43 ± 0.48
# ΔM = -0.107 +0.035 -0.053 mag
# Chi squared (MAP): 75.37
# Log evidence: -194.19
# Degrees of freedom: 68

# At a z_pivot = 0.23 (corr(wp, wa) = -0.006)
# Flat wpwaCDM: w(z) = wp + wa * (1 / (1 + z_pivot) - 1 / (1 + z))

# H0 = 65.2 +1.4 -2.0 km/s/Mpc
# ωb = 0.0193 +0.0019 -0.0033
# ωc = 0.1180 +0.0019 -0.0013
# wp = -0.942 ± 0.043 (prior ~ U[-1.5, -0.5])
# wa = -0.87 ± 0.50 (prior ~ U[-3, 2])
# Ωm h^2 = 0.1380 +0.0020 -0.0033
# Ωm = 0.325 +0.014 -0.012
# rd = 151.3 +3.7 -2.5 Mpc
# ln(fp) = -0.50 +0.11 -0.12
# n_cc = 1.44 ± 0.48
# ΔM = -0.107 +0.035 -0.053 mag
# Chi squared (MAP): 76.02
# Log evidence: -193.79
# Degrees of freedom: 68
# ---------------------------------
