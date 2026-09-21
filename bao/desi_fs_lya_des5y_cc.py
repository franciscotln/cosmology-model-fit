from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from cmb.data_early_lcdm_compression import r_drag
from y2005cc.data_no_loubser import get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data
from y2025DESdovekie.data import (
    effective_sample_size as sn_sample,
    get_data as get_sn_data,
)

c = c0 / 1000  # km/s

cc_legend, z_cc, H_cc, diag_stat_cc, cov_mat_sys_cc = get_cc_data(split_sys=True)
sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

N_cc = len(z_cc)

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]

# ----- PARAMS -----
names = ["ln_fp", "n_cc", "dM", "h0", "Obh2", "Om", "1000_dz"]
labels = ["ln(f_{p,cc})", "n_{cc}", "ΔM", "H_0", "Ω_b h^2", "Ω_m", "1000 Δz"]
bounds = np.array([
    (-2, 1),  # ln(fp): CC error rescaling (overestimated)
    (-3, 7),  # n_cc: CC error rescaling power (overestimated)
    (-0.55, +0.55),  # ΔM: magnitude offset
    (45, 90),  # H0: Hubble constant at present
    (0.001, 0.04),  # Ωb h^2: baryon density parameter at present
    (0.2, 0.7),  # Ωm: matter density parameter at present
    (-1.5, +1.5),  # 1000 x Δz
])
# ------------------

@njit
def rho_de(z, w0, wa):
    cubed = (1.0 + z) ** 3
    # return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2  # wzCDM
    # return 1.0  # ΛCDM
    # return cubed ** (1.0 + w0)  # wCDM
    return cubed ** (1.0 + w0 + wa) * np.exp(-3 * wa * z / (1.0 + z))  # w0waCDM


@njit
def H_z(z, theta):
    H0, Om = theta[3], theta[5]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_DH_grid(theta):
    dh_grid = c / H_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return (cum_dm, dh_grid)


@njit
def DM_z(z, dm_dh_grid):
    return interp_hermite(z, z_grid, dm_dh_grid[0], dm_dh_grid[1])


@njit
def DH_z(z, dm_dh_grid):
    return interp_pchip(z, z_grid, dm_dh_grid[1])


@njit
def DV_z(z, DM, DH):
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
quantities = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, theta, dm_dh_grid):
    H0, Obh2, Om = theta[3], theta[4], theta[5]
    Omh2 = Om * (H0 / 100)**2
    inv_rd = 1.0 / r_drag(Obh2, Omh2)

    DM = DM_z(z, dm_dh_grid)
    DH = DH_z(z, dm_dh_grid)

    N = z.size
    results = np.empty(N, dtype=np.float64)

    for i in range(N):
            q = qty[i]
            if q == 0:
                results[i] = DV_z(z[i], DM[i], DH[i]) * inv_rd
            elif q == 1:
                results[i] = DM[i] * inv_rd
            elif q == 2:
                results[i] = DH[i] * inv_rd
            elif q == 3:
                results[i] = DM[i] / DH[i]

    return results


@njit
def get_z_cosmo(params):
    # z_turn = 0.10563
    z_offset = 1e-03 * params[6] * np.where(z_cmb <= 0.10563, 1, -1)
    return z_cmb + z_offset


def mu_corr(params, dm_grid):
    # For plotting purposes only
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, dm_grid) / DM_z(z_cmb, dm_grid))


@njit
def mu_theory(theta, DM):
    dL = (1.0 + z_hel) * DM
    return theta[2] + 25.0 + 5 * np.log10(dL)


@njit
def chi_squared(theta, cho_cc):
    dm_dh_grid = DM_DH_grid(theta)

    z_cosmo = get_z_cosmo(theta)
    DM_cosmo = DM_z(z_cosmo, dm_dh_grid)
    delta_sn = mu_values - mu_theory(theta, DM_cosmo)
    y_sn = solve_triangular(cho_sn, delta_sn)
    chi_sn = np.dot(y_sn, y_sn)

    delta_bao = bao["value"] - bao_theory(bao["z"], quantities, theta, dm_dh_grid)
    y_bao = solve_triangular(cho_bao, delta_bao)
    chi_bao = np.dot(y_bao, y_bao)

    delta_cc = H_cc - H_z(z_cc, theta)
    y_cc = solve_triangular(cho_cc, delta_cc)
    chi_cc = np.dot(y_cc, y_cc)

    return chi_sn + chi_bao + chi_cc


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


@njit
def get_fz(theta):
    z_pivot = 1.035
    f_piv, n = np.exp(theta[0]), theta[1]
    return f_piv * ((1.0 + z_cc) / (1.0 + z_pivot))**n


@njit
def log_likelihood(theta):
    cov_mat_cc = cov_mat_sys_cc + np.diag(diag_stat_cc ** 2 * get_fz(theta)**2)
    cho_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2 * np.sum(np.log(np.diag(cho_cc)))
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc

    return -0.5 * (chi_squared(theta, cho_cc) + normalization_cc)


@njit
def log_probability_jit(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def log_probability(theta):
    return log_probability_jit(theta)


def main():
    from multiprocessing import Pool
    import emcee
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from sn.plotting import plot_predictions as plot_sn_predictions
    from ohd.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 1000
    nsteps = 4000 + burn_in
    state0 = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [(emcee.moves.KDEMove(), 0.20), (emcee.moves.DEMove(), 0.80)]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(state0, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"})

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    samples = sampler.get_chain(discard=burn_in, flat=False)
    flat_log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=False)

    # reshape for getdist
    chain_list = np.moveaxis(samples, 1, 0)
    loglike_list = np.moveaxis(log_probs, 1, 0)

    gd_samples = MCSamples(
        samples=chain_list,
        loglikes=-loglike_list,
        names=names,
        labels=labels,
        label='DES + DESI + CC'
    )
    gd_samples.addDerived(
        r_drag(gd_samples["Obh2"], gd_samples["Om"] * (gd_samples["h0"] / 100)**2),
        name="rdrag",
        label="r_{drag}",
    )
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    MAP_PARAMS = flat_samples[np.argmax(flat_log_probs)]
    fz_cc = get_fz(MAP_PARAMS)
    DOF = sn_sample + len(bao) + N_cc - ndim

    print(f"log likelihood (MAP): {log_likelihood(MAP_PARAMS):.2f}")
    print(f"DOF: {DOF}")

    dm_dh_grid = DM_DH_grid(MAP_PARAMS)

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, MAP_PARAMS, dm_dh_grid),
        data=bao,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, MAP_PARAMS),
        z=z_cc,
        H=H_cc,
        H_err=np.sqrt(np.diag(cov_mat_sys_cc) + diag_stat_cc**2),
        label=f"{cc_legend} $H_0$: {MAP_PARAMS[3]:.1f} km/s/Mpc",
        err_scaling=1 / fz_cc,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(MAP_PARAMS, dm_dh_grid),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(MAP_PARAMS, DM_z(z_cmb, dm_dh_grid)),
        label=f"$Ω_m$={MAP_PARAMS[5]:.3f}",
        x_scale="log",
    )
    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=names,
        title_limit=1,
        contour_colors=["C0"],
        filled=True,
    )
    plt.show()

if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# H0 = 67.9 ± 2.7 km/s/Mpc
# Ωm = 0.3083 ± 0.0069
# rd = 148.4 +5.5 -6.1 Mpc
#
# ΔM = -0.073 ± 0.087 mag
# ln(fp_cc) = -0.47 ± 0.27
# n_cc = 3.05 +0.87 -1.2
#
# Chi squared (MAP): 1683.23
# DOF: 1758
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.10563 positive z > 0.10563 negative
# z_cosmo = z_cmb ± Δz

# H0 = 68.2 ± 2.8 km/s/Mpc
# Ωm = 0.3033 ± 0.0070
# rd = 148.4 +5.5 -6.2 Mpc
# 1000 Δz = 0.53 ± 0.20 (prior ~ U[-1.5, +1.5])
#
# ΔM = -0.070 ± 0.088 mag
# ln(fp_cc) = -0.47 ± 0.27
# n_cc = 3.06 +0.86 -1.20
#
# Chi squared (MAP): 1674.83
# DOF: 1757
# ---------------------------------


# ----------- Flat wCDM -----------
# H0 = 67.2 ± 2.7 km/s/Mpc
# Ωm = 0.3043 ± 0.0072
# rd = 148.7 +5.3 -6.4 Mpc
# w = -0.934 ± 0.035 (prior ~ U[-1.5, -0.5])
#
# ΔM = -0.082 +0.091 -0.081 mag
# ln(fp_cc) = -0.48 ± 0.27
# n_cc = 3.07 +0.88 -1.2
#
# Chi squared (MAP): 1677.53
# DOF: 1757
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < 0 enforced in the likelihood
#
# H0 = 66.7 ± 2.8 km/s/Mpc
# rd = 149.0 +5.4 -6.4 Mpc
# Ωm = 0.320 +0.013 -0.0091
# w0 = -0.837 ± 0.072 (prior ~ U[-1.5, 0])
# wa = -0.68 ± 0.45 (prior ~ U[-3, 3])
#
# ΔM = -0.081 +0.091 -0.082 mag
# ln(fp_cc) = -0.45 ± 0.27
# n_cc = 3.09 +0.88 -1.2
#
# Chi squared (MAP): 1675.22
# DOF: 1756
# ---------------------------------


# ---------- Flat w0waCDM ---------
# at z_pivot = 0.168
# w0 + wa < 0 enforced in the likelihood

# H0 = 66.8 ± 2.7 km/s/Mpc
# rd = 149.0 +5.4 -6.4 Mpc
# Ωm = 0.321 +0.013 -0.0096
# wp = -0.936 ± 0.034 (prior ~ U[-1.5, 0])
# wa = -0.69 ± 0.44 (prior ~ U[-3, 3])
#
# ΔM = -0.081 ± 0.087 mag
# ln(fp_cc) = -0.45 ± 0.27
# n_cc = 3.07 +0.86 -1.20
#
# Chi squared (MAP): 1678.63
# DOF: 1756
# ---------------------------------


# --- Assuming standard early times physics with rdrag = rdrag(Obh2, Omh2) ---


# ----------- Flat ΛCDM -----------
# H0 = 68.2 ± 2.7 km/s/Mpc
# Ωb h^2 = 0.0218 ± 0.0033
# Ωm = 0.3083 ± 0.0068
# rd = 147.8 +5.3 -6.1 Mpc
#
# ΔM = -0.064 ± 0.085 mag
# ln(fp_cc) = -0.48 ± 0.27
# n_cc = 3.04 +0.86 -1.20
#
# log likelihood (MAP): -960.95
# DOF: 1758
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.10563 positive z > 0.10563 negative
# z_cosmo = z_cmb ± Δz

# H0 = 68.5 ± 2.7 km/s/Mpc
# Ωb h^2 = 0.0221 ± 0.0034
# Ωm = 0.3033 ± 0.0070
# 1000 Δz = 0.53 ± 0.20
# rd = 147.8 +5.4 -6.2 Mpc
#
# ΔM = -0.061 ± 0.086 mag
# ln(fp_cc) = -0.47 ± 0.27
# n_cc = 3.04 +0.86 -1.2
#
# log likelihood (MAP): -957.40
# DOF: 1757
# ---------------------------------


# ----------- Flat wCDM -----------
# H0 = 67.5 ± 2.7 km/s/Mpc
# Ωb h^2 = 0.0231 ± 0.0035
# Ωm = 0.3042 ± 0.0073
# w = -0.934 ± 0.035
# rd = 147.9 +5.3 -6.2 Mpc
#
# ΔM = -0.071 ± 0.086 mag
# ln(fp_cc) = -0.47 ± 0.27
# n_cc = 3.03 +0.85 -1.2
#
# log likelihood (MAP): -959.20
# DOF: 1757
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < 0 enforced in the likelihood

# H0 = 67.1 ± 2.7 km/s/Mpc
# Ωb h^2 = 0.0210 +0.0033 -0.0037
# Ωm = 0.321 +0.013 -0.0095
# w0 = -0.837 ± 0.071 (prior ~ U[-1.5, 0])
# wa = -0.68 ± 0.43 (prior ~ U[-3, 3])
# rd = 148.2 +5.4 -6.2 Mpc
#
# ΔM = -0.070 ± 0.086 mag
# ln(fp_cc) = -0.45 ± 0.27
# n_cc = 3.05 +0.86 -1.2
#
# log likelihood (MAP): -957.82
# DOF: 1756
# ---------------------------------


# ---------- Flat w0waCDM ---------
# at z_pivot = 0.168
# w0 + wa < 0 enforced in the likelihood

# H0 = 67.1 ± 2.7 km/s/Mpc
# Ωb h^2 = 0.0210 +0.0033 -0.0037
# Ωm = 0.320 +0.013 -0.0096
# w0 = -0.936 ± 0.034 (prior ~ U[-1.5, 0])
# wa = -0.68 ± 0.44 (prior ~ U[-3, 3])
# rd = 148.2 +5.3 -6.2 Mpc
#
# ΔM = -0.070 ± 0.086 mag
# ln(fp_cc) = -0.46 ± 0.27
# n_cc = 3.06 +0.87 -1.20
#
# log likelihood (MAP): -957.92
# DOF: 1756
# ---------------------------------