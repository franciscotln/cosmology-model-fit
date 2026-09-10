from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data_fs_lya import get_data as get_bao_data
from y2025DESdovekie.data import (
    effective_sample_size as sn_sample,
    get_data as get_sn_data,
)

cc_legend, z_cc_vals, H_cc_vals, H_cc_err, cov_mat_sys_cc = get_cc_data(split_sys=True)
sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

N_cc = len(z_cc_vals)

c = c0 / 1000  # km/s

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]

# ----- PARAMS -----
names = ["ln_fp", "n_cc", "dM", "h0", "rd", "Om", "v100"]
labels = ["ln(f_{p,cc})", "n_{cc}", "ΔM", "H_0", "r_d", "Ω_m", "v_{100}"]
bounds = np.array(
    [
        (np.log(0.3), np.log(1.2)),  # ln(fp): CC error rescaling (overestimated)
        (-4, +4),  # n_cc: CC error rescaling power (overestimated)
        (-0.55, +0.55),  # ΔM: magnitude offset
        (45, 90),  # H0: Hubble constant at present
        (110, 175),  # r_d: sound horizon at drag epoch
        (0.2, 0.7),  # Ωm: matter density parameter at present
        (-4.5, +4.5),  # v x 100 km/s
    ]
)
# ------------------

@njit
def rho_de(z, w0):
    cubed = (1.0 + z) ** 3
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2  # wzCDM
    # return 1.0  # ΛCDM
    # return cubed ** (1.0 + w0)  # wCDM
    # return cubed ** (1.0 + w0 + wa) * np.exp(-3 * wa * z / (1.0 + z))  # w0waCDM


@njit
def H_z(z, theta):
    H0, Om = theta[3], theta[5]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_grid(theta):
    dh_grid = c / H_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return (cum_dm, dh_grid)


@njit
def DM_z(z, dm_grid):
    return interp_hermite(z, z_grid, dm_grid[0], dm_grid[1])


@njit
def DH_z(z, dm_grid):
    return interp_pchip(z, z_grid, dm_grid[1])


@njit
def DV_z(z, DM, DH):
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
quantities = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, rdrag, dm_grid):
    DM = DM_z(z, dm_grid)
    DH = DH_z(z, dm_grid)
    inv_rd = 1.0 / rdrag

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
    # Heaviside step function
    v_km_s = 100 * params[6] * np.where(z_cmb <= 0.10563, 1, -1)
    z_pec = v_km_s / c
    return -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)


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
    dm_grid = DM_grid(theta)

    z_cosmo = get_z_cosmo(theta)
    DM_cosmo = DM_z(z_cosmo, dm_grid)
    delta_sn = mu_values - mu_theory(theta, DM_cosmo)
    y_sn = solve_triangular(cho_sn, delta_sn)
    chi_sn = np.dot(y_sn, y_sn)

    delta_bao = bao["value"] - bao_theory(bao["z"], quantities, theta[4], dm_grid)
    y_bao = solve_triangular(cho_bao, delta_bao)
    chi_bao = np.dot(y_bao, y_bao)

    delta_cc = H_cc_vals - H_z(z_cc_vals, theta)
    y_cc = solve_triangular(cho_cc, delta_cc)
    chi_cc = np.dot(y_cc, y_cc)

    return chi_sn + chi_bao + chi_cc


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


z_pivot = 0.615


@njit
def log_likelihood(theta):
    fp, n_cc = np.exp(theta[0]), theta[1]
    fz_cc = fp * ((1.0 + z_cc_vals) / (1.0 + z_pivot)) ** n_cc
    cov_mat_cc = np.diag(H_cc_err ** 2 * fz_cc**2) + cov_mat_sys_cc
    cho_cc = np.linalg.cholesky(cov_mat_cc)
    logdet_cc = 2.0 * np.sum(np.log(np.diag(cho_cc)))
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
    from log_evidence import log_evidence

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 1000
    nsteps = 3000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

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
    log_evd = log_evidence(flat_samples, flat_log_probs, log_probability, bounds)

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
    try:
        gd_samples.addDerived(100 * gd_samples["v100"], name="v_km_s", label="v_{km/s}")
        gd_samples.updateBaseStatistics()
    except Exception as e:
        # v100 not present in the samples
        pass

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    MAP_PARAMS = flat_samples[np.argmax(flat_log_probs)]
    fz_cc = np.exp(MAP_PARAMS[0]) * ((1.0 + z_cc_vals) / (1.0 + z_pivot)) ** MAP_PARAMS[1]
    cov_mat_cc = np.diag(H_cc_err ** 2 * fz_cc**2) + cov_mat_sys_cc
    cho_cc = np.linalg.cholesky(cov_mat_cc)

    DOF = sn_sample + len(bao) + N_cc - ndim

    print(f"Chi squared (MAP): {chi_squared(MAP_PARAMS, cho_cc):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"DOF: {DOF}")

    dm_grid = DM_grid(MAP_PARAMS)

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, MAP_PARAMS[4], dm_grid),
        data=bao,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=f"{bao_legend}: $r_d$={MAP_PARAMS[4]:.1f} Mpc",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, MAP_PARAMS),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=H_cc_err,
        label=f"{cc_legend} $H_0$: {MAP_PARAMS[3]:.1f} km/s/Mpc",
        err_scaling=1 / fz_cc,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(MAP_PARAMS, dm_grid),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(MAP_PARAMS, DM_z(z_cmb, dm_grid)),
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
# H0 = 67.3 ± 2.9 km/s/Mpc
# rd = 149.9 +5.8 -6.5 Mpc
# Ωm = 0.3077 ± 0.0068
# ln(fp_cc) = -0.50 +0.11 -0.13
# n_cc = 1.38 ± 0.47
# ΔM = -0.094 ± 0.093 mag
# Chi squared (MAP): 1685.78
# Log evidence: -990.86
# DOF: 1761
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.10563 inflow z > 0.10563 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# H0 = 67.6 ± 2.8 km/s/Mpc
# rd = 149.8 +5.6 -6.6 Mpc
# Ωm = 0.3027 ± 0.0070
# v = -152 ± 56 km/s (prior ~ U[-450, 450])
# ln(fp_cc) = -0.50 +0.11 -0.13
# n_cc = 1.39 ± 0.47
# ΔM = -0.089 ± 0.090 mag
# Chi squared (MAP): 1677.51 (2.88 sigma)
# Log evidence: -989.03
# DOF: 1760
# ---------------------------------


# ----------- Flat wCDM -----------
# H0 = 66.5 ± 2.8 km/s/Mpc
# rd = 150.2 +5.6 -6.7 Mpc
# Ωm = 0.3040 ± 0.0073
# w0 = -0.937 ± 0.035 (prior ~ U[-1.5, -0.5])
# ln(f_p) = -0.49 +0.11 -0.13
# n_cc = 1.37 ± 0.47
# ΔM = -0.103 ± 0.091 mag
# Chi squared (MAP): 1683.69 (1.44 sigma)
# Log evidence: -991.66
# DOF: 1760
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
#
# H0 = 66.2 ± 2.9 km/s/Mpc
# rd = 150.5 +5.7 -6.8 Mpc
# Ωm = 0.3089 ± 0.0069
# w0 = -0.889 ± 0.048 (prior ~ U[-1, -1/3])
# ln(fp_cc) = -0.49 +0.11 -0.13
# n_cc = 1.39 ± 0.47
# ΔM = -0.106 ± 0.092 mag
# Chi squared (MAP): 1680.41 (2.32 sigma)
# Log evidence: -990.61
# DOF: 1760
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < 0 enforced in the likelihood
#
# H0 = 66.2 ± 2.8 km/s/Mpc
# rd = 150.3 ± 6.3 Mpc
# Ωm = 0.321 +0.013 -0.0093
# w0 = -0.835 ± 0.073 (prior ~ U[-1.5, 0])
# wa = -0.72 ± 0.45 (prior ~ U[-3, 2])
# ln(fp_cc) = -0.49 +0.11 -0.13
# n_cc = 1.42 ± 0.48
# ΔM = -0.099 ± 0.091 mag
# Chi squared (MAP): 1677.47 (2.15 sigma)
# Log evidence: -992.23
# DOF: 1759
# ---------------------------------


# ---------- Flat w0waCDM ---------
# at z_pivot = 0.168
# w0 + wa < 0 enforced in the likelihood

# ln(fp_cc) = -0.49 +0.11 -0.13
# n_cc = 1.43 ± 0.48
# ΔM = -0.0996 ± 0.0900 mag
# H0 = 66.2 ± 2.8 km/s/Mpc
# rd = 150.4 +5.8 -6.5 Mpc
# Ωm = 0.321 +0.012 -0.0094
# wp = -0.939 ± 0.035 (prior ~ U[-1.5, 0]) => 1.74 sigma from -1
# wa = -0.73 ± 0.44 (prior ~ U[-3, 2]) => 1.66 sigma from 0
# Chi squared (MAP): 1679.75
# Log evidence: -992.22
# DOF: 1759
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < 0 enforced in the likelihood
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.10563 inflow z > 0.10563 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)
#
# H0 = 67.2 ± 3.0 km/s/Mpc
# rd = 150.1 +5.7 -6.8 Mpc
# Ωm = 0.303 +0.026 -0.013
# w0 = -0.949 +0.089 -0.130 (prior ~ U[-1.5, 0])
# wa = -0.17 ± 0.62 (prior ~ U[-3, 2])
# v = -122 ± 98 km/s
# ln(fp_cc) = -0.50 +0.11 -0.13
# n_cc = 1.40 ± 0.47
# ΔM = -0.093 +0.096 -0.087 mag
# Chi squared (MAP): 1679.09
# Log evidence: -992.76
# DOF: 1758
# ---------------------------------
