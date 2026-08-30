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

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = c0 / 1000  # km/s

z_max = max(np.max(z_cmb), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


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
    zp1 = 1.0 + z
    cubed = zp1 * zp1 * zp1
    return H0 * np.sqrt(Om * cubed + (1.0 - Om))


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
    return (z * DH * DM * DM) ** (1 / 3)


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
def chi_squared(theta, f_array):
    dm_grid = DM_grid(theta)

    z_cosmo = get_z_cosmo(theta)
    DM_cosmo = DM_z(z_cosmo, dm_grid)
    delta_sn = mu_values - mu_theory(theta, DM_cosmo)
    chi_sn = solve_triangular(cho_sn, delta_sn)

    delta_bao = bao["value"] - bao_theory(bao["z"], quantities, theta[4], dm_grid)
    chi_bao = solve_triangular(cho_bao, delta_bao)

    delta_cc = H_cc_vals - H_z(z_cc_vals, theta)
    chi_cc = solve_triangular(cho_cc, f_array * delta_cc)

    return chi_sn + chi_bao + chi_cc


bounds = np.array(
    [
        (0.1, 4.0),  # f0: CC error rescaling (overestimated)
        (-2.0, 2.0),  # fa: CC error rescaling (overestimated)
        (-0.55, 0.55),  # ΔM: magnitude offset
        (50.0, 85.0),  # H0: Hubble constant at present
        (110.0, 175.0),  # r_d: sound horizon at drag epoch
        (0.2, 0.7),  # Ωm: matter density parameter at present
        (-4.5, 4.5),  # v x 100 km/s
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


@njit
def log_likelihood(theta):
    f0, fa = theta[0], theta[1]
    f_array = f0 + fa * z_cc_vals
    if np.any(f_array <= 1e-4):
        # ensure f_array is positive
        return -np.inf

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2.0 * np.log(f_array).sum()
    return -0.5 * chi_squared(theta, f_array) - 0.5 * normalization_cc


@njit
def log_probability_jit(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def log_probability(theta):
    return log_probability_jit(theta)


def main():
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from ohd.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions
    from log_evidence import log_evidence

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2500 + burn_in
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

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    [
        (f0_16, f0_50, f0_84),
        (fa_16, fa_50, fa_84),
        (dM_16, dM_50, dM_84),
        (h0_16, h0_50, h0_84),
        (rd_16, rd_50, rd_84),
        (Om_16, Om_50, Om_84),
        (v_16, v_50, v_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    MAP_PARAMS = samples[np.argmax(log_probs)]
    f_array = MAP_PARAMS[0] + MAP_PARAMS[1] * z_cc_vals

    DOF = sn_sample + len(bao) + N_cc - ndim

    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"f0_cc: {f0_50:.2f} +{(f0_84 - f0_50):.2f} -{(f0_50 - f0_16):.2f}")
    print(f"fa_cc: {fa_50:.2f} +{(fa_84 - fa_50):.2f} -{(fa_50 - fa_16):.2f}")
    print(f"Chi squared (MAP): {chi_squared(MAP_PARAMS, f_array):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"DOF: {DOF}")

    dm_grid = DM_grid(MAP_PARAMS)

    labels = ["$f_{0,CCH}$", "$f_{a,CCH}$", "ΔM", "$H_0$", "$r_d$", "$Ω_m$", "$v_{100}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, rd_50, dm_grid),
        data=bao,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=f"{bao_legend}: $r_d$={rd_50:.1f} Mpc",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, MAP_PARAMS),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_array,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values - mu_corr(MAP_PARAMS, dm_grid),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(MAP_PARAMS, DM_z(z_cmb, dm_grid)),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# H0: 67.9 +2.0 -2.0 km/s/Mpc
# r_d: 148.2 +4.5 -4.2 Mpc
# Ωm: 0.308 +0.007 -0.007
# ΔM: -0.072 +0.063 -0.065 mag
# f0_cc: 2.27 +0.36 -0.35
# fa_cc: -0.83 +0.31 -0.28
# Chi squared (MAP): 1685.49
# Log evidence: -991.86
# DOF: 1761
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.10563 inflow z > 0.10563 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# v: -1.52 +0.56 -0.57 x 100 km/s (prior ~ U[-4.5, 4.5])
# H0: 68.1 +2.0 -2.0 km/s/Mpc
# r_d: 148.4 +4.5 -4.3 Mpc
# Ωm: 0.303 +0.007 -0.007
# ΔM: -0.071 +0.063 -0.065 mag
# f0_cc: 2.28 +0.35 -0.35
# fa_cc: -0.84 +0.31 -0.28
# Chi squared (MAP): 1679.38 (2.47 sigma significance)
# Log evidence: -990.10 (Δ logZ = 1.76 in favour of v step corrections)
# DOF: 1760
# ---------------------------------


# ----------- Flat wCDM -----------
# w0: -0.936 +0.035 -0.035 (prior ~ U[-1.5, -0.5])
# H0: 67.2 +2.0 -2.0 km/s/Mpc
# r_d: 148.4 +4.4 -4.3 Mpc
# Ωm: 0.304 +0.007 -0.007
# ΔM: -0.079 +0.063 -0.064 mag
# f0_cc: 2.25 +0.35 -0.34
# fa_cc: -0.81 +0.30 -0.28
# Chi squared (MAP): 1682.69 (1.67 sigma away from ΛCDM)
# Log evidence: -992.66 (Δ logZ = -0.80 in favour of ΛCDM)
# DOF: 1760
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
#
# w0: -0.889 +0.049 -0.049 (prior ~ U[-1, -1/3])
# wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
# H0: 66.9 +2.0 -2.0 km/s/Mpc
# r_d: 148.6 +4.6 -4.3 Mpc
# Ωm: 0.309 +0.007 -0.007
# ΔM: -0.081 +0.063 -0.065 mag
# f0_cc: 2.26 +0.36 -0.35
# fa_cc: -0.82 +0.31 -0.28
# Chi squared (MAP): 1681.46 (2.01 sigma away from ΛCDM)
# Log evidence: -991.22 (Δ logZ = 0.64 in favour of wzCDM)
# DOF: 1760
# ---------------------------------


# ---------- Flat w0waCDM ---------
# TODO: re-run after CCH updates
#
# w0 + wa < 0 enforced in the likelihood
# 
# w0: -0.845 +0.073 -0.070 (prior ~ U[-1.5, 0])
# wa: -0.654 +0.438 -0.434 (prior ~ U[-3, 2])
# H0: 67.4 +2.2 -2.2 km/s/Mpc
# r_d: 147.2 +4.9 -4.5 Mpc
# Ωm: 0.321 +0.011 -0.012
# ΔM: -0.06 +0.07 -0.07 mag
# f0_cc: 1.49 +0.17 -0.17
# fa_cc: -0.80 +0.18 -0.17
# Chi squared: 1677.85 (2.07 sigma away from ΛCDM)
# Log evidence: -989.61 (Δ logZ = -1.5 in favour of ΛCDM)
# DOF: 1759
# ---------------------------------
