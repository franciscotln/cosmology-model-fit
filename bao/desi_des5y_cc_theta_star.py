from numba import njit
import numpy as np
from scipy.linalg import cho_factor
import cmb.data_planck_act_compression as cmb
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
from y2025DESdovekie.data import (
    effective_sample_size as sn_size,
    get_data as get_sn_data,
)
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, z_sn_hel_vals, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

z_max = max(np.max(z_sn_vals), np.max(bao["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1 + w0 + (1 - w0) * zp1**3)) ** 2


@njit
def Ez(z, H0, Obh2, Och2):
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return np.sqrt(radiation_term + matter_term + neutrino_term + dark_energy_term)


@njit
def H_z(z, th):
    H0 = th[2]
    return H0 * Ez(z, H0=H0, Obh2=th[3], Och2=th[4])


cmb.set_HZ(H_z)


@njit
def DM_grid(theta):
    dh_grid = c / H_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return (cum_dm, dh_grid)


@njit
def DM_z(z, dm_interp):
    y_vals, y_derivs = dm_interp
    return interp_hermite(z, z_grid, y_vals, y_derivs)


@njit
def DH_z(z, dm_interp):
    return interp_pchip(z, z_grid, dm_interp[1])


@njit
def DV_z(z, DH, DM):
    return (z * DH * DM**2) ** (1 / 3)


dv_rs, dm_rs, dh_rs = 0, 1, 2
qty_map = {"DV_over_rs": dv_rs, "DM_over_rs": dm_rs, "DH_over_rs": dh_rs}
quantities = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, theta, dm_interp):
    Obh2, Och2 = theta[3], theta[4]
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnu_h2)

    DM = DM_z(z, dm_interp)
    DH = DH_z(z, dm_interp)
    results = np.empty(z.size, dtype=np.float64)

    DV_mask = qty == dv_rs
    DM_mask = qty == dm_rs
    DH_mask = qty == dh_rs
    results[DH_mask] = DH[DH_mask] / rd
    results[DM_mask] = DM[DM_mask] / rd
    results[DV_mask] = DV_z(z[DV_mask], DH[DV_mask], DM[DV_mask]) / rd
    return results


@njit
def get_z_cosmo(params):
    # Heaviside step at z = 0.11
    v_km_s = 100 * params[5] * np.where(z_sn_vals <= 0.11, 1, -1)
    return -1.0 + (1.0 + z_sn_vals) / (1.0 + v_km_s / c)


def mu_corr(params, z_obs):
    # For plotting purposes only
    z_cosmo = get_z_cosmo(params)
    DM_interp = DM_grid(params)
    return 5.0 * np.log10(DM_z(z_cosmo, DM_interp) / DM_z(z_obs, DM_interp))


@njit
def mu_theory(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_sn_hel_vals) * DM)


@njit
def chi_squared(theta):
    z_cosmo = get_z_cosmo(theta)
    if np.any(z_cosmo <= 0):
        return np.inf

    DM_interp = DM_grid(theta)

    delta_sn = mu_values - mu_theory(theta[1], DM_z(z_cosmo, DM_interp))
    y_sn = solve_triangular(cho_sn, delta_sn)
    chi_sn = np.dot(y_sn, y_sn)

    delta_bao = bao["value"] - bao_theory(bao["z"], quantities, theta, DM_interp)
    y_bao = solve_triangular(cho_bao, delta_bao)
    chi_bao = np.dot(y_bao, y_bao)

    delta_cc = H_cc_vals - H_z(z_cc_vals, theta)
    y_cc = solve_triangular(cho_cc, delta_cc)
    chi_cc = np.dot(y_cc, y_cc) * theta[0] ** 2

    delta = (cmb.DISTANCE_PRIORS - cmb.cmb_distances(theta[3], theta[4], theta))[1]
    chi_theta_star = delta**2 / cmb.covariance[1, 1]

    return chi_theta_star + chi_sn + chi_bao + chi_cc


bounds = np.array(
    [
        (0.5, 2.5),  # f_cc: CC error rescaling (overestimated)
        (-0.60, 0.60),  # ΔM: magnitude offset
        (50.0, 85.0),  # H0: Hubble constant at present
        (0.005, 0.035),  # Ωb x h^2: baryon density parameter
        (0.05, 0.30),  # Ωc x h^2: cold dark matter density parameter at present
        (-4.5, 4.5),  # v km/s
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
    f_cc = theta[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(theta) - 0.5 * normalization_cc


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

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.2),
        (emcee.moves.DEMove(), 0.8),
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
        (f_cc_16, f_cc_50, f_cc_84),
        (dM_16, dM_50, dM_84),
        (h0_16, h0_50, h0_84),
        (wb_16, wb_50, wb_84),
        (wc_16, wc_50, wc_84),
        (v_16, v_50, v_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    DOF = 1 + sn_size + len(bao) + len(z_cc_vals) - ndim

    Omh2_samples = samples[:, 3] + samples[:, 4] + Omnu_h2
    Om_samples = Omh2_samples / (samples[:, 2] / 100) ** 2
    r_d_samples = cmb.r_drag(samples[:, 3], Omh2_samples)
    rd_16, rd_50, rd_84 = np.percentile(r_d_samples, [15.9, 50, 84.1])
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])

    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"ωb: {wb_50:.4f} +{(wb_84 - wb_50):.4f} -{(wb_50 - wb_16):.4f} Mpc")
    print(f"ωc: {wc_50:.4f} +{(wc_84 - wc_50):.4f} -{(wc_50 - wc_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"v: {v_50:.3f} +{(v_84 - v_50):.3f} -{(v_50 - v_16):.3f} x 100 km/s")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degrees of freedom: {DOF}")

    dm_interp = DM_grid(best_fit)
    labels = ["$f_{CCH}$", "ΔM", "$H_0$", "$ω_b$", "$ω_c$", "$v_{100}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit, dm_interp),
        data=bao,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_50,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_values - mu_corr(best_fit, z_sn_vals),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit[1], DM_z(z_sn_vals, dm_interp)),
        label=rf"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# Flat ΛCDM: w(z) = -1
# H0: 67.2 +1.3 -1.2 km/s/Mpc
# ωb: 0.0206 +0.0019 -0.0017 Mpc
# ωc: 0.1161 +0.0011 -0.0010
# ωm: 0.1373 +0.0028 -0.0024
# Ωm: 0.304 +0.007 -0.007
# r_d: 150.2 +2.3 -2.5 Mpc
# ΔM: -0.097 +0.039 -0.036 mag
# f_cc: 1.51 +0.17 -0.17
# Chi squared: 1683.73
# Log evidence: -990.20
# Degrees of freedom: 1761
# ---------------------------------


# Flat ΛCDM: w(z) = -1 with velocity step correction
# H0: 68.2 +1.4 -1.3 km/s/Mpc
# ωb: 0.0219 +0.0020 -0.0019 Mpc
# ωc: 0.1162 +0.0012 -0.0011
# ωm: 0.1387 +0.0031 -0.0027
# Ωm: 0.298 +0.007 -0.007
# v: -1.62 +0.56 -0.57 (prior ~U[-4.5, 4.5]) x 100 km/s
# r_d: 148.7 +2.4 -2.5 Mpc
# ΔM: -0.070 +0.041 -0.039 mag
# f_cc: 1.51 +0.18 -0.17
# Chi squared: 1675.78 (2.82 sigma significance)
# Log evidence: -987.98 (Δ logZ = 2.22 in favour of ΛCDM with velocity step for SNe)
# Degrees of freedom: 1760
# ---------------------------------


# Flat wCDM: w(z) = w0
# H0: 68.6 +1.6 -1.5 km/s/Mpc
# ωb: 0.0251 +0.0030 -0.0028 Mpc
# ωc: 0.1151 +0.0015 -0.0014
# ωm: 0.1407 +0.0041 -0.0035
# Ωm: 0.299 +0.007 -0.007
# w0: -0.918 +0.031 -0.032 (prior ~U[-1.5, -0.5])
# r_d: 145.5 +3.2 -3.3 Mpc
# ΔM: -0.033 +0.051 -0.048 mag
# f_cc: 1.51 +0.18 -0.17
# Chi squared: 1677.51 (2.49 sigma significance)
# Log evidence: -989.62 (Δ logZ = 0.58 in favour of wCDM)
# Degrees of freedom: 1760
# ---------------------------------


# Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
# H0: 67.9 +1.5 -1.4 km/s/Mpc
# ωb: 0.0238 +0.0025 -0.0023 Mpc
# ωc: 0.1161 +0.0015 -0.0012
# ωm: 0.1405 +0.0038 -0.0032
# Ωm: 0.305 +0.007 -0.007
# w0: -0.864 +0.047 -0.049 (prior ~U[-1, -1/3])
# r_d: 146.5 +2.8 -3.0 Mpc
# ΔM: -0.047 +0.047 -0.043 mag
# f_cc: 1.51 +0.18 -0.17
# Chi squared: 1676.49 (2.69 sigma significance)
# Log evidence: -988.31 (Δ logZ = 1.89 in favour of wzCDM)
# Degrees of freedom: 1760
# ---------------------------------


# Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
# w0 + wa < 0 enforced in the likelihood with 0.33 correction in evidence
#
# H0: 67.3 +1.8 -1.7 km/s/Mpc
# ωb: 0.0224 +0.0035 -0.0031 Mpc
# ωc: 0.1172 +0.0018 -0.0022
# ωm: 0.1399 +0.0037 -0.0030
# Ωm: 0.309 +0.011 -0.011
# w0: -0.861 +0.064 -0.060 (prior ~U[-1.5, 0])
# wa: -0.38 +0.34 -0.37 (prior ~U[-2.5, 2])
# r_d: 147.9 +3.4 -3.7 Mpc
# ΔM: -0.066 +0.055 -0.050 mag
# f_cc: 1.50 +0.17 -0.17
# Chi squared: 1676.37 (2.24 sigma significance)
# Log evidence: -991.14 + 0.33 (Δ logZ = -0.61 in favour of ΛCDM)
# Degrees of freedom: 1759
# ---------------------------------
