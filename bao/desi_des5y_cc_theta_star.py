from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_planck_act_compression as cmb
from interpolator import interp_quad
from y2025DESdovekie.data import (
    effective_sample_size as sn_size,
    get_data as get_sn_data,
)
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, z_sn_hel_vals, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(cov_matrix_bao)
inv_cov_cc = np.linalg.inv(cov_matrix_cc)

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0, wa):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1 + w0 + (1 - w0) * zp1**3)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0=-1.0, wa=0.0):
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0, wa)

    return np.sqrt(radiation_term + matter_term + neutrino_term + dark_energy_term)


@njit
def H_z(z, theta):
    H0, Obh2, Och2, w0 = theta[2:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


@njit
def DH_z(z, theta):
    return c / H_z(z, theta)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_quad(z, z_grid, cum_dm)


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, theta):
    Obh2, Och2 = theta[3], theta[4]
    rd = cmb.r_drag(Obh2, Obh2 + Och2 + Omnu_h2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


@njit
def mu_theory(theta):
    dL = (1.0 + z_sn_hel_vals) * DM_z(z_sn_vals, theta)
    return theta[1] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return y @ y


def chi_squared(theta):
    delta = (cmb.DISTANCE_PRIORS - cmb.cmb_distances(H_z, theta[3], theta[4], theta))[1]
    chi_theta_star = delta**2 / cmb.covariance[1, 1]

    delta_sn = mu_values - mu_theory(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_cc = H_cc_vals - H_z(z_cc_vals, theta)
    chi_cc = delta_cc @ inv_cov_cc @ delta_cc * theta[0] ** 2

    return chi_theta_star + chi_sn + chi_bao + chi_cc


bounds = np.array(
    [
        (0.5, 2.5),  # f_cc: CC error rescaling (overestimated)
        (-0.60, 0.60),  # ΔM: magnitude offset
        (50.0, 85.0),  # H0: Hubble constant at present
        (0.005, 0.035),  # Ωb x h^2: baryon density parameter
        (0.05, 0.30),  # Ωc x h^2: cold dark matter density parameter at present
        (-1.0, -1 / 3),  # w0: dark energy equation of state at present
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(theta):
    f_cc = theta[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(theta) - 0.5 * normalization_cc


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def main():
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from cosmic_chronometers.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions
    from log_evidence import log_evidence

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.25),
        (emcee.moves.DEMove(), 0.75),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff7f0e"}
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
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    deg_of_freedom = 1 + sn_size + len(bao_data["z"]) + len(z_cc_vals) - ndim

    Omh2_samples = samples[:, 3] + samples[:, 4] + Omnu_h2
    Om_samples = Omh2_samples / (samples[:, 2] / 100) ** 2
    r_d_samples = cmb.r_drag(samples[:, 3], Omh2_samples)
    rd_16, rd_50, rd_84 = np.percentile(r_d_samples, [15.9, 50, 84.1])
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"ωb: {wb_50:.4f} +{(wb_84 - wb_50):.4f} -{(wb_50 - wb_16):.4f} Mpc")
    print(f"ωc: {wc_50:.4f} +{(wc_84 - wc_50):.4f} -{(wc_50 - wc_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    labels = ["$f_{CCH}$", "ΔM", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
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
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=rf"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
f_cc: 1.48 +0.19 -0.17
H0: 67.3 +1.4 -1.2 km/s/Mpc
ωb: 0.0207 +0.0020 -0.0018 Mpc
ωc: 0.1162 +0.0011 -0.0010
ωm: 0.1375 +0.0029 -0.0025
Ωm: 0.303 +0.007 -0.007
w0: -1
wa: 0
r_d: 150.0 +2.3 -2.5 Mpc
Chi squared: 1678.70
Log evidence: -971.51
Degrees of freedom: 1756

===============================

Flat wCDM: w(z) = w0
f_cc: 1.48 +0.18 -0.18
H0: 68.7 +1.6 -1.5 km/s/Mpc
ωb: 0.0252 +0.0031 -0.0028 Mpc
ωc: 0.1152 +0.0016 -0.0014
ωm: 0.1409 +0.0042 -0.0035
Ωm: 0.299 +0.007 -0.007
w0: -0.917 +0.032 -0.032 (prior width 1.0: -1.5 to -0.5)
wa: 0
r_d: 145.3 +3.2 -3.4 Mpc
Chi squared: 1672.68
Log evidence: -970.83 (Δ logZ = 0.68 over ΛCDM)
Degrees of freedom: 1755

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
f_cc: 1.48 +0.18 -0.18
H0: 68.0 +1.5 -1.4 km/s/Mpc
ωb: 0.0240 +0.0026 -0.0023 Mpc
ωc: 0.1161 +0.0015 -0.0013
ωm: 0.1407 +0.0039 -0.0033
Ωm: 0.305 +0.007 -0.007
w0: -0.863 +0.047 -0.049 (prior width 2/3: -1.0 to -1/3)
wa: d w(z)/dz at z=0 = -(3/2) * (1 - w0^2)
r_d: 146.4 +2.8 -3.0 Mpc
Chi squared: 1671.65
Log evidence: -969.56 (Δ logZ = 1.95 over ΛCDM)
Degrees of freedom: 1755

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
TODO
w0: (prior width 1.5: -1.5 to 0.0)
wa: (prior width 5.0: -3.0 to 2.0)
"""
