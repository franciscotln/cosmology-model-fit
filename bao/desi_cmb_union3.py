from numba import njit
import numpy as np
from interpolator import interp_hermite
from y2026union3_1.data import get_data
from y2025BAO.data import get_data as get_bao_data
from y2024DESBAO.data import get_data as get_des_bao_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
des_bao_legend, des_bao_data, des_bao_cov_matrix = get_des_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(bao_cov_matrix)
inv_cov_des_bao = np.linalg.inv(des_bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, 3000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


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
    return H0 * Ez(z, H0, Obh2=params[2], Och2=params[3], w0=params[4])


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
qty_desi = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)
qty_des = np.array([qty_map[q] for q in des_bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnuh2
    rd = cmb.r_drag(Obh2, Omh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


@njit
def mu_theory(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


@njit
def chi2_sn(params):
    delta_sn = mu_vals - mu_theory(params)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_bao(params):
    delta_bao_des = des_bao_data["value"] - bao_theory(
        des_bao_data["z"], qty_des, params
    )
    chi2_des_bao = delta_bao_des @ inv_cov_des_bao @ delta_bao_des
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], qty_desi, params)
    chi2_desi_bao = delta_bao @ inv_cov_bao @ delta_bao
    return chi2_desi_bao + chi2_des_bao


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    return chi2_cmb + chi2_bao(params) + chi2_sn(params)


bounds = np.array(
    [
        (-1.0, 1.0),  # ΔM
        (60.0, 75.0),  # H0
        (0.010, 0.030),  # ωb = Ωb * h^2
        (0.01, 0.25),  # ωc = Ωc * h^2
        (-1.0, -1 / 3),  # w0
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def q0(Om, w0=-1.0):
    """Calculate the deceleration parameter at z=0"""
    return Om / 2 + (1.0 + 3 * w0) * (1.0 - Om) / 2


def main():
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence
    from gelman_rubin import gelman_rubin
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(6) as pool:
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
    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    one_sigma_ci = [15.9, 50, 84.1]
    [
        (dM_16, dM_50, dM_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, one_sigma_ci, axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    degs_of_freedom = (
        len(z_cmb)
        + len(des_bao_data["z"])
        + len(bao_data["z"])
        + len(cmb.DISTANCE_PRIORS)
        - len(bounds)
    )
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    q0_samples = q0(Om_samples, samples[:, 4])

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_ci)
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, one_sigma_ci)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_ci)
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(z_drag_samples, one_sigma_ci)
    q0_16, q0_50, q0_84 = np.percentile(q0_samples, one_sigma_ci)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r*: {cmb.rs_z(z_st_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"z_d: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}")
    print(f"r_d: {cmb.rs_z(z_dr_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"q0: {q0_50:.3f} +{(q0_84 - q0_50):.3f} -{(q0_50 - q0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degs of freedom: {degs_of_freedom}")

    labels = ["$Δ_M$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=des_bao_data,
        errors=np.sqrt(np.diag(des_bao_cov_matrix)),
        title=des_bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
Union 3.1 SNe 2026
Compressed Planck + ACT
DESI BAO DR2 2025
DES BAO 2025
"""

"""
Flat ΛCDM w(z) = -1
ΔM: -0.050 +0.007 -0.007 mag
H0: 68.43 +0.27 -0.27 km/s/Mpc
Ωm: 0.300 +0.004 -0.004
ωb: 0.02257 +0.00010 -0.00010
ωc: 0.1174 +0.0006 -0.0006
ωm: 0.1406 +0.0006 -0.0006
z*: 1089.41 +0.16 -0.15
r*: 144.97 Mpc
z_d: 1060.20 +0.23 -0.23
r_d: 147.57 Mpc
q0: -0.550 +0.005 -0.005
Chi squared: 45.9
Log evidence: -41.9
Degs of freedom: 35
"""


"""
Flat ΛCDM w(z) = -1, varying the absolute mag SNe: M(z) = ΔM + tanh(1 - z^(0.1 * p))
ΔM: -0.064 +0.010 -0.010 mag
p: 0.194 +0.098 -0.095 (prior U(-0.4, 0.8))
H0: 68.51 +0.27 -0.27 km/s/Mpc
Ωm: 0.299 +0.004 -0.004
ωb: 0.02258 +0.00010 -0.00010
ωc: 0.1172 +0.0007 -0.0007
ωm: 0.1404 +0.0006 -0.0006
z*: 1089.37 +0.15 -0.15
r*: 145.01 Mpc
z_d: 1060.21 +0.23 -0.23
r_d: 147.61 Mpc
q0: -0.551 +0.005 -0.005
Chi squared: 41.7
Log evidence: -41.4
Degs of freedom: 34
"""


"""
Flat wCDM w(z) = w0
ΔM: -0.054 +0.011 -0.011 mag
H0: 68.11 +0.69 -0.68 km/s/Mpc
Ωm: 0.303 +0.006 -0.006
ωb: 0.02259 +0.00011 -0.00010
ωc: 0.1171 +0.0008 -0.0009
ωm: 0.1403 +0.0008 -0.0008
w0: -0.986 +0.027 -0.028 (prior U(-1.3, -0.5))
z*: 1089.36 +0.17 -0.17
r*: 145.03 Mpc
z_d: 1060.21 +0.23 -0.23
r_d: 147.63 Mpc
q0: -0.532 +0.035 -0.036
Chi squared: 45.6
Log evidence: -44.2
Degs of freedom: 34
"""


"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
ΔM: -0.062 +0.009 -0.009 mag
H0: 67.12 +0.73 -0.77 km/s/Mpc
Ωm: 0.3110 +0.0072 -0.0066
ωb: 0.02260 +0.00010 -0.00010
ωc: 0.1168 +0.0007 -0.0007
ωm: 0.1401 +0.0007 -0.0007
w0: -0.894 +0.058 -0.055 (prior U(-1.0, -1/3). Truncated posterior. 1.93 sigma to the prior left edge)
z*: 1089.32 +0.16 -0.16
r*: 145.09 Mpc
z_d: 1060.22 +0.23 -0.23
r_d: 147.69 Mpc
q0: -0.425 +0.068 -0.065
Chi squared: 43.0
Log evidence: -42.0
Degs of freedom: 34
"""


"""
Flat w(z) = w0 + wa * z / (1 + z)
ΔM: -0.048 +0.011 -0.011 mag
H0: 66.82 +0.80 -0.79 km/s/Mpc
Ωm: 0.3180 +0.0080 -0.0078
ωb: 0.02251 +0.00011 -0.00011
ωc: 0.1188 +0.0010 -0.0010
ωm: 0.1420 +0.0009 -0.0009
w0: -0.754 +0.082 -0.080 (prior U(-1.5, 0.0))
wa: -0.81 +0.27 -0.29 (prior U(-2.5, 1.0))
z*: 1089.61 +0.19 -0.19
r*: 144.64 Mpc
z_d: 1060.17 +0.23 -0.23
r_d: 147.26 Mpc
q0: -0.272 +0.090 -0.090
Chi squared: 36.1
Log evidence: -41.7 (removed excluded volume from constraint wa + w0 > 0)
Degs of freedom: 33
"""
