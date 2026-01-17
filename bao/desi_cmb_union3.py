from numba import njit
import numpy as np
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data
import cmb.data_early_lcdm_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0, wa):
    a3 = 1 / (1.0 + z) ** 3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0, wa)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0 = params[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


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
    dL = (1.0 + z_sn_vals) * DM_z(z_sn_vals, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


@njit
def chi2_sn(params):
    delta_sn = mu_vals - mu_theory(params)
    return delta_sn @ inv_cov_sn @ delta_sn


@njit
def chi2_bao(params):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    return delta_bao @ inv_cov_bao @ delta_bao


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, *params[1:])
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    return chi2_cmb + chi2_bao(params) + chi2_sn(params)


bounds = np.array(
    [
        (-1.0, 1.0),  # ΔM
        (60.0, 75.0),  # H0
        (0.010, 0.030),  # ωb = Ωb * h^2
        (0.01, 0.25),  # ωc = Ωc * h^2
        (-1.0, -1 / 3),  # w0
    ],
    dtype=np.float64,
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
    return Om / 2 + (1 + 3 * w0) * (1 - Om) / 2


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
    burn_in = 400
    nsteps = 4000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

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
        len(z_sn_vals) + len(bao_data["z"]) + len(cmb.DISTANCE_PRIORS) - len(bounds)
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
    print(f"r*: {cmb.rs_z(Ez, z_st_50, *best_fit[1:]):.2f} Mpc")
    print(f"z_d: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}")
    print(f"r_d: {cmb.rs_z(Ez, z_dr_50, *best_fit[1:]):.2f} Mpc")
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
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1

** Planck + ACT compression **
H0: 68.38 +0.28 -0.27 km/s/Mpc
Ωm: 0.3009 +0.0036 -0.0036
ωb: 0.02257 +0.00010 -0.00010
ωc: 0.1175 +0.0007 -0.0007
ωm: 0.1407 +0.0006 -0.0006
w0: -1
wa: 0
z*: 1089.43 +0.16 -0.16
r*: 144.94 Mpc
z_d: 1060.21 +0.23 -0.23
r_d: 147.54 Mpc
Chi squared: 42.91
Log evidence: -37.3
Degs of freedom: 34

** Early-time ΛCDM **
H0: 68.30 +0.29 -0.29 km/s/Mpc
Ωm: 0.3010 +0.0038 -0.0038
ωb: 0.02236 +0.00012 -0.00012
ωc: 0.1174 +0.0007 -0.0006
ωm: 0.1404 +0.0006 -0.0006
w0: -1
wa: 0
z*: 1089.72 +0.19 -0.19
r*: 145.13 Mpc
z_d: 1059.88 +0.27 -0.27
r_d: 147.80 Mpc
Chi squared: 42.15
Log evidence: -36.9
Degs of freedom: 34
"""


"""
Flat wCDM w(z) = w0

** Planck + ACT compression **
H0: 67.79 +0.70 -0.70 km/s/Mpc
Ωm: 0.3052 +0.0060 -0.0059
ωb: 0.02259 +0.00011 -0.00011
ωc: 0.1170 +0.0009 -0.0009
ωm: 0.1402 +0.0008 -0.0008
w0: -0.974 +0.028 -0.028 (prior width 0.8: -1.3 to -0.5)
wa: 0
z*: 1089.35 +0.18 -0.17
r*: 145.05 Mpc
z_d: 1060.21 +0.23 -0.23
r_d: 147.65 Mpc
q0: -0.516 +0.036 -0.037
Chi squared: 42.02
Log evidence: -39.4 (Δ logZ = -2.1 in favour of ΛCDM)
Degs of freedom: 33

** Early-time ΛCDM **
H0: 67.62 +0.71 -0.70 km/s/Mpc
Ωm: 0.3058 +0.0060 -0.0060
ωb: 0.02241 +0.00013 -0.00013
ωc: 0.1168 +0.0009 -0.0009
ωm: 0.1398 +0.0008 -0.0008
w0: -0.969 +0.029 -0.029 (prior width 0.8: -1.3 to -0.5)
wa: 0
z*: 1089.69 +0.21 -0.21
r*: 145.24 Mpc
z_d: 1059.89 +0.28 -0.28
r_d: 147.91 Mpc
q0: -0.509 +0.037 -0.038
Chi squared: 40.85
Log evidence: -38.6 (Δ logZ = -1.7 in favour of ΛCDM)
Degs of freedom: 33
"""


"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)

** Planck + ACT compression **
H0: 66.60 +0.81 -0.81 km/s/Mpc
Ωm: 0.3157 +0.0077 -0.0074
ωb: 0.02260 +0.00010 -0.00010
ωc: 0.1168 +0.0007 -0.0007
ωm: 0.1400 +0.0007 -0.0007
w0: -0.856 +0.062 -0.062 (prior width 0.8: -1.0 to -1/3; 2.32 sigma to the prior left edge)
wa: -0.401 [derived: wa = -1.5 * (1 - w0^2)]
z*: 1089.31 +0.16 -0.16
r*: 145.10 Mpc
z_d: 1060.22 +0.22 -0.23
r_d: 147.70 Mpc
q0: -0.379 +0.071 -0.073
Chi squared: 37.92
Log evidence: -36.3 (Δ logZ = 1.0 against ΛCDM)
Degs of freedom: 33

** Early-time ΛCDM **
H0: 66.48 +0.82 -0.82 km/s/Mpc
Ωm: 0.3161 +0.0078 -0.0076
ωb: 0.02242 +0.00012 -0.00012
ωc: 0.1166 +0.0007 -0.0007
ωm: 0.1397 +0.0007 -0.0007
w0: -0.850 +0.062 -0.063 (prior width 0.8: -1.0 to -1/3; 2.34 sigma to the prior left edge)
wa: -0.416 [devired: wa = -1.5 * (1 - w0^2)]
z*: 1089.65 +0.19 -0.19
r*: 145.27 Mpc
z_d: 1059.91 +0.27 -0.27
r_d: 147.93 Mpc
q0: -0.372 +0.072 -0.074
Chi squared: 36.73
Log evidence: -35.6 (Δ logZ = 1.3 against ΛCDM)
Degs of freedom: 33
"""


"""
Flat w(z) = w0 + wa * z / (1 + z)

** Planck + ACT compression **
H0: 66.08 +0.84 -0.83 km/s/Mpc
Ωm: 0.3254 +0.0087 -0.0086
ωb: 0.02251 +0.00011 -0.00011
ωc: 0.1189 +0.0010 -0.0010
ωm: 0.1421 +0.0009 -0.0009
w0: -0.690 +0.089 -0.087 (prior width 1.5: -1.5 to 0.0)
wa: -0.962 +0.289 -0.306 (prior width 4.0: -3.0 to 1.0)
z*: 1089.63 +0.19 -0.19
r*: 144.62 Mpc
z_d: 1060.18 +0.23 -0.23
r_d: 147.24 Mpc
Chi squared: 29.60
Log evidence: -35.2 (Δ logZ = 2.1 against ΛCDM)
Degs of freedom: 32

** Early-time ΛCDM **
H0: 65.96 +0.85 -0.83 km/s/Mpc
Ωm: 0.3257 +0.0088 -0.0086
ωb: 0.02225 +0.00013 -0.00013
ωc: 0.1188 +0.0010 -0.0010
ωm: 0.1417 +0.0009 -0.0010
w0: -0.692 +0.091 -0.088 (prior width 1.5: -1.5 to 0.0)
wa: -0.957 +0.297 -0.317 (prior width 4.0: -3.0 to 1.0)
z*: 1089.99 +0.24 -0.24
r*: 144.83 Mpc
z_d: 1059.75 +0.28 -0.28
r_d: 147.53 Mpc
Chi squared: 29.39
Log evidence: -35.0 (Δ logZ = 1.9 against ΛCDM)
Degs of freedom: 32

Flat w(z) = w0 + wa * [(1 + z)^2 - 1] / [(1 + z)^2 + 1]
rho(z) = rho0 * (1 + z)^[3 * (1 + w0 + wa)] * [2 * (1 + z)^2 / (1 + (1 + z)^2)]^(-3 * wa)

** Planck + ACT compression **
H0: 66.11 +0.84 -0.82 km/s/Mpc
Ωm: 0.3251 +0.0087 -0.0085
ωb: 0.02251 +0.00011 -0.00011
ωc: 0.1189 +0.0010 -0.0010
ωm: 0.1421 +0.0009 -0.0009
w0: -0.706 +0.085 -0.083 (prior width 1.25: -1.25 to 0.0)
wa: -0.784 +0.236 -0.250 (prior width 3.5: -2.5 to 1.0)
z*: 1089.63 +0.19 -0.19
r*: 144.61 Mpc
z_d: 1060.18 +0.23 -0.23
r_d: 147.23 Mpc
Chi squared: 29.57
Log evidence: -35.1 (Δ logZ = 2.2 against ΛCDM)
Degs of freedom: 32

** Early-time ΛCDM **
H0: 66.00 +0.83 -0.83 km/s/Mpc
Ωm: 0.3254 +0.0088 -0.0085
ωb: 0.02225 +0.00013 -0.00013
ωc: 0.1189 +0.0010 -0.0010
ωm: 0.1417 +0.0009 -0.0009
w0: -0.709 +0.086 -0.083 (prior width 1.25: -1.25 to 0.0)
wa: -0.779 +0.240 -0.259 (prior width 3.5: -2.5 to 1.0)
z*: 1089.99 +0.24 -0.24
r*: 144.82 Mpc
z_d: 1059.75 +0.28 -0.28
r_d: 147.52 Mpc
Chi squared: 29.36
Log evidence: -34.8 (Δ logZ = 2.1 against ΛCDM)
Degs of freedom: 32
"""
