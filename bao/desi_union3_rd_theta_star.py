from numba import njit
import numpy as np
import y2023cmbearlylcdm.data as cmb
from y2023union3.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(cov_matrix_bao)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0, wa):
    a3 = 1 / (1 + z) ** 3
    return 4 / ((1 + w0) * a3 + (1 - w0)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1 + z

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
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
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
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, rd, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


@njit
def theory_mu(params):
    dL = (1 + z_cmb) * DM_z(z_cmb, params)
    return params[0] + 25 + 5 * np.log10(dL)


def chi_squared(params):
    dists = cmb.cmb_distances(Ez, *params[1:])
    rd = dists[1]

    delta_cmb = cmb.DISTANCE_PRIORS - dists
    chi2_rd = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_sn = mu_values - theory_mu(params)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, rd, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    return chi_sn + chi_bao + chi2_rd


bounds = np.array(
    [
        (-1.0, 1.0),  # ΔM
        (50.0, 90.0),  # H0
        (0.010, 0.030),  # Ob * h^2
        (0.05, 0.30),  # Ωc * h^2
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


def main():
    import emcee
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence
    from gelman_rubin import gelman_rubin

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.70),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)
    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    one_sigma_ci = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_ci, axis=0).T
    [
        (dM_16, dM_50, dM_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)
    theta_100 = cmb.cmb_distances(Ez, *best_fit[1:])[0]

    omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = omh2_samples / (samples[:, 1] / 100) ** 2
    rd_samples = cmb.r_drag(samples[:, 2], omh2_samples)
    zd_samples = cmb.z_drag(samples[:, 2], omh2_samples)
    zst_samples = cmb.z_star(samples[:, 2], omh2_samples)
    omh2_16, omh2_50, omh2_84 = np.percentile(omh2_samples, one_sigma_ci)
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, one_sigma_ci)
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, one_sigma_ci)
    zd_16, zd_50, zd_84 = np.percentile(zd_samples, one_sigma_ci)
    zst_16, zst_50, zst_84 = np.percentile(zst_samples, one_sigma_ci)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {omh2_50:.4f} +{(omh2_84 - omh2_50):.4f} -{(omh2_50 - omh2_16):.4f}")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"z_d: {zd_50:.2f} +{(zd_84 - zd_50):.2f} -{(zd_50 - zd_16):.2f}")
    print(f"z*: {zst_50:.2f} +{(zst_84 - zst_50):.2f} -{(zst_50 - zst_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, zst_50, *best_fit[1:]):.2f} Mpc")
    print(f"100 θ*: {theta_100:.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {2 + len(bao_data['z']) + len(z_cmb) - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, rd_50, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$Δ_M$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
(100 θ*, r_drag)CMB + DESI DR2 BAO + Union3 SnIa
"""

"""
Flat ΛCDM w(z) = -1
-- Early ΛCDM arXiv:2302.12911 --
H0: 68.77 +0.44 -0.43 km/s/Mpc
ωb: 0.02276 +0.00030 -0.00029
ωc: 0.1169 +0.0007 -0.0007
ωm: 0.1403 +0.0006 -0.0006
Ωm: 0.2966 +0.0047 -0.0047
w0: -1
wa: 0
r_d: 147.49 +0.28 -0.28 Mpc
z_d: 1060.71 +0.65 -0.65
z*: 1089.25 +0.40 -0.39
r*: 144.94 Mpc
100 θ*: 1.04105
Chi squared: 39.9
Log evidence: -35.8
Degrees of freedom: 33

-- ATC DR6 --
H0: 69.59 +0.52 -0.52 km/s/Mpc
ωb: 0.02399 +0.00051 -0.00051
ωc: 0.1173 +0.0008 -0.0008
ωm: 0.1419 +0.0009 -0.0008
Ωm: 0.2930 +0.0048 -0.0047
w0: -1
wa: 0
r_d: 146.05 +0.56 -0.55 Mpc
z_d: 1063.31 +1.08 -1.10
z*: 1087.66 +0.62 -0.60
r*: 143.92 Mpc
100 θ*: 1.04082
Chi squared: 41.4
Log evidence: -35.7
Degrees of freedom: 33

-- ATC DR6 + Planck --
H0: 68.95 +0.44 -0.43 km/s/Mpc
ωb: 0.02304 +0.00031 -0.00030
ωc: 0.1169 +0.0007 -0.0007
ωm: 0.1406 +0.0007 -0.0006
Ωm: 0.2957 +0.0047 -0.0047
w0: -1
wa: 0
r_d: 147.18 +0.29 -0.29 Mpc
z_d: 1061.21 +0.65 -0.65
z*: 1088.77 +0.41 -0.42
r*: 144.74 Mpc
100 θ*: 1.04096
Chi squared: 40.2
Log evidence: -35.9
Degrees of freedom: 33
"""

"""
Flat wCDM w(z) = w0

-- Early ΛCDM arXiv:2302.12911 --
H0: 67.28 +0.71 -0.70 km/s/Mpc
ωb: 0.02351 +0.00045 -0.00043
ωc: 0.1140 +0.0014 -0.0015
ωm: 0.1381 +0.0011 -0.0011
Ωm: 0.3051 +0.0058 -0.0057
w0: -0.906 +0.036 -0.037 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 147.44 +0.28 -0.28 Mpc
z_d: 1062.26 +0.91 -0.89
z*: 1087.96 +0.64 -0.64
r*: 145.14 Mpc
100 θ*: 1.04100
Chi squared: 33.4
Log evidence: -35.4 (Δ logZ = 0.4 against ΛCDM)
Degrees of freedom: 32

-- ATC DR6 --
H0: 67.93 +0.75 -0.74 km/s/Mpc
ωb: 0.02501 +0.00065 -0.00063
ωc: 0.1140 +0.0014 -0.0015
ωm: 0.1396 +0.0012 -0.0012
Ωm: 0.3025 +0.0058 -0.0057
w0: -0.893 +0.036 -0.036 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 145.82 +0.55 -0.56 Mpc
z_d: 1065.38 +1.33 -1.31
z*: 1086.20 +0.79 -0.78
r*: 144.00 Mpc
100 θ*: 1.04073
Chi squared: 32.7
Log evidence: -34.2 (Δ logZ = 1.4 against ΛCDM)
Degrees of freedom: 32

-- ATC DR6 + Planck --
H0: 67.40 +0.72 -0.71 km/s/Mpc
ωb: 0.02381 +0.00045 -0.00044
ωc: 0.1140 +0.0014 -0.0015
ωm: 0.1384 +0.0011 -0.0011
Ωm: 0.3046 +0.0058 -0.0057
w0: -0.903 +0.036 -0.037 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 147.13 +0.29 -0.29 Mpc
z_d: 1062.75 +0.92 -0.91
z*: 1087.57 +0.64 -0.64
r*: 144.92 Mpc
100 θ*: 1.04092
Chi squared: 33.2
Log evidence: -35.2 (Δ logZ = 0.7 against ΛCDM)
Degrees of freedom: 32
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)

-- Early ΛCDM arXiv:2302.12911 --
H0: 66.50 +0.84 -0.82 km/s/Mpc
ωb: 0.02318 +0.00033 -0.00033
ωc: 0.1153 +0.0009 -0.0009
ωm: 0.1391 +0.0007 -0.0007
Ωm: 0.3146 +0.0078 -0.0077
w0: -0.788 +0.065 -0.067 (prior width -2/3: -1.0 to -1/3)
wa: d w(z)/d z at z=0 = -1.5 * (1 - w0**2)
r_d: 147.44 +0.28 -0.28 Mpc
z_d: 1061.55 +0.70 -0.71
z*: 1088.62 +0.44 -0.44
r*: 145.03 Mpc
100 θ*: 1.04100
Chi squared: 30.4
Log evidence: -32.5 (Δ logZ = 3.3 against ΛCDM)
Degrees of freedom: 32

-- ATC DR6 --
H0: 67.14 +0.87 -0.85 km/s/Mpc
ωb: 0.02459 +0.00055 -0.00054
ωc: 0.1155 +0.0009 -0.0009
ωm: 0.1408 +0.0009 -0.0009
Ωm: 0.3123 +0.0077 -0.0076
w0: -0.769 +0.065 -0.067 (prior width -2/3: -1.0 to -1/3)
wa: d w(z)/d z at z=0 = -1.5 * (1 - w0**2)
r_d: 145.85 +0.56 -0.56 Mpc
z_d: 1064.48 +1.14 -1.14
z*: 1086.82 +0.65 -0.63
r*: 143.91 Mpc
100 θ*: 1.04075
Chi squared: 30.0
Log evidence: -31.5 (Δ logZ = 4.2 against ΛCDM)
Degrees of freedom: 32

-- ATC DR6 + Planck --
H0: 66.63 +0.84 -0.83 km/s/Mpc
ωb: 0.02348 +0.00034 -0.00033
ωc: 0.1153 +0.0009 -0.0009
ωm: 0.1394 +0.0008 -0.0008
Ωm: 0.3140 +0.0077 -0.0076
w0: -0.784 +0.065 -0.067 (prior width -2/3: -1.0 to -1/3)
wa: d w(z)/d z at z=0 = -1.5 * (1 - w0**2)
r_d: 147.13 +0.29 -0.29 Mpc
z_d: 1062.05 +0.70 -0.70
z*: 1088.09 +0.46 -0.46
r*: 144.83 Mpc
100 θ*: 1.04093
Chi squared: 30.3
Log evidence: -32.4 (Δ logZ = 3.5 against ΛCDM)
Degrees of freedom: 32
"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)

-- Early ΛCDM arXiv:2302.12911 --
H0: 66.11 +0.92 -0.89 km/s/Mpc
ωb: 0.02250 +0.00057 -0.00050
ωc: 0.1181 +0.0017 -0.0021
ωm: 0.1412 +0.0013 -0.0016
Ωm: 0.3228 +0.0104 -0.0105
w0: -0.719 +0.102 -0.100 (prior width 1.5: -1.5 to 0.0)
wa: -0.807 +0.403 -0.417 (prior width 4.5: -3.0 to 1.5)
r_d: 147.47 +0.28 -0.27 Mpc
z_d: 1060.27 +1.17 -1.05
z*: 1089.59 +0.81 -0.91
r*: 144.84 Mpc
100 θ*: 1.04100
Chi squared: 29.1
Log evidence: -34.6 (Δ logZ = 1.2 against ΛCDM)
Degrees of freedom: 31

-- ATC DR6 --
H0: 66.88 +0.95 -0.93 km/s/Mpc
ωb: 0.02401 +0.00079 -0.00073
ωc: 0.1177 +0.0019 -0.0023
ωm: 0.1423 +0.0015 -0.0017
Ωm: 0.3180 +0.0105 -0.0105
w0: -0.735 +0.101 -0.097 (prior width 1.5: -1.5 to 0.0)
wa: -0.681 +0.391 -0.406 (prior width 4.5: -3.0 to 1.5)
r_d: 145.93 +0.57 -0.56 Mpc
z_d: 1063.47 +1.61 -1.54
z*: 1087.66 +1.01 -1.06
r*: 143.79 Mpc
100 θ*: 1.04071
Chi squared: 29.6
Log evidence: -34.0 (Δ logZ = 1.6 against ΛCDM)
Degrees of freedom: 31

-- ATC DR6 + Planck --
H0: 66.27 +0.90 -0.89 km/s/Mpc
ωb: 0.02281 +0.00059 -0.00050
ωc: 0.1180 +0.0017 -0.0021
ωm: 0.1414 +0.0014 -0.0016
Ωm: 0.3218 +0.0104 -0.0102
w0: -0.721 +0.102 -0.100 (prior width 1.5: -1.5 to 0.0)
wa: -0.787 +0.401 -0.418 (prior width 4.5: -3.0 to 1.5)
r_d: 147.16 +0.29 -0.30 Mpc
z_d: 1060.79 +1.20 -1.05
z*: 1089.16 +0.80 -0.91
r*: 144.63 Mpc
100 θ*: 1.04095
Chi squared: 29.2
Log evidence: -34.5 (Δ logZ = 1.4 against ΛCDM)
Degrees of freedom: 31
"""
