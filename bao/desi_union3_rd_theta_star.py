from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
import y2025cmb_p_actbase_lcdm_camb.data as cmb
from y2023union3.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data

c = c0 / 1000  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2(2.044)
Omnu_h2 = cmb.Omnu_h2

sn_legend, z_cmb, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, Obc, Or, w0=-1, wa=0):
    Ol = 1 - Obc - Or
    inv_a = 1 + z
    cubic = inv_a**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Or * inv_a**4 + Obc * cubic + Ol * rho_de)


@njit
def theory_mu(params):
    dL = (1 + z_cmb) * DM_z(z_cmb, params)
    return params[0] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0 = params[1:]
    h = H0 / 100
    Obc = (Obh2 + Och2 + Omnu_h2) / h**2
    Or = Orh2 / h**2
    return H0 * Ez(z, Obc, Or, w0)


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


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    dists = cmb.cmb_distances(Ez, *params[1:])
    rd = dists[1]

    delta_cmb = cmb.DISTANCE_PRIORS - dists
    chi2_rd = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, rd, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi_sn + chi_bao + chi2_rd


bounds = np.array(
    [
        (-0.8, 0.8),  # ΔM
        (50.0, 90.0),  # H0
        (0.010, 0.030),  # Ob * h^2
        (0.05, 0.3),  # Ωm * h^2
        (-1.5, 0.0),  # w0
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
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence
    from sn.plotting import plot_predictions as plot_sn_predictions
    from gelman_rubin import gelman_rubin
    from .plot_predictions import plot_bao_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 300
    nsteps = 2000 + burn_in
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
        print("effective samples", ndim * nwalkers * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)
    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    [
        (dM_16, dM_50, dM_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)
    theta_100, rd = cmb.cmb_distances(Ez, *best_fit[1:])

    omh2_samples = samples[:, 2] + samples[:, 3] + Omnu_h2
    Om_samples = omh2_samples / (samples[:, 1] / 100) ** 2
    omh2_16, omh2_50, omh2_84 = np.percentile(omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {omh2_50:.4f} +{(omh2_84 - omh2_50):.4f} -{(omh2_50 - omh2_16):.4f}")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {rd:.2f} Mpc")
    print(f"100 θ*: {theta_100:.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {1 + len(bao_data['z']) + len(z_cmb) - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, rd, best_fit),
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
(100 θ*, r_drag)CMB
"""


"""
Flat ΛCDM w(z) = -1

-- Early ΛCDM arXiv:2302.12911 --
H0: 68.78 +0.42 -0.42 km/s/Mpc
ωb: 0.02277 +0.00029 -0.00029
ωc: 0.1168 +0.0007 -0.0007
ωm: 0.1402 +0.0006 -0.0006
Ωm: 0.2965 +0.0047 -0.0045
w0: -1
wa: 0
r_d: 147.49 Mpc
100 θ*: 1.04106
Chi squared: 39.9
Log evidence: -35.6
Degrees of freedom: 32

-- ATC DR6 --
H0: 69.59 +0.52 -0.52 km/s/Mpc
ωb: 0.02399 +0.00051 -0.00051
ωc: 0.1173 +0.0008 -0.0008
ωm: 0.1419 +0.0009 -0.0008
Ωm: 0.2930 +0.0048 -0.0047
w0: -1
wa: 0
r_d: 146.05 Mpc
100 θ*: 1.04082
Chi squared: 41.3
Log evidence: -35.4
Degrees of freedom: 32

-- ATC DR6 + Planck --
H0: 68.95 +0.44 -0.44 km/s/Mpc
ωb: 0.02304 +0.00030 -0.00030
ωc: 0.1169 +0.0007 -0.0007
ωm: 0.1406 +0.0007 -0.0006
Ωm: 0.2958 +0.0047 -0.0046
w0: -1
wa: 0
r_d: 147.18 Mpc
100 θ*: 1.04097
Chi squared: 40.2
Log evidence: -35.6
Degrees of freedom: 32
"""


"""
Flat wCDM w(z) = w0

-- Early ΛCDM arXiv:2302.12911 --
H0: 67.28 +0.71 -0.68 km/s/Mpc
ωb: 0.02351 +0.00043 -0.00042
ωc: 0.1140 +0.0014 -0.0014
ωm: 0.1381 +0.0011 -0.0011
Ωm: 0.3051 +0.0057 -0.0056
w0: -0.906 +0.035 -0.036 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 147.43 Mpc
100 θ*: 1.04101
Chi squared: 33.3
Log evidence: -35.2 (Δ logZ = 0.4 against ΛCDM)
Degrees of freedom: 31

-- ATC DR6 --
H0: 67.93 +0.75 -0.73 km/s/Mpc
ωb: 0.02502 +0.00065 -0.00063
ωc: 0.1140 +0.0014 -0.0015
ωm: 0.1396 +0.0012 -0.0012
Ωm: 0.3025 +0.0057 -0.0057
w0: -0.892 +0.036 -0.036 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 145.81 Mpc
100 θ*: 1.04069
Chi squared: 32.7
Log evidence: -34.0 (Δ logZ = 1.4 against ΛCDM)
Degrees of freedom: 31

-- ATC DR6 + Planck --
H0: 67.41 +0.72 -0.71 km/s/Mpc
ωb: 0.02380 +0.00045 -0.00044
ωc: 0.1140 +0.0014 -0.0015
ωm: 0.1384 +0.0011 -0.0011
Ωm: 0.3046 +0.0058 -0.0057
w0: -0.904 +0.036 -0.037 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 147.12 Mpc
100 θ*: 1.04091
Chi squared: 33.2
Log evidence: -35.0 (Δ logZ = 0.6 against ΛCDM)
Degrees of freedom: 31
"""


"""
Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)

-- Early ΛCDM arXiv:2302.12911 --
ΔM: -0.168 +0.089 -0.088 mag
H0: 66.52 +0.84 -0.80 km/s/Mpc
ωb: 0.02321 +0.00033 -0.00033
ωc: 0.1152 +0.0009 -0.0009
ωm: 0.1390 +0.0007 -0.0007
Ωm: 0.3142 +0.0075 -0.0076
w0: -0.800 +0.064 -0.065 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/d z at z=0 = (9/4) * (1 + w0)
r_d: 147.44 Mpc
100 θ*: 1.04100
Chi squared: 30.5
Log evidence: -33.2 (Δ logZ = 2.4 against ΛCDM)
Degrees of freedom: 31

-- ATC DR6 --
H0: 67.17 +0.87 -0.84 km/s/Mpc
ωb: 0.02461 +0.00054 -0.00054
ωc: 0.1155 +0.0009 -0.0009
ωm: 0.1408 +0.0009 -0.0009
Ωm: 0.3120 +0.0076 -0.0076
w0: -0.783 +0.065 -0.065 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/d z at z=0 = (9/4) * (1 + w0)
r_d: 145.84 Mpc
100 θ*: 1.04071
Chi squared: 30.1
Log evidence: -32.1 (Δ logZ = 3.3 against ΛCDM)
Degrees of freedom: 31

-- ATC DR6 + Planck --
H0: 66.66 +0.85 -0.82 km/s/Mpc
ωb: 0.02350 +0.00034 -0.00034
ωc: 0.1152 +0.0009 -0.0009
ωm: 0.1394 +0.0008 -0.0008
Ωm: 0.3137 +0.0077 -0.0077
w0: -0.796 +0.064 -0.066 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/d z at z=0 = (9/4) * (1 + w0)
r_d: 147.12 Mpc
100 θ*: 1.04095
Chi squared: 30.4
Log evidence: -33.1 (Δ logZ = 2.5 against ΛCDM)
Degrees of freedom: 31
"""


"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)

-- Early ΛCDM arXiv:2302.12911 --
H0: 66.12 +0.89 -0.87 km/s/Mpc
ωb: 0.02249 +0.00053 -0.00049
ωc: 0.1181 +0.0017 -0.0020
ωm: 0.1412 +0.0013 -0.0015
Ωm: 0.3230 +0.0102 -0.0099
w0: -0.717 +0.099 -0.096 (prior width 1.5: -1.5 to 0.0)
wa: -0.823 +0.385 -0.399 (prior width 4.5: -3.0 to 1.5)
r_d: 147.46 Mpc
100 θ*: 1.04097
Chi squared: 29.2
Log evidence: -34.3 (Δ logZ = 1.3 against ΛCDM)
Degrees of freedom: 30

-- ATC DR6 --
H0: 66.94 +0.90 -0.91 km/s/Mpc
ωb: 0.02396 +0.00078 -0.00074
ωc: 0.1163 +0.0019 -0.0023
ωm: 0.1408 +0.0015 -0.0018
Ωm: 0.3142 +0.0102 -0.0102
w0: -0.747 +0.095 -0.092 (prior width 1.5: -1.5 to 0.0)
wa: -0.584 +0.366 -0.385 (prior width 4.5: -3.0 to 1.5)
r_d: 145.92 Mpc
100 θ*: 1.04079
Chi squared: 30.1
Log evidence: -34.0 (Δ logZ = 2.2 against ΛCDM)
Degrees of freedom: 30

-- ATC DR6 + Planck --
H0: 66.28 +0.87 -0.88 km/s/Mpc
ωb: 0.02281 +0.00054 -0.00050
ωc: 0.1180 +0.0017 -0.0019
ωm: 0.1414 +0.0013 -0.0015
Ωm: 0.3218 +0.0101 -0.0098
w0: -0.722 +0.099 -0.094 (prior width 1.5: -1.5 to 0.0)
wa: -0.787 +0.376 -0.398 (prior width 4.5: -3.0 to 1.5)
r_d: 147.15 Mpc
100 θ*: 1.04090
Chi squared: 29.2
Log evidence: -34.3 (Δ logZ = 1.3 against ΛCDM)
Degrees of freedom: 30
"""
