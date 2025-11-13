from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import y2023cmbearlylcdm.data as cmb
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2(2.044)
Omnu_h2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
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
    dL = (1 + z_hel) * DM_z(z_cmb, params)
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
        (-0.4, 0.4),  # ΔM
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
    from gelman_rubin import gelman_rubin
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
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
    print(f"Degrees of freedom: {1 + len(bao_data['z']) + sn_size - len(best_fit)}")

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

-- Early time ΛCDM arXiv:2302.12911 --
H0: 68.54 +0.41 -0.41 km/s/Mpc
ωb: 0.02265 +0.00028 -0.00029
ωc: 0.1172 +0.0007 -0.0007
ωm: 0.1405 +0.0006 -0.0006
Ωm: 0.2992 +0.0046 -0.0044
w0: -1
wa: 0
r_d: 147.52 Mpc
100 θ*: 1.04107
Chi squared: 1662.1
Log evidence: -849.1
Degrees of freedom: 1745

-- ATC DR6 only --
H0: 69.36 +0.49 -0.49 km/s/Mpc
ωb: 0.02354 +0.00049 -0.00049
ωc: 0.1169 +0.0007 -0.0007
ωm: 0.1411 +0.0008 -0.0008
Ωm: 0.2933 +0.0046 -0.0045
w0: -1
wa: 0
r_d: 146.21 Mpc
100 θ*: 1.04089
Chi squared: 1666.7
Log evidence: -850.5
Degrees of freedom: 1745
"""


"""
Flat wCDM w(z) = w0

-- Early time ΛCDM arXiv:2302.12911 --
H0: 67.19 +0.55 -0.54 km/s/Mpc
ωb: 0.02355 +0.00040 -0.00039
ωc: 0.1138 +0.0012 -0.0013
ωm: 0.1380 +0.0010 -0.0010
Ωm: 0.3057 +0.0049 -0.0049
w0: -0.901 +0.027 -0.027 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 147.43 Mpc
100 θ*: 1.04100
Chi squared: 1649.3
Log evidence: -845.8 (Δ logZ = 3.3 against ΛCDM)
Degrees of freedom: 1744

-- ATC DR6 only --
H0: 67.85 +0.58 -0.58 km/s/Mpc
ωb: 0.02490 +0.00060 -0.00061
ωc: 0.1128 +0.0013 -0.0013
ωm: 0.1383 +0.0011 -0.0011
Ωm: 0.3004 +0.0048 -0.0047
w0: -0.882 +0.027 -0.027 (prior width 1.5: -1.5 to 0.0)
wa: 0
r_d: 145.83 Mpc
100 θ*: 1.04071
Chi squared: 1648.3
Log evidence: -844.4 (Δ logZ = 6.1 against ΛCDM)
Degrees of freedom: 1744
"""


"""
Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)

-- Early time ΛCDM arXiv:2302.12911 --
H0: 66.89 +0.57 -0.56 km/s/Mpc
ωb: 0.02317 +0.00032 -0.00031
ωc: 0.1154 +0.0008 -0.0008
ωm: 0.1392 +0.0007 -0.0007
Ωm: 0.3111 +0.0055 -0.0055
w0: -0.829 +0.042 -0.042 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
r_d: 147.44 Mpc
100 θ*: 1.04102
Chi squared: 1646.5
Log evidence: -844.0 (Δ logZ = 5.1 against ΛCDM)
Degrees of freedom: 1744

-- ATC DR6 only --
H0: 67.56 +0.61 -0.61 km/s/Mpc
ωb: 0.02433 +0.00052 -0.00052
ωc: 0.1148 +0.0009 -0.0009
ωm: 0.1398 +0.0009 -0.0009
Ωm: 0.3063 +0.0056 -0.0054
w0: -0.807 +0.042 -0.041 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
r_d: 145.89 Mpc
r_d: 145.90 Mpc
100 θ*: 1.04075
Chi squared: 1646.3
Log evidence: -843.0 (Δ logZ = 7.5 against ΛCDM)
Degrees of freedom: 1744
"""


"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)

-- Early time ΛCDM arXiv:2302.12911 --
TODO
w0: (prior width 1.5: -1.5 to 0.0)
wa: (prior width 4.0: -2.5 to 1.5)

-- ATC DR6 only --
TODO
"""
