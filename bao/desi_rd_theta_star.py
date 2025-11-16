from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from y2025BAO.data import get_data as get_bao_data
import y2025cmb_actbase_lcdm_camb.data as cmb

c = c0 / 1000  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2(2.044)
Omnu_h2 = cmb.Omnu_h2

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

z_grid = np.linspace(0, np.max(bao_data["z"]) + 0.1, num=2000)
dx = np.diff(z_grid)


@njit
def Ez(z, Obc, Or, w0=-1, wa=0):
    Ol = 1 - Obc - Or
    inv_a = 1 + z
    cubic = inv_a**3
    rho_de = (4 * cubic / (1 + 3 * cubic)) ** (4 * (1 + w0))
    return np.sqrt(Or * inv_a**4 + Obc * cubic + Ol * rho_de)


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0 = params
    h = H0 / 100
    Obc = (Obh2 + Och2 + Omnu_h2) / h**2
    Or = Orh2 / h**2
    return H0 * Ez(z, Obc, Or, w0)


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
    distances = cmb.cmb_distances(Ez, *params)
    rd = distances[1]
    delta_cmb = cmb.DISTANCE_PRIORS - distances
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, rd, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi_bao + chi2_cmb


bounds = np.array(
    [
        (60, 75),  # H0
        (0.010, 0.030),  # Obh2
        (0.01, 0.3),  # Och2
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
    from .plot_predictions import plot_bao_predictions
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 300
    nsteps = 3000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.70),
    ]
    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    best_fit = np.percentile(samples, 50, axis=0)
    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0)
    H0_err, obh2_err, och2_err, w0_err = np.diff(pct, axis=0).T

    Om_h2_samples = np.full_like(samples[:, 0], Omnu_h2) + samples[:, 1] + samples[:, 2]
    Om_samples = Om_h2_samples / (samples[:, 0] / 100) ** 2
    Om_h2_50 = np.percentile(Om_h2_samples, 50)
    Om_h2_err = np.diff(np.percentile(Om_h2_samples, [15.9, 50, 84.1]))
    Om_50 = np.percentile(Om_samples, 50)
    Om_err = np.diff(np.percentile(Om_samples, [15.9, 50, 84.1]))

    thetastar_best, rd_best = cmb.cmb_distances(Ez, *best_fit)

    print("Gelman-Rubin:", gelman_rubin(chains_samples))
    print(f"H0: {best_fit[0]:.2f} +{H0_err[0]:.2f} -{H0_err[0]:.2f} km/s/Mpc")
    print(f"ωb: {best_fit[1]:.5f} +{obh2_err[1]:.5f} -{obh2_err[0]:.5f}")
    print(f"ωc: {best_fit[2]:.5f} +{och2_err[1]:.5f} -{och2_err[0]:.5f}")
    print(f"w0: {best_fit[3]:.3f} +{w0_err[1]:.3f} -{w0_err[0]:.3f}")
    print(f"ωm: {Om_h2_50:.5f} +{Om_h2_err[1]:.5f} -{Om_h2_err[0]:.5f}")
    print(f"Ωm: {Om_50:.4f} +{Om_err[1]:.4f} -{Om_err[0]:.4f}")
    print(f"rdrag: {rd_best:.2f} Mpc")
    print(f"100 θ*: {thetastar_best:.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {1 + len(bao_data['z']) - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, rd_best, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_corner_and_chains(
        labels=["$H_0$", "$ω_b$", "$ω_c$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
*******************************
DESI BAO DR2 2025 + (100 θ*, rdrag)CMB
*******************************

Flat ΛCDM w(z) = -1

-- ACT DR6 + Planck --
H0: 69.15 +0.44 -0.44 km/s/Mpc
ωb: 0.02314 +0.00031 -0.00031
ωc: 0.11659 +0.00076 -0.00076
ωm: 0.14037 +0.00065 -0.00066
Ωm: 0.2936 +0.0048 -0.0047
w0: -1
wa: 0
rdrag: 147.16 Mpc
100 θ*: 1.04095
Chi squared: 10.54
Log Evidence: -18.00
Degs of freedom: 11

-- ACT DR6 --
H0: 69.82 +0.52 -0.52 km/s/Mpc
ωb: 0.02414 +0.00051 -0.00051
ωc: 0.11696 +0.00077 -0.00077
ωm: 0.14173 +0.00085 -0.00084
Ωm: 0.2907 +0.0049 -0.0048
w0: -1
wa: 0
rdrag: 145.97 Mpc
100 θ*: 1.04080
Chi squared: 11.15
Log Evidence: -17.50
Degs of freedom: 11

===============================

Flat wCDM w(z) = w0

-- ACT DR6 + Planck --
H0: 68.18 +0.97 -0.97 km/s/Mpc
ωb: 0.02351 +0.00050 -0.00047
ωc: 0.11513 +0.00159 -0.00169
ωm: 0.13928 +0.00123 -0.00130
Ωm: 0.2995 +0.0072 -0.0074
w0: -0.945 +0.050 -0.052 (prior width 1.5: -1.5 to 0.0)
wa: 0
rdrag: 147.13 Mpc
100 θ*: 1.04092
Chi squared: 9.30
Log Evidence: -19.87
Degs of freedom: 10

-- ACT DR6 --
H0: 68.59 +0.98 -0.98 km/s/Mpc
ωb: 0.02473 +0.00070 -0.00067
ωc: 0.11500 +0.00164 -0.00173
ωm: 0.14036 +0.00134 -0.00138
Ωm: 0.2983 +0.0072 -0.0072
w0: -0.928 +0.050 -0.052 (prior width 1.5: -1.5 to 0.0)
wa: 0
rdrag: 145.85 Mpc
100 θ*: 1.04074
Chi squared: 9.10
Log Evidence: -18.99
Degs of freedom: 10

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)

-- ATC DR6 + Planck --
H0: 67.11 +1.48 -1.48 km/s/Mpc
ωb: 0.02342 +0.00038 -0.00038
ωc: 0.11551 +0.00112 -0.00112
ωm: 0.13957 +0.00089 -0.00089
Ωm: 0.3099 +0.0132 -0.0131
w0: -0.834 +0.120 -0.124 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
rdrag: 147.13 Mpc
100 θ*: 1.04093
Chi squared: 8.54
Log Evidence: -18.67
Degs of freedom: 10

-- ATC DR6 --
H0: 67.42 +1.47 -1.47 km/s/Mpc
ωb: 0.02456 +0.00058 -0.00058
ωc: 0.11566 +0.00112 -0.00113
ωm: 0.14087 +0.00101 -0.00101
Ωm: 0.3098 +0.0130 -0.0130
w0: -0.803 +0.117 -0.122 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
rdrag: 145.86 Mpc
100 θ*: 1.04076
Chi squared: 8.34
Log Evidence: -17.78
Degs of freedom: 10

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)

-- ACT DR6 + Planck --
H0: 64.36 +2.40 -2.40 km/s/Mpc
ωb: 0.02256 +0.00065 -0.00053
ωc: 0.11903 +0.00187 -0.00244
w0: -0.514 +0.298 -0.280
wa: -1.406 +0.907 -0.972
Ωm: 0.3432 +0.0299 -0.0290
rdrag: 147.15 Mpc
100 θ*: 1.04088
Chi squared: 7.24
Log Evidence: -21.58
Degs of freedom: 9

-- ACT DR6 --
TODO
"""
