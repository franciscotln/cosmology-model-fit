from numba import njit
import numpy as np
from interpolator import interp_pchip
from y2025BAO.data import get_data as get_bao_data
import y2025cmb_p_actbase_lcdm_camb.data as cmb

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

z_grid = np.linspace(0, np.max(bao_data["z"]) + 0.1, num=3000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0, wa):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1 + w0 + (1 - w0) * zp1**3)) ** 2  # wzCDM
    # return zp1 ** (3 * (1.0 + w0))  # wCDM


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

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0 = params
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
    return interp_pchip(z, z_grid, cum_dm)


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


def chi_squared(params):
    distances = cmb.cmb_distances(H_z, params[1], params[2], params)
    rd = distances[1]
    delta_cmb = cmb.DISTANCE_PRIORS - distances
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, rd, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    return chi_bao + chi2_cmb


bounds = np.array(
    [
        (60, 75),  # H0
        (0.010, 0.030),  # Obh2
        (0.01, 0.3),  # Och2
        (-1.0, -0.2),  # w0
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


def main():
    import emcee
    from multiprocessing import Pool
    from bao.plot_predictions import plot_bao_predictions
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
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]
    with Pool(5) as pool:
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

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    best_fit = np.percentile(samples, 50, axis=0)
    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0)
    H0_err, obh2_err, och2_err, w0_err = np.diff(pct, axis=0).T

    Om_h2_samples = Omnu_h2 + samples[:, 1] + samples[:, 2]
    Om_samples = Om_h2_samples / (samples[:, 0] / 100) ** 2
    Om_h2_50 = np.percentile(Om_h2_samples, 50)
    Om_h2_err = np.diff(np.percentile(Om_h2_samples, [15.9, 50, 84.1]))
    Om_50 = np.percentile(Om_samples, 50)
    Om_err = np.diff(np.percentile(Om_samples, [15.9, 50, 84.1]))

    thetastar_best, rd_best = cmb.cmb_distances(H_z, best_fit[1], best_fit[2], best_fit)

    print("Gelman-Rubin:", gelman_rubin(chains_samples))
    print(f"H0: {best_fit[0]:.2f} +{H0_err[0]:.2f} -{H0_err[0]:.2f} km/s/Mpc")
    print(f"ωb: {best_fit[1]:.5f} +{obh2_err[1]:.5f} -{obh2_err[0]:.5f}")
    print(f"ωc: {best_fit[2]:.5f} +{och2_err[1]:.5f} -{och2_err[0]:.5f}")
    print(f"w0: {best_fit[3]:.3f} +{w0_err[1]:.3f} -{w0_err[0]:.3f}")
    print(f"ωm: {Om_h2_50:.4f} +{Om_h2_err[1]:.4f} -{Om_h2_err[0]:.4f}")
    print(f"Ωm: {Om_50:.4f} +{Om_err[1]:.4f} -{Om_err[0]:.4f}")
    print(f"rdrag: {rd_best:.2f} Mpc")
    print(f"100 θ*: {thetastar_best:.5f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {1 + len(bao_data['z']) - len(best_fit)}")

    labels = ["$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, rd_best, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )


if __name__ == "__main__":
    main()

"""
*******************************
DESI BAO DR2
(100 θ*, rdrag)CMB from ACT DR6 + Planck
*******************************
"""

"""
Flat ΛCDM w(z) = -1
H0: 69.15 +0.44 -0.44 km/s/Mpc
ωb: 0.02314 +0.00031 -0.00031
ωc: 0.11658 +0.00075 -0.00075
ωm: 0.1404 +0.0007 -0.0006
Ωm: 0.2936 +0.0048 -0.0047
w0: -1
wa: 0
rdrag: 147.16 Mpc
100 θ*: 1.04095
Chi squared: 10.54
Log Evidence: -18.00
Degs of freedom: 11

===============================

Flat wCDM w(z) = w0
H0: 68.18 +0.97 -0.97 km/s/Mpc
ωb: 0.02351 +0.00050 -0.00047
ωc: 0.11511 +0.00159 -0.00170
w0: -0.945 +0.050 -0.052 (prior width 1.0: -1.5 to -0.5)
wa: 0
ωm: 0.1393 +0.0012 -0.0013
Ωm: 0.2995 +0.0073 -0.0074
rdrag: 147.13 Mpc
100 θ*: 1.04092
Chi squared: 9.30
Log Evidence: -19.46
Degs of freedom: 10

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
H0: 66.92 +1.39 -1.39 km/s/Mpc
ωb: 0.02344 +0.00036 -0.00035
ωc: 0.11543 +0.00098 -0.00106
w0: -0.809 +0.112 -0.108 (prior width 0.8: -1.0 to -0.2)
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
ωm: 0.1395 +0.0008 -0.0009
Ωm: 0.3116 +0.0125 -0.0112
rdrag: 147.13 Mpc
100 θ*: 1.04087
Chi squared: 8.54
Log Evidence: -17.99
Degs of freedom: 10

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
"""
