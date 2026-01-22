from numba import njit
import numpy as np
from interpolator import interp_pchip
from y2025BAO.data import get_data as get_bao_data
import cmb.data_early_lcdm_compression as cmb

c = cmb.c  # speed of light in km/s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

z_grid = np.linspace(0, np.max(bao_data["z"]) + 0.1, num=4000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0, wa):
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2  # wzCDM
    # return 1  # ΛCDM
    # return (1 + z) ** (3 * (1 + w0))  # wCDM
    # return (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))  # w0waCDM


@njit
def Ez(z, H0, Obh2, Och2, w0=-1.0, wa=0.0):
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
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
def bao_theory(z, qty, params):
    Obh2, Och2 = params[1], params[2]
    rd = cmb.r_drag(Obh2, Obh2 + Och2 + Omnu_h2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(
        H_z, params[1], params[2], params
    )
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    return chi2_cmb + chi_bao


bounds = np.array(
    [
        (50.0, 80.0),  # H0
        (0.020, 0.024),  # ωb = Ωb * h^2
        (0.05, 0.30),  # ωc = Ωc * h^2
        (-1.0, 0.0),  # w0
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


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
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from bao.plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 200
    burn_in = 500
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.30),
        (emcee.moves.DEMove(), 0.70),
    ]

    with Pool(6) as pool:
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
    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    one_sigma_contours = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_contours, axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)
    degs_of_freedom = len(bao_data["z"]) + len(cmb.DISTANCE_PRIORS) - len(best_fit)

    Om_h2_samples = samples[:, 1] + samples[:, 2] + Omnu_h2
    Om_samples = Om_h2_samples / (samples[:, 0] / 100) ** 2
    z_st_samples = cmb.z_star(samples[:, 1], Om_h2_samples)
    r_d_samples = cmb.r_drag(samples[:, 1], Om_h2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Om_h2_samples, one_sigma_contours)
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, one_sigma_contours)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, one_sigma_contours)
    rd_16, rd_50, rd_84 = np.percentile(r_d_samples, one_sigma_contours)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r*: {cmb.rs_z(H_z, z_st_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Log Z: {log_evidence(samples, log_probs, log_probability, bounds):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {degs_of_freedom}")

    labels = ["$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: DESI DR2 2024
CMB Compressed priors: (θ∗, ωb, ωbc)CMB Early Times ΛCDM
*******************************
"""

"""
Flat ΛCDM w(z) = -1
H0: 68.40 +0.30 -0.30 km/s/Mpc
ωb: 0.02237 +0.00012 -0.00012
ωc: 0.1172 +0.0007 -0.0007
ωm: 0.1402 +0.0006 -0.0006
Ωm: 0.300 +0.004 -0.004
w0: -1
wa: 0
r*: 145.16 Mpc
z*: 1089.76 +0.18 -0.18
r_d: 147.83 +0.19 -0.19 Mpc
Log Z: -19.44
Chi squared: 13.49
Degs of freedom: 13
"""

"""
Flat wCDM w(z) = w0
H0: 68.88 +0.97 -0.94 km/s/Mpc
ωb: 0.02235 +0.00013 -0.00013
ωc: 0.1176 +0.0009 -0.0009
ωm: 0.1406 +0.0009 -0.0009
Ωm: 0.296 +0.007 -0.008
w0: -1.021 +0.038 -0.040 (prior width 1.0: -1.5 to -0.5)
wa: 0
r*: 145.08 Mpc
z*: 1089.83 +0.22 -0.21
r_d: 147.77 +0.22 -0.22 Mpc
Log Z: -21.65
Chi squared: 13.28
Degs of freedom: 12
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
H0: 67.25 +0.82 -1.14 km/s/Mpc
ωb: 0.02241 +0.00012 -0.00012
ωc: 0.1168 +0.0007 -0.0007
ωm: 0.1398 +0.0007 -0.0007
Ωm: 0.309 +0.010 -0.007
w0: -0.911 +0.087 -0.062 (prior width 1.0: -1.0 to 0.0; truncated posterior on the left)
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
r*: 145.24 Mpc
z*: 1089.68 +0.18 -0.19
r_d: 147.91 +0.20 -0.20 Mpc
Log Z: -20.59
Chi squared: 13.87
Degs of freedom: 12
"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)

H0: 63.86 +2.10 -2.10 km/s/Mpc
ωb: 0.02222 +0.00014 -0.00014
ωc: 0.1193 +0.0011 -0.0011
ωm: 0.1421 +0.0010 -0.0010
Ωm: 0.348 +0.025 -0.023
w0: -0.468 +0.255 -0.229 (prior width 3.0: -2.0 to 1.0)
wa: -1.565 +0.658 -0.755 (prior width 8.5: -6.0 to 2.5)
r*: 144.74 Mpc
z*: 1090.07 +0.25 -0.25
r_d: 147.46 +0.24 -0.24 Mpc
Log Z: -20.93
Chi squared: 7.04
Degs of freedom: 11
"""


"""
*******************************
Dataset: DESI DR2 2024
CMB Compressed priors: (R, lA = π / θ*, ωb)CMB Planck PR3 with lensing
*******************************
"""

"""
Flat ΛCDM w(z) = -1
H0: 68.42 +0.29 -0.29 km/s/Mpc
ωb: 0.02254 +0.00013 -0.00013
ωc: 0.1177 +0.0006 -0.0006
ωm: 0.1409 +0.0006 -0.0006
Ωm: 0.301 +0.004 -0.004
w0: -1
wa: 0
r*: 144.89 Mpc
z*: 1089.50 +0.18 -0.18
r_d: 147.50 +0.19 -0.19 Mpc
Log Z: -20.39
Chi squared: 15.81
Degs of freedom: 13
"""

"""
Flat wCDM w(z) = w0
H0: 69.25 +0.97 -0.93 km/s/Mpc
ωb: 0.02250 +0.00013 -0.00013
ωc: 0.1183 +0.0009 -0.0009
ωm: 0.1414 +0.0008 -0.0008
Ωm: 0.295 +0.007 -0.007
w0: -1.035 +0.037 -0.039
wa: 0
r*: 144.78 Mpc
z*: 1089.61 +0.21 -0.21
r_d: 147.40 +0.22 -0.22 Mpc
Log Z: -22.34
Chi squared: 15.08
Degs of freedom: 12
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
H0: 67.44 +0.71 -1.05 km/s/Mpc
ωb: 0.02257 +0.00013 -0.00013
ωc: 0.1174 +0.0007 -0.0007
ωm: 0.1406 +0.0007 -0.0007
Ωm: 0.309 +0.009 -0.007
w0: -0.925 +0.081 -0.053 (prior width 1.0: -1.0 to 0.0; truncated posterior on the left)
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
r*: 144.95 Mpc
z*: 1089.44 +0.18 -0.18
r_d: 147.56 +0.20 -0.20 Mpc
Log Z: -21.56
Chi squared: 16.43
Degs of freedom: 12
"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
"""
