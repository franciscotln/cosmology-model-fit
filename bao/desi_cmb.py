from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2025BAO.data import get_data as get_bao_data
import cmb.data_desi_compression as cmb

c = cmb.c  # speed of light in km/s
Orh2 = cmb.Omega_r_h2(2.044)  # 2 relativistic species
Omnu_h2 = cmb.Omnu_h2  # 1 massive species with m_nu = 0.06 eV

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]
cmb_cho = cho_factor(cmb.covariance, lower=True)[0]

z_grid = np.linspace(0, np.max(bao_data["z"]) + 0.1, num=1000)
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


def bao_theory(z, qty, params):
    Obh2, Och2 = params[1], params[2]
    Omh2 = Obh2 + Och2 + Omnu_h2
    rd = cmb.r_drag(wb=Obh2, wm=Omh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, *params)
    chi2_cmb = solve_triang(cmb_cho, delta_cmb)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    return chi2_cmb + chi_bao


bounds = np.array(
    [
        (50, 80),  # H0
        (0.021, 0.023),  # ωb = Ωb * h^2
        (0.05, 0.30),  # ωc = Ωc * h^2
        (-1.5, 0.0),  # w0
    ],
    dtype=np.float64,
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
    from .plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 200
    burn_in = 200
    nsteps = 2200 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
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
    print(f"r*: {cmb.rs_z(Ez, z_st_50, *best_fit):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
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
Dataset: DESI DR2 2024 + (θ∗,ωb,ωbc)CMB
*******************************

Flat ΛCDM w(z) = -1
H0: 68.40 +0.29 -0.29 km/s/Mpc
ωb: 0.02238 +0.00012 -0.00012
ωc: 0.1172 +0.0006 -0.0006
ωm: 0.1402 +0.0006 -0.0006
Ωm: 0.300 +0.004 -0.004
w0: -1
wa: 0
r*: 145.18 Mpc
z*: 1089.68 +0.18 -0.18
r_d: 147.84 +0.18 -0.18 Mpc
Chi squared: 13.55
Degs of freedom: 15

===============================

Flat wCDM w(z) = w0
H0: 68.88 +0.97 -0.92 km/s/Mpc
ωb: 0.02235 +0.00013 -0.00013
ωc: 0.1176 +0.0009 -0.0009
ωm: 0.1405 +0.0009 -0.0009
Ωm: 0.296 +0.007 -0.007
w0: -1.021 +0.038 -0.040 (prior width -1.5 to -0.3)
wa: 0
r*: 145.10 Mpc
z*: 1089.75 +0.22 -0.22
r_d: 147.77 +0.22 -0.22 Mpc
Chi squared: 13.33
Degs of freedom: 14

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)
H0: 68.14 +1.56 -1.46 km/s/Mpc
ωb: 0.02238 +0.00013 -0.00013
ωc: 0.1171 +0.0008 -0.0008
ωm: 0.1401 +0.0008 -0.0008
Ωm: 0.302 +0.013 -0.013
w0: -0.982 +0.107 -0.109 (prior width -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
r*: 145.19 Mpc
z*: 1089.67 +0.21 -0.21
r_d: 147.85 +0.21 -0.20 Mpc
Chi squared: 13.50
Degs of freedom: 14

===============================

Flat w(z) = w0 + wa * z / (1 + z)
Overfits, the uncertainties go wild and the prior are very wide
The posterior volume is also very large, making the evidence small

H0: 63.85 +2.09 -2.07 km/s/Mpc
ωb: 0.02222 +0.00014 -0.00014
ωc: 0.1193 +0.0011 -0.0011
ωm: 0.1421 +0.0010 -0.0010
Ωm: 0.349 +0.025 -0.023
w0: -0.467 +0.253 -0.227 (prior width -2.0 to +1.5)
wa: -1.563 +0.649 -0.753 (prior width -6.0 to 2.5)
r*: 144.75 Mpc
z*: 1090.07 +0.25 -0.25
r_d: 147.45 +0.24 -0.24 Mpc
Chi squared: 7.01
"""
