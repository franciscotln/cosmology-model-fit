from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from y2025DESdovekie.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=2000)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    Om, w0 = params[3], params[4]
    zp1 = 1 + z
    cubed = zp1**3
    rho_de = (4 * cubed / (1 + 3 * cubed)) ** (4 * (1 + w0))
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


@njit
def theory_mu(params):
    dL = (1 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


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
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[1]


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao


bounds = np.array(
    [
        (-0.5, 0.5),  # ΔM
        (120.0, 165.0),  # r_d
        (50.0, 90.0),  # H0
        (0.10, 0.50),  # Ωm
        (-1.5, 0.0),  # w0
    ],
    dtype=np.float64,
)

# Planck prior on Ωm * h^2
# Fit r_drag directly as a free parameter without early universe physics
Omh2_planck = 0.1430
Omh2_planck_sigma = 0.0011

# log-normalization for the prior:
norm_uniform = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))
norm_gauss_omh2 = -0.5 * np.log(2 * np.pi * Omh2_planck_sigma**2)


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    Omh2 = params[3] * (params[2] / 100) ** 2
    log_prior_omh2 = -0.5 * ((Omh2_planck - Omh2) / Omh2_planck_sigma) ** 2
    return norm_uniform + norm_gauss_omh2 + log_prior_omh2


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
    from .plot_predictions import plot_bao_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 300
    nsteps = 3000 + burn_in
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

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    [
        (dM_16, dM_50, dM_84),
        (rd_16, rd_50, rd_84),
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {len(bao_data['z']) + sn_size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
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
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$Δ_M$", "$r_d$", "$H_0$", "$Ω_M$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM w(z) = -1
r_d: 147.58 +1.22 -1.21 Mpc
H0: 68.34 +0.91 -0.88 km/s/Mpc
Ωm: 0.306 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1645.3
Log evidence: -832.1
Degrees of freedom: 1723

===============================

Flat wCDM w(z) = w0
r_d: 143.79 +2.08 -2.14 Mpc
H0: 69.35 +1.06 -1.02 km/s/Mpc
Ωm: 0.297 +0.009 -0.009
w0: -0.909 +0.038 -0.037
wa: 0
Chi squared: 1639.5
Log evidence: -832.0 (Δ logZ = 0.1 against ΛCDM)
Degrees of freedom: 1722

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)**3)
r_d: 145.17 +1.52 -1.53 Mpc
H0: 68.54 +0.90 -0.89 km/s/Mpc
Ωm: 0.304 +0.008 -0.008
w0: -0.869 +0.049 -0.050
wa: d w(z)/dz at z=0 = -(9/4) * (1 + w0)
Chi squared: 1638.5
Log evidence: -831.2 (Δ logZ = 0.9 against ΛCDM)
Degrees of freedom: 1722

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
r_d: 147.29 +2.70 -3.66 Mpc
H0: 67.54 +1.89 -1.43 km/s/Mpc
Ωm: 0.314 +0.013 -0.017
w0: -0.846 +0.072 -0.066
wa: -0.516 +0.470 -0.468
Chi squared: 1638.1
Log evidence: -832.9 (Δ logZ = -0.8 in favour of ΛCDM)
Degrees of freedom: 1721
"""
