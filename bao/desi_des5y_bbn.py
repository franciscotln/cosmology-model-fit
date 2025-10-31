from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
import y2024BBN.prior_lcdm_schoneberg as bbn
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data


c = c0 / 1000  # km/s

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def r_drag(wb, wm):
    """
    arXiv:2106.00428v2 (eq 8)
    Alternatively z_drag from the same paper can we used
    to compute the integral over c_s / H(z) yielding the same results.
    """
    a1 = 0.00257366
    a2 = 0.05032
    a3 = 0.013
    a4 = 0.7720642
    a5 = 0.24346362
    a6 = 0.00641072
    a7 = 0.5350899
    a8 = 32.7525
    a9 = 0.315473

    term_A_denominator = (a1 * (wb**a2)) + (a3 * (wb**a4) * (wm**a5)) + (a6 * (wm**a7))
    term_A = 1.0 / term_A_denominator
    term_B = a8 / (wm**a9)
    return term_A - term_B


@njit
def Ez(z, params):
    Om, w0 = params[1], params[3]
    Ode = 1 - Om
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = np.exp((1 + w0) * (1 - 1 / cubed))

    return np.sqrt(Om * cubed + Ode * rho_de)


def theory_mu(params):
    dL = (1 + z_hel) * DM_z(z_cmb, params)
    return params[-1] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


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
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    rd = r_drag(wb=Obh2, wm=Omh2)
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
    delta_bbn = bbn.Obh2 - params[2]
    chi2_bbn = (delta_bbn / bbn.Obh2_sigma) ** 2

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_bbn + chi_bao + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-1.5, 0.0),  # w0
        (-0.7, 0.7),  # ΔM
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
    from .plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2200 + burn_in
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
        print("mean acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    one_sigma_contours = [15.9, 50, 84.1]

    pct = np.percentile(samples, one_sigma_contours, axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (w0_16, w0_50, w0_84),
        (dM_16, dM_50, dM_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    rd_samples = r_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_contours)
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, one_sigma_contours)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degrees of freedom: {1 + len(bao_data['z']) + sn_size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
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
        labels=["$H_0$", "$Ω_m$", "$ω_b$", "$w_0$", "$Δ_M$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
*******************************
DESI DR2 + DES5Y + BBN Schöngerg+2024
*******************************

Flat ΛCDM w(z) = -1
H0: 68.9 +0.6 -0.6 km/s/Mpc
Ωm: 0.3106 +0.0079 -0.0077
ωb: 0.02218 +0.00054 -0.00054
ωm: 0.14728 +0.00488 -0.00474
w0: -1
wa: 0
ΔM: -0.040 +0.021 -0.020
r_d: 145.99 +1.48 -1.46 Mpc
Chi squared: 1658.97
Log Evidence: -841.81
Degrees of freedom: 1745

===============================

Flat wCDM w(z) = w0
H0: 65.2 +1.2 -1.3 km/s/Mpc
Ωm: 0.2982 +0.0090 -0.0088
ωb: 0.02219 +0.00054 -0.00055
ωm: 0.12696 +0.00766 -0.00743
w0: -0.872 +0.038 -0.038 (prior width 1.5: -1.5 to 0.0)
wa: 0
ΔM: -0.128 +0.034 -0.035
r_d: 151.55 +2.37 -2.32 Mpc
Chi squared: 1648.10
Log Evidence: -839.23 (Δ logZ = 2.58 against ΛCDM)
Degrees of freedom: 1744

===============================

Flat w(z) = -1 + (1 + w0) / (1 + z)^3
H0: 66.0 +1.0 -0.9 km/s/Mpc
Ωm: 0.3094 +0.0079 -0.0077
ωb: 0.02218 +0.00054 -0.00054
ωm: 0.13471 +0.00553 -0.00535
w0: -0.800 +0.054 -0.054 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -3 * (1 + w0)
ΔM: -0.093 +0.025 -0.025
r_d: 149.35 +1.72 -1.70 Mpc
Chi squared: 1646.00
Log Evidence: -837.80 (Δ logZ = 4.81 against ΛCDM)
Degrees of freedom: 1744

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.3 +1.4 -1.5 km/s/Mpc
Ωm: 0.3219 +0.0126 -0.0149
ωb: 0.02219 +0.00054 -0.00054
ωm: 0.14598 +0.01045 -0.01208
w0: -0.782 +0.071 -0.068 (prior width 1.5: -1.5 to 0.0)
wa: -0.740 +0.447 -0.452 (prior width 4.5: -3.0 to 1.5)
ΔM: -0.046 +0.044 -0.052
r_d: 146.35 +3.34 -2.74 Mpc
Chi squared: 1645.51
Log Evidence: -839.11 (Δ logZ = 2.70 against ΛCDM)
Degrees of freedom: 1743
"""
