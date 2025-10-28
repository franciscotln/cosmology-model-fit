from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
import y2024BBN.prior_lcdm_shonberg as bbn
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
def r_drag(wb, wm, n_eff=3.04):  # arXiv:2503.14738v2 (eq 2)
    return (
        147.05 * (0.02236 / wb) ** 0.13 * (0.1432 / wm) ** 0.23 * (3.04 / n_eff) ** 0.1
    )


@njit
def Ez(z, params):
    Om, w0 = params[1], params[3]
    Ode = 1 - Om
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed**2 / (1 + cubed**2)) ** (1 + w0)

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


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

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
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.30),
                (emcee.moves.DEMove(), 0.56),
                (emcee.moves.DESnookerMove(), 0.14),
            ],
        )
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

    one_sigma_contours = [15.9, 50, 84.1]

    pct = np.percentile(samples, one_sigma_contours, axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]
    dM_16, dM_50, dM_84 = pct[4]

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
    print(
        f"Log Evidence: {log_evidence(samples, log_probs, log_probability, bounds):.2f}"
    )
    print(f"Degrees of freedom: {1 + bao_data['z'].size + sn_size - len(best_fit)}")

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
Flat ΛCDM w(z) = -1
H0: 68.6 +0.5 -0.5 km/s/Mpc
Ωm: 0.3105 +0.0079 -0.0077
ωb: 0.02219 +0.00053 -0.00053
ωm: 0.14621 +0.00430 -0.00417
w0: -1
ΔM: -0.047 +0.017 -0.018 mag
r_d: 146.50 +1.24 -1.23 Mpc
Chi squared: 1658.97
Log Evidence: -841.93
Degrees of freedom: 1745

===============================

Flat wCDM w(z) = w0
H0: 65.4 +1.1 -1.2 km/s/Mpc
Ωm: 0.2980 +0.0088 -0.0089
ωb: 0.02219 +0.00054 -0.00054
ωm: 0.12741 +0.00715 -0.00708
w0: -0.872 +0.037 -0.038 (prior width 1.5: -1.5 to 0.0)
ΔM: -0.123 +0.031 -0.032 mag
r_d: 151.22 +2.15 -2.05 Mpc
Chi squared: 1648.09
Log Evidence: -839.26 (Δ logZ = 2.67 against ΛCDM)
Degrees of freedom: 1744

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**6)
H0: 66.2 +0.8 -0.8 km/s/Mpc
Ωm: 0.3116 +0.0079 -0.0077
ωb: 0.02218 +0.00054 -0.00054
ωm: 0.13635 +0.00486 -0.00470
w0: -0.778 +0.059 -0.059 (prior width 1.5: -1.5 to 0.0)
ΔM: -0.085 +0.021 -0.021
r_d: 148.87 +1.45 -1.41 Mpc
Chi squared: 1645.67
Log Evidence: -837.62 (Δ logZ = 4.31 against ΛCDM)
Degrees of freedom: 1744

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.1 +1.1 -1.3 km/s/Mpc
Ωm: 0.3217 +0.0126 -0.0149
ωb: 0.02218 +0.00053 -0.00054
ωm: 0.14495 +0.00912 -0.01106
w0: -0.782 +0.072 -0.068 (prior width 1.5: -1.5 to 0.0)
wa: -0.735 +0.452 -0.451 (prior width 4.5: -3.0 to 1.5)
ΔM: -0.053 +0.036 -0.045
r_d: 146.83 +2.81 -2.22 Mpc
Chi squared: 1645.56
Log Evidence: -839.24 (Δ logZ = 2.69 against ΛCDM)
Degrees of freedom: 1743
"""
