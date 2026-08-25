from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cc_cov_matrix = get_cc_data()
bao_legend, data, bao_cov_matrix = get_bao_data()

cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]
cho_cc = cho_factor(cc_cov_matrix, lower=True)[0]

logdet_cc = np.linalg.slogdet(cc_cov_matrix)[1]
N_cc = len(z_cc_vals)

c = c0 / 1000  # Speed of light in km/s

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    cubic = (1.0 + z) ** 3
    return (2 * cubic / (1.0 + w0 + (1.0 - w0) * cubic)) ** 2


@njit
def Ez(z, params):
    O_m = params[3]
    return np.sqrt(O_m * (1.0 + z) ** 3 + (1.0 - O_m) * Ode_z(z, w0=params[4]))


@njit
def H_z(z, params):
    return params[1] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
desi_qty = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def theory_bao(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[2]


@njit
def chi_squared(params):
    f_cc = params[0]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = f_cc**2 * solve_triangular(cho_cc, delta_cc)

    delta_bao = data["value"] - theory_bao(data["z"], desi_qty, params)
    chi_bao = solve_triangular(cho_bao, delta_bao)
    return chi_cc + chi_bao


bounds = np.array(
    [
        (0.5, 2.5),  # f_cc
        (45.0, 90.0),  # H0
        (120.0, 175.0),  # r_d
        (0.1, 0.7),  # Ωm
        (-1.0, 0.0),  # w0
    ]
)

normalization = -np.log(np.prod(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


@njit
def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


@njit
def log_probability_jit(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def log_probability(params):
    return log_probability_jit(params)


def main():
    import emcee
    from corner_plot import plot_corner_and_chains
    from multiprocessing import Pool
    from ohd.plot_predictions import plot_cc_predictions
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from bao.plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 100
    burn_in = 1000
    nsteps = 4000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"})

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
    print(f"Gelman-Rubin: {gelman_rubin(chains_samples)}")

    [
        (f_cc_16, f_cc_50, f_cc_84),
        (h0_16, h0_50, h0_84),
        (rd_16, rd_50, rd_84),
        (Om_16, Om_50, Om_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 1] ** 2 * samples[:, 3] / 100**2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {len(data['z']) + len(z_cc_vals) - len(best_fit)}")

    labels = ["$f_{CCH}$", "$H_0$", "$r_d$", "$Ω_m$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_bao(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=f"{bao_legend}: $H_0$={h0_50:.2f}, $r_d$={rd_50:.2f}",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cc_cov_matrix)) / f_cc_50,
        label=f"{cc_legend}: $H_0$={h0_50:.1f} km/s/Mpc",
    )


if __name__ == "__main__":
    main()


# ********************************
# Data sets:
# - DESI DR2
# - CCH compilation
# ********************************


# ----------- Flat ΛCDM -----------
# f_cc: 1.50 +0.17 -0.17
# H0: 68.9 +2.3 -2.3 km/s/Mpc
# r_d: 147.2 +4.9 -4.6 Mpc
# Ωm: 0.299 +0.009 -0.008
# ωm: 0.1418 +0.0093 -0.0090
# Chi squared: 47.60
# log likelihood: -154.49
# Log evidence: -165.5
# Degrees of freedom: 47
# ---------------------------------


# ----------- Flat wCDM -----------
# f_cc: 1.50 +0.18 -0.17
# H0: 67.8 +2.5 -2.5 km/s/Mpc
# r_d: 147.4 +4.9 -4.5 Mpc
# Ωm: 0.298 +0.009 -0.009
# ωm: 0.1369 +0.0103 -0.0100
# w0: -0.922 +0.074 -0.077 (prior U[-2, 0])
# Chi squared: 46.44
# log likelihood: -153.97
# Log evidence: -167.3
# Degrees of freedom: 46
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
#
# f_cc: 1.49 +0.17 -0.17
# H0: 66.5 +2.6 -2.6 km/s/Mpc
# r_d: 147.5 +4.9 -4.6 Mpc
# Ωm: 0.311 +0.012 -0.011
# ωm: 0.1380 +0.0094 -0.0092
# w0: -0.795 +0.128 -0.120 (prior U[-1, 0]. Posterior truncated at 1.7 sigma to the left of the mean)
# wa: d w(z)/dz at z=0 = -1.5 * (1 - w0**2)
# Chi squared: 45.92
# log likelihood: -153.82
# Log evidence: -165.7 (needs Nautilus for better estimate)
# Degrees of freedom: 46
# ---------------------------------


# ---------- Flat w0waCDM----------
# Enforced w0 + wa < 0 in likelihood
#
# f_cc: 1.47 +0.17 -0.17
# H0: 65.3 +3.3 -3.2 km/s/Mpc
# r_d: 147.5 +4.9 -4.6 Mpc
# Ωm: 0.339 +0.034 -0.045
# ωm: 0.1430 +0.0113 -0.0127
# w0: -0.63 +0.30 -0.30 (prior U[-2, 1])
# wa: -1.2 +1.3 -1.1 (prior U[-3, 1])
# Chi squared: 44.03
# log likelihood: -153.41
# Log evidence: -167.4
# Degrees of freedom: 45
# ---------------------------------
