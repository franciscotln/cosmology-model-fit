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
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    cubic = (1.0 + z) ** 3
    return (2 * cubic / (1.0 + w0 + (1.0 - w0) * cubic)) ** 2


@njit
def Ez(z, params):
    om = params[4]
    return np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om) * Ode_z(z, w0=params[5]))


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


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
    return results / params[3]


@njit
def chi_squared(params, f_cc_arr):
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = solve_triangular(cho_cc, f_cc_arr * delta_cc)

    delta_bao = data["value"] - theory_bao(data["z"], desi_qty, params)
    chi_bao = solve_triangular(cho_bao, delta_bao)
    return chi_cc + chi_bao


bounds = np.array(
    [
        (0.1, 6.0),  # f0_cc
        (-9.0, 9.0),  # fa_cc
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
    f0_cc, fa_cc = params[0], params[1]
    f_cc_arr = f0_cc + fa_cc * z_cc_vals / (1.0 + z_cc_vals)
    if np.any(f_cc_arr <= 1e-4):
        return -np.inf

    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2.0 * np.log(f_cc_arr).sum()
    return -0.5 * chi_squared(params, f_cc_arr) - 0.5 * normalization_cc


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

    print(f"Gelman-Rubin: {gelman_rubin(chains_samples)}")

    [
        (f0_16, f0_50, f0_84),
        (fa_16, fa_50, fa_84),
        (h0_16, h0_50, h0_84),
        (rd_16, rd_50, rd_84),
        (Om_16, Om_50, Om_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = samples[np.argmax(log_probs)]
    f_array = best_fit[0] + best_fit[1] * z_cc_vals / (1.0 + z_cc_vals)

    Omh2_samples = samples[:, 1] ** 2 * samples[:, 3] / 100**2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    print(f"f0_cc: {f0_50:.2f} +{(f0_84 - f0_50):.2f} -{(f0_50 - f0_16):.2f}")
    print(f"fa_cc: {fa_50:.1f} +{(fa_84 - fa_50):.1f} -{(fa_50 - fa_16):.1f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit, f_array):.2f}")
    print(f"log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Degrees of freedom: {len(data['z']) + len(z_cc_vals) - len(best_fit)}")

    labels = ["$f_{0CCH}$", "$f_{aCCH}$", "$H_0$", "$r_d$", "$Ω_m$", "$w_0$"]
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
        H_err=np.sqrt(np.diag(cc_cov_matrix)) / f_array,
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
# f0_cc: 3.01 +0.57 -0.56
# fa_cc: -3.4 +1.1 -1.1
# H0: 68.3 +1.8 -1.8 km/s/Mpc
# r_d: 148.6 +4.0 -3.8 Mpc
# Ωm: 0.298 +0.008 -0.008
# ωm: 0.1705 +0.1309 -0.0959
# Chi squared: 48.21
# log likelihood: -154.64
# Degrees of freedom: 47
# ---------------------------------


# ----------- Flat wCDM -----------
# f0_cc: 2.97 +0.56 -0.56
# fa_cc: -3.3 +1.2 -1.1
# H0: 67.7 +2.0 -2.0 km/s/Mpc
# r_d: 148.2 +4.1 -3.9 Mpc
# Ωm: 0.297 +0.009 -0.009
# ωm: 0.1633 +0.1300 -0.0951
# w0: -0.939 +0.074 -0.076 (prior U[-2, 0])
# Chi squared: 48.26
# log likelihood: -154.30
# Degrees of freedom: 46
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
# f0_cc: 2.93 +0.56 -0.55
# fa_cc: -3.3 +1.1 -1.1
# H0: 66.6 +2.1 -2.1 km/s/Mpc
# r_d: 148.3 +4.1 -3.9 Mpc
# Ωm: 0.309 +0.011 -0.011
# ωm: 0.1563 +0.1268 -0.0893
# w0: -0.825 +0.123 -0.108 (prior U[-1, 0]. Posterior truncated to the left of the mean)
# wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
# Chi squared: 46.14
# log likelihood: -154.16
# Degrees of freedom: 46
# ---------------------------------


# ---------- Flat w0waCDM----------
# Enforced w0 + wa < 0 in likelihood
#
# f0_cc: 2.90 +0.56 -0.54
# fa_cc: -3.2 +1.1 -1.1
# H0: 65.4 +3.1 -3.0 km/s/Mpc
# r_d: 148.6 +4.1 -3.9 Mpc
# Ωm: 0.334 +0.033 -0.043
# ωm: 0.1566 +0.1236 -0.0896
# w0: -0.675 +0.297 -0.292 (prior U[-3, 1])
# wa: -1.088 +1.233 -1.098 (prior U[-3, 2])
# Chi squared: 47.65
# log likelihood: -153.83
# Degrees of freedom: 45
# ---------------------------------
