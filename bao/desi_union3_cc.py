from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2023union3.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, sn_mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = 299792.458  # Speed of light in km/s

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, params):
    O_m, w0 = params[4], params[5]
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubed + (1 - O_m) * rho_de)


@njit
def DM(params):
    dh_grid = DH_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return cum_dm


@njit
def mu_theory(params):
    dL = (1 + z_sn_vals) * np.interp(z_sn_vals, z_grid, DM(params))
    return params[1] + 25 + 5 * np.log10(dL)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


@njit
def DM_z(z, params):
    return np.interp(z, z_grid, DM(params))


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

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[3]


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_sn = sn_mu_vals - mu_theory(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    f_cc = params[0]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = solve_triang(cho_cc, delta_cc) * f_cc**2

    return chi_sn + chi_bao + chi_cc


bounds = np.array(
    [
        (0.5, 2.5),  # f_cc: overestimation of CC errors we divide the cov by f^2
        (-0.7, 0.7),  # ΔM
        (55, 80),  # H0
        (125, 170),  # r_d
        (0.2, 0.7),  # Ωm
        (-1.5, 0.0),  # w0
    ],
    dtype=np.float64,
)

normalization = -np.log(np.prod(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee, corner
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from cosmic_chronometers.plot_predictions import plot_cc_predictions
    from .plot_predictions import plot_bao_predictions
    from log_evidence import log_evidence

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

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, pool=pool, moves=moves
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [dM_16, dM_50, dM_84],
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 4] * samples[:, 2] ** 2 / 100**2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    deg_of_freedom = (
        len(z_sn_vals) + len(bao_data["value"]) + len(z_cc_vals) - len(best_fit)
    )

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evidence(samples, log_probs, log_probability):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=sn_mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f}, $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_50,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )

    labels = ["$f_{CCH}$", "ΔM", "$H_0$", "$r_d$", "Ωm", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()

    plt.figure(figsize=(16, 1.5 * ndim))
    for n in range(ndim):
        plt.subplot2grid((ndim, 1), (n, 0))
        plt.plot(chains_samples[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
f_cc: 1.48 +0.19 -0.18
H0: 68.6 +2.3 -2.3 km/s/Mpc
r_d: 147.1 +4.9 -4.6 Mpc
Ωm: 0.304 +0.008 -0.008
ωm: 0.1435 +0.0095 -0.0093
Chi squared: 71.17
Log evidence: -161.76
Degrees of freedom: 63

===============================

Flat wCDM: w(z) = w0
f_cc: 1.46 +0.18 -0.18
H0: 67.1 +2.3 -2.3 km/s/Mpc
r_d: 147.2 +5.0 -4.6 Mpc
Ωm: 0.298 +0.009 -0.009
ωm: 0.1344 +0.0099 -0.0096
w0: -0.871 +0.051 -0.051 (prior width 1.5: -1.5 to 0.0)
Chi squared: 64.49
Log evidence: -161.11 (Δ logZ = 0.65 over ΛCDM)
Degrees of freedom: 62

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 1.46 +0.18 -0.18
H0: 66.7 +2.4 -2.3 km/s/Mpc
r_d: 147.2 +5.0 -4.7 Mpc
Ωm: 0.310 +0.009 -0.009
ωm: 0.1378 +0.0097 -0.0091
w0: -0.812 +0.066 -0.067 (prior width 1.5: -1.5 to 0.0)
Chi squared: 62.78
Log evidence: -160.17 (Δ logZ = 1.59 over ΛCDM)
Degrees of freedom: 62

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
f_cc: 1.45 +0.18 -0.17
H0: 66.3 +2.3 -2.4 km/s/Mpc
r_d: 147.1 +5.0 -4.7 Mpc
Ωm: 0.329 +0.016 -0.018
ωm: 0.1440 +0.0111 -0.0113
w0: -0.723 +0.114 -0.108 (prior width 1.5: -1.5 to 0.0)
wa: -0.897 +0.564 -0.562 (prior width 5.0: -3.0 to 2.0)
Chi squared: 61.12
Log evidence: -161.00 (Δ logZ = 0.76 over ΛCDM)
Degrees of freedom: 61
"""
