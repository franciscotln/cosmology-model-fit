from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2022pantheonSHOES.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

c = 299792.458  # Speed of light in km/s

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
legend, z_sn_vals, z_sn_hel_vals, apparent_mag_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]
cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

cho_sn_T = cho_sn.T
cho_bao_T = cho_bao.T
cho_cc_T = cho_cc.T

sn_grid = np.linspace(0, np.max(z_sn_vals), num=1000)
dx_sn = np.diff(sn_grid)

bao_grid = np.linspace(0, np.max(bao_data["z"]), num=1000)
dx_bao = np.diff(bao_grid)


@njit
def Ez(z, params):
    O_m, w0 = params[3], params[4]
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return (O_m * cubed + (1 - O_m) * rho_de) ** 0.5


@njit
def DM(grid, zs, dx, params):
    dh_grid = DH_z(grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(zs, grid, cum_dm)


@njit
def sn_apparent_mag(params):
    comoving_distance = DM(sn_grid, z_sn_vals, dx_sn, params)
    return params[1] + 25 + 5 * np.log10((1 + z_sn_hel_vals) * comoving_distance)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    return DM(bao_grid, z, dx_bao, params)


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
    return results / params[2]


bounds = np.array(
    [
        (40.0, 90.0),  # H0
        (-20.0, -19.0),  # M
        (115.0, 170.0),  # r_d
        (0.0, 1.0),  # Ωm
        (-1.5, 0.0),  # w0
        (0.4, 2.5),  # f_cc
    ],
    dtype=np.float64,
)


def solve_triang(cho_L, cho_L_T, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    z = solve_triangular(cho_L_T, y, lower=False, check_finite=False)
    return delta @ z


def chi_squared(params):
    delta_sn = apparent_mag_values - sn_apparent_mag(params)
    chi_sn = solve_triang(cho_sn, cho_sn_T, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = solve_triang(cho_bao, cho_bao_T, delta_bao)

    f_cc = params[-1]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = solve_triang(cho_cc, cho_cc_T, delta_cc) * f_cc**2

    return chi_sn + chi_bao + chi_cc


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


def log_likelihood(params):
    f_cc = params[-1]
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
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2400 + burn_in
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
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    print("Gelman-Rubin R-hat:", gelman_rubin(chains_samples))

    [
        [h0_16, h0_50, h0_84],
        [M_16, M_50, M_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
        [f_cc_16, f_cc_50, f_cc_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    deg_of_freedom = (
        z_sn_vals.size + bao_data["value"].size + z_cc_vals.size - len(best_fit)
    )

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"H0: {h0_50:.2f} +{(h0_84 - h0_50):.2f} -{(h0_50 - h0_16):.2f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evidence(samples, log_probs, log_probability):.1f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=legend,
        x=z_sn_vals,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=sn_apparent_mag(best_fit),
        label=f"Best fit: $\Omega_m$={Om_50:.3f}, $H_0$={h0_50:.2f} km/s/Mpc",
        x_scale="log",
    )

    labels = ["$H_0$", "M", "$r_d$", "Ωm", "$w_0$", "$f_{CC}$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
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
f_cc: 1.47 +0.19 -0.18
H0: 68.62 +2.28 -2.24 km/s/Mpc
M: -19.403 +0.071 -0.072 mag
r_d: 147.1 +4.9 -4.7 Mpc
Ωm: 0.305 +0.008 -0.008
w0: -1
Chi squared: 1448.42
Log Evidence: -854.9
Degrees of freedom: 1631

===============================

Flat wCDM: w(z) = w0
f_cc: 1.47 +0.18 -0.18
H0: 67.85 +2.28 -2.29
M: -19.414 +0.071 -0.073
r_d: 146.98 +4.95 -4.62
Ωm: 0.304 +0.008 -0.008
w0: -0.899 +0.046 -0.048
Chi squared: 1443.69
Log Evidence: -855.4
Degrees of freedom: 1630

===============================

Flat: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 1.46 +0.18 -0.18
H0: 67.82 +2.32 -2.27 km/s/Mpc
M: -19.415 +0.072 -0.073 mag
r_d: 147.0 +4.9 -4.7 Mpc
Ωm: 0.304 +0.008 -0.008
w0: -0.899 +0.047 -0.048
Chi squared: 1443.55
Log Evidence: -855.6
Degrees of freedom: 1630
"""
