from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data_shoes import get_data

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
legend, z_cmb, z_hel, mb_vals, ceph_dists, cov_matrix_sn = get_data(z_cut_ceph=0.0044)
# At z_ceph < 0.0044 it seems the outflow velocity drops to zero or even becomes slightly negative.

ceph_mask = ceph_dists != -9

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    return (1.0 + z) ** (3 * (1.0 + w0))  # wCDM


@njit
def Ez(z, theta):
    Om = theta[2]
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def H_z(z, theta):
    return theta[1] * Ez(z, theta)


@njit
def DH_z(z, theta):
    return c / H_z(z, theta)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
desi_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, theta):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / theta[3]


@njit
def mu_theory(theta):
    return 25 + 5 * np.log10((1.0 + z_hel) * DM_z(z_cmb, theta))


@njit
def v_outflow(v_flow):
    return 100 * v_flow * (5 / np.log(10)) / (c * z_cmb)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi2_sn(theta):
    mu_the = np.where(ceph_mask, ceph_dists, mu_theory(theta))
    mb_theory = mu_the + theta[0] + v_outflow(theta[4])
    delta_sn = mb_vals - mb_theory
    return solve_triang(cho_sn, delta_sn)


@njit
def chi2_bao(theta):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, theta)
    return delta_bao @ inv_cov_bao @ delta_bao


def chi_squared(theta):
    return chi2_sn(theta) + chi2_bao(theta)


bounds = np.array(
    [
        (-20.0, -18.5),  # M (mag)
        (50.0, 100.0),  # H0 (km/s/Mpc)
        (0.2, 0.7),  # Ωm
        (120.0, 170.0),  # rd (Mpc)
        (-1.5, 3.5),  # v_outflow in units of 100 km/s
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(theta):
    return -0.5 * chi_squared(theta)


def log_probability(theta):
    lp = log_prior(theta)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def main():
    import emcee
    from multiprocessing import Pool
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.25),
        (emcee.moves.DEMove(), 0.75),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

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

    [
        (M_16, M_50, M_84),
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (rd_16, rd_50, rd_84),
        (v_f_16, v_f_50, v_f_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    print(f"M0: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"v_flow: {v_f_50:.3f} +{(v_f_84 - v_f_50):.3f} -{(v_f_50 - v_f_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(samples[np.argmax(log_probs)]):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {len(bao_data) + len(z_cmb) - len(best_fit)}")

    labels = ["$M_0$", "$H_0$", "$Ω_m$", "$r_{drag}$", "$v_{flow}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=legend,
        x=z_cmb,
        y=mb_vals - M_50 - v_outflow(v_f_50),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
M0: -19.230 +- 0.033 mag
H0: 74.3 +- 1.1 km/s/Mpc
Ωm: 0.304 +- 0.008
rd: 135.9 +- 2.2 Mpc
Chi2 (MAP): 1448.7
Log evidence: -739.5
Degrees of freedom: 1648
"""

"""
Flat ΛCDM
Void outflow corrections of SNe M(z) = M_inf + v_corr
v_corr = 100 * v_flow * (5 / np.log(10)) / (c * z_cmb) with v_flow in units 100 km/s

v_flow: 104.3 +- 40.8 km/s (prior ~ U(-1.5, 3.5) in units of 100 km/s)
M_inf: -19.336 +- 0.053 mag
H0: 71.2 +- 1.6 km/s/Mpc
Ωm: 0.300 +- 0.008
rd: 142.2 +- 3.4 Mpc
Chi2 (MAP): 1442.2 (2.55 sigma away from no outflow case)
Log evidence: -737.9 (Δ logZ = +1.6 in favour of outflow model)
Degrees of freedom: 1647
"""

"""
Flat wCDM
w0: -0.915 +0.040 -0.040 (prior ~ U(-1.5, -0.5))
M0: -19.227 +0.032 -0.033 mag
H0: 74.02 +1.13 -1.12 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
rd: 134.83 +2.25 -2.21 Mpc
Chi2 (MAP): 1444.2 (2.12 sigma away from ΛCDM)
Log evidence: -739.5 (Δ logZ = 0.0 identical to ΛCDM)
Degrees of freedom: 1647
"""

"""
Flat w0waCDM
M0: -19.227 +- 0.033 mag
H0: 74.0 +- 1.1 km/s/Mpc
Ωm: 0.304 +0.015 -0.023
rd: 134.9 +- 2.2 Mpc
w0: -0.892 +0.061 -0.057 (prior ~ U(-1.5, -0.5))
wa: -0.17 +0.47 -0.45 (prior ~ U(-2.5, 2.5))
Chi2 (MAP): 1443.9 (1.69 sigma away from ΛCDM)
Log evidence: -740.5 (Δ logZ = -1.0 in favour of ΛCDM)
Degrees of freedom: 1646
"""
