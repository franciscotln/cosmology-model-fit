from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data_shoes import get_data

bao_legend, data, bao_cov_matrix = get_bao_data()
legend, z_cmb, z_hel, mb_vals, ceph_dists, cov_matrix_sn = get_data(z_cut_ceph=0.0043)

ceph_mask = ceph_dists != -9

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    return (1.0 + z) ** (3 * (1.0 + w0)) # wCDM


@njit
def Ez(z, theta):
    Om = theta[2]
    zp1 = 1.0 + z
    cubed = zp1**3
    return np.sqrt(Om * cubed + (1.0 - Om))


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
desi_qty = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


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
def v_bulk(theta):
    return 100 * theta[4] * (5 / np.log(10)) / (c * z_cmb)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    mu_the = np.where(ceph_mask, ceph_dists, mu_theory(theta))
    mb_theory = mu_the + theta[0] + v_bulk(theta)
    delta_sn = mb_vals - mb_theory
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = data["value"] - bao_theory(data["z"], desi_qty, theta)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao
    return chi_sn + chi_bao


bounds = np.array(
    [
        (-20.0, -18.5),  # M
        (50.0, 100.0),  # H0
        (0.2, 0.7),  # Ωm
        (120.0, 170.0),  # rd
        (-1.5, 3.5),  # v_bulk in units of 100 km/s
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
        (v_b_16, v_b_50, v_b_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    print(f"M0: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"v_bulk: {v_b_50:.3f} +{(v_b_84 - v_b_50):.3f} -{(v_b_50 - v_b_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(samples[np.argmax(log_probs)]):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {data['z'].size + z_cmb.size - len(best_fit)}")

    labels = ["$M_0$", "$H_0$", "$Ω_m$", "$r_{drag}$", "$v_{bulk}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=legend,
        x=z_cmb,
        y=mb_vals - M_50 - v_bulk(best_fit),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
M0: -19.232 +0.032 -0.032 mag
H0: 74.25 +1.13 -1.11 km/s/Mpc
Ωm: 0.305 +0.008 -0.008
rd: 136.00 +2.22 -2.17 Mpc
Chi2 (MAP): 1450.2
Log evidence: -740.3
Degrees of freedom: 1650
"""

"""
Flat ΛCDM
Bulk v_pec corrections of SNe M(z) = M0 + v_pec_corr
v_pec_corr = 100 * v_pec * (5 / np.log(10)) / (c * z_cmb) with v_pec in units 100 km/s

v_pec: 100 +40 -41 km/s (prior ~ U(-1.5, 3.5) in units of 100 km/s)
M0: -19.335 +0.053 -0.053 mag
H0: 71.28 +1.62 -1.58 km/s/Mpc
Ωm: 0.300 +0.008 -0.008
rd: 142.14 +3.44 -3.40 Mpc
Chi2 (MAP): 1444.3 (2.43 sigma away from no v_pec correction)
Log evidence: -738.9 (Δ logZ = 1.4 in favor of v_pec correction)
Degrees of freedom: 1649
"""

"""
Flat wCDM
w0: -0.914 +0.040 -0.040 (prior ~ U(-1.5, -0.5))
M0: -19.229 +0.032 -0.032 mag
H0: 73.95 +1.12 -1.11 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
rd: 134.94 +2.25 -2.20 Mpc
Chi2 (MAP): 1445.6 (2.14 sigma away from ΛCDM)
Log evidence: -740.3 (Δ logZ = 0.0, same evidence as ΛCDM)
Degrees of freedom: 1649
"""

"""
Flat w0waCDM (enforced w0 + wa < 0)
M0: -19.228 +0.032 -0.032 mag
H0: 73.88 +1.11 -1.11 km/s/Mpc
Ωm: 0.304 +0.015 -0.023
w0: -0.891 +0.061 -0.057 (prior ~ U(-1.5, -0.5))
wa: -0.18 +0.48 -0.45 (prior ~ U(-2.5, 2.5))
Chi2 (MAP): 1445.5
Log evidence: -741.4 (Δ logZ = -1.1 in favour of ΛCDM)
Degrees of freedom: 1648
"""
