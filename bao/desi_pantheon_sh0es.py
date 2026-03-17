from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data_shoes import get_data

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
legend, z_cmb, z_hel, mb_vals, ceph_dists, cov_matrix_sn = get_data()

ceph_mask = ceph_dists != -9

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    return (1.0 + z) ** (3 * (1.0 + w0))  # wCDM


@njit
def H_z(z, theta):
    H0, Om = theta[1], theta[2]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


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


correction_mask = ~ceph_mask & (z_cmb <= 0.1)


@njit
def mu_corr(params):
    z_pec = 100 * params[4] / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)

    return np.where(
        correction_mask,
        5.0 * np.log10(DM_z(z_cosmo, params) / DM_z(z_cmb, params)),
        0.0,
    )


@njit
def mu_theory(theta):
    return 25 + 5 * np.log10((1.0 + z_hel) * DM_z(z_cmb, theta))


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi2_sn(theta):
    mu_pred = np.where(ceph_mask, ceph_dists, mu_theory(theta))
    mB_theory = mu_pred + theta[0]
    delta_sn = mb_vals - mB_theory - mu_corr(theta)
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
        (-3.0, 1.5),  # v x 100 km/s
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
    print(f"v: {v_f_50:.3f} +{(v_f_84 - v_f_50):.3f} -{(v_f_50 - v_f_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(samples[np.argmax(log_probs)]):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {len(bao_data) + len(z_cmb) - len(best_fit)}")

    labels = ["$M_0$", "$H_0$", "$Ω_m$", "$r_{drag}$", "$v_{100}$"]
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
        y=mb_vals - (M_50 + mu_corr(best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
M0: -19.247 +0.030 -0.029 mag
H0: 73.7 +1.0 -1.0 km/s/Mpc
Ωm: 0.305 +0.008 -0.008
rd: 136.9 +2.0 -2.0 Mpc
Chi2 (MAP): 1465.3
Log evidence: -747.9
Degrees of freedom: 1666
"""

"""
Flat ΛCDM
Isotropic velocity SNe observed redshifts (limit to z <= 0.1)
z_cosmo = -1 + (1 + z) / (1 + v/c)

v: -86 +- 40 km/s
M: -19.244 +- 0.030 mag
H0: 74.23 +- 1.05 km/s/Mpc
Ωm: 0.301 +- 0.008
rd: 136.5 +2.0 -2.0 Mpc
Chi2 (MAP): 1460.7 (2.14 sigma away from no v corrections)
Log evidence: -747.1 (Δ logZ = 0.8 in favour of v corrections)
Degrees of freedom: 1665
"""

"""
Flat wCDM
w0: -0.915 +0.039 -0.040 (prior ~ U(-1.3, -0.6))
M0: -19.244 +0.029 -0.029 mag
H0: 73.4 +1.0 -1.0 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
rd: 135.9 +2.1 -2.0 Mpc
Chi2 (MAP): 1460.8 (2.1 sigma away from ΛCDM)
Log evidence: -747.6 (Δ logZ = 0.3 against ΛCDM)
Degrees of freedom: 1665
"""

"""
Flat w0waCDM
TODO
"""
