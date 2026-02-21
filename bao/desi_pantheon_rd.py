from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data import get_data

legend, z_cmb, z_hel, mb_vals, cov_matrix_sn = get_data()
bao_legend, data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(bao_cov_matrix, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s

z_max = max(np.max(z_cmb), np.max(data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    cubed = zp1**3
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2  # thawing quintessence


@njit
def Ez(z, theta):
    Om, w0 = theta[2], theta[4]
    zp1 = 1.0 + z
    cubed = zp1**3
    return np.sqrt(Om * cubed + (1.0 - Om) * Ode_z(z, w0))


@njit
def apparent_mag(theta):
    dL = (1.0 + z_hel) * DM_z(z_cmb, theta)
    return theta[0] + 25 + 5 * np.log10(dL)


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
    cum_dm[1:] = np.cumsum(dz * dh)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


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


bounds = np.array(
    [
        (-20.0, -19.0),  # M
        (50.0, 100.0),  # H0
        (0.2, 0.7),  # Ωm
        (144.0, 150.0),  # rd
        (-1.0, -1 / 3),  # w0
    ]
)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    delta_sn = mb_vals - apparent_mag(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = data["value"] - bao_theory(data["z"], quantities, theta)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    prior_rd = -0.5 * ((theta[3] - 147.14) / 0.29) ** 2  # Planck + ACT
    return normalization + prior_rd


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
    burn_in = 200
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
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    print(f"M0: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(samples[np.argmax(log_probs)]):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {data['z'].size + z_cmb.size - len(best_fit)}")

    labels = ["$M_0$", "$H_0$", "$Ω_m$", "$r_{drag}$", "$w_0$"]
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
        y=mb_vals - M_50,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag(best_fit) - M_50,
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
M0: -19.403 +- 0.013 mag
H0: 68.65 +- 0.48 km/s/Mpc
Ωm: 0.304 +- 0.008
r_d: 147.14 +- 0.29 Mpc
Chi2 (MAP): 1416.1
Log evidence: -722.7
Degrees of freedom: 1599
"""

"""
Flat ΛCDM
v_bulk corrections of SNe M(z) = M0 + v_bulk_corr
v_bulk_corr = 100 * v_bulk * (5 / ln(10)) / (c * z_cmb) with v_bulk in units 100 km/s

M: -19.410 +- 0.014 mag
v_bulk: 95 +- 43 km/s (prior ~ U(-1.30, 3.15))
H0: 68.92 +- 0.49 km/s/Mpc
Ωm: 0.300 +- 0.008
r_d: 147.14 +- 0.29 Mpc
Chi2 (MAP): 1411.4 (2.17 sigma away from no v_bulk corrections)
Log evidence: -721.7 (Δ ln(Z) = 1.0 in favor of v_bulk corrections)
Degrees of freedom: 1598
"""

"""
Flat wCDM
M0: -19.417 +- 0.015 mag
H0: 67.81 +- 0.60 km/s/Mpc
Ωm: 0.298 +- 0.009
r_d: 147.14 +- 0.29 Mpc
w0: -0.914 +- 0.040 (prior ~ U(-4/3, -2/3))
Chi2 (MAP): 1411.5 (2.14 sigma away from ΛCDM)
Log evidence: -722.3 (Δ ln(Z) = 0.4 against ΛCDM)
Degrees of freedom: 1598
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
M0: -19.415 +- 0.014 mag
H0: 67.76 +0.61 -0.60 km/s/Mpc
Ωm: 0.304 +- 0.008
r_d: 147.14 +- 0.29 Mpc
w0: -0.881 +0.053 -0.052 (prior ~ U(-1.0, -1/3))
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
Chi2 (MAP): 1411.4 (2.17 sigma away from ΛCDM)
Log evidence: -722.0 (Δ ln(Z) = 0.7 against ΛCDM)
Degrees of freedom: 1598
"""

"""
Flat w0waCDM (w0 + wa < 0 enforced)
M0: -19.416 +- 0.015 mag
H0: 67.77 +0.63 -0.60 km/s/Mpc
Ωm: 0.304 +0.015 -0.023
r_d: 147.14 +- 0.29 Mpc
w0: -0.891 +0.063 -0.057 (prior ~ U(-1.5, -0.5))
wa: -0.17 +0.48 -0.45 (prior ~ U(-3.0, 2.5))
Chi2 (MAP): 1411.4 (1.67 sigma away from ΛCDM)
Log evidence: -723.9 (Δ ln(Z) = -1.2 in favour of ΛCDM)
Degrees of freedom: 1597
"""
