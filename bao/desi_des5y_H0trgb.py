from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025DESdovekie.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

c = c0 / 1000  # Speed of light in km/s

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(cov_matrix_bao)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=2000)
dz = np.diff(z_grid)


@njit
def Ez(z, theta):
    Om, w0 = theta[3], theta[4]
    zp1 = 1.0 + z
    cubed = zp1**3
    rho_de = (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2
    return np.sqrt(Om * cubed + (1.0 - Om) * rho_de)


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
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, theta):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / theta[2]


@njit
def theory_mu(theta):
    dL = (1.0 + z_hel) * DM_z(z_cmb, theta)
    return theta[0] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    delta_sn = mu_values - theory_mu(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = np.dot(delta_bao, np.dot(inv_cov_bao, delta_bao))
    return chi_sn + chi_bao


bounds = np.array(
    [
        (-0.5, +0.5),  # ΔM
        (56.0, 85.0),  # H0
        (120.0, 160.0),  # r_d
        (0.1, 0.7),  # Ωm
        (-1.0, -1 / 3),  # w0
    ],
    dtype=np.float64,
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        # TRGB prior on H0 from Freedman et al. 2025
        H0_prior = -0.5 * ((theta[1] - 70.39) / 1.8) ** 2
        return normalization + H0_prior
    return -np.inf


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
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions
    from log_evidence import log_evidence

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.25),
        (emcee.moves.DEMove(), 0.75),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    [
        [dM_16, dM_50, dM_84],
        [H0_16, H0_50, H0_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(samples[np.argmax(log_probs)]):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degrees of freedom: {bao_data['value'].size + sn_size - len(best_fit)}")

    labels = ["$Δ_M$", "$H_0$", "$r_{drag}$", "$Ω_m$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
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


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
ΔM: 0.003 +0.055 -0.056 mag
H0: 70.35 +1.78 -1.79 km/s/Mpc
r_d: 143.38 +3.89 -3.65 Mpc
Ωm: 0.306 +0.008 -0.008
Chi2 (MAP): 1645.29
Log Evidence: -836.05
Degrees of freedom: 1723
"""

"""
Flat wCDM
ΔM: 0.024 +0.056 -0.058 mag
H0: 70.3 +1.8 -1.8 km/s/Mpc
r_d: 141.7 +3.9 -3.7 Mpc
Ωm: 0.297 +0.009 -0.009
w0: -0.908 +0.037 -0.038 (prior U(-1.5, -0.5))
Chi2 (MAP): 1639.51 (2.40 sigma away from ΛCDM)
Log Evidence: -835.54 (Δ logZ 0.51 against ΛCDM)
Degrees of freedom: 1722
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
ΔM: 0.031 +0.055 -0.058 mag
H0: 70.4 +1.8 -1.8 km/s/Mpc
r_d: 141.4 +3.9 -3.7 Mpc
Ωm: 0.305 +0.008 -0.008
w0: -0.861 +0.051 -0.052 (prior U(-1.0, -0.333))
wa: d w(z)/dz at z=0 = -(3/2) * (1 - w0^2)
Chi2 (MAP): 1638.47 (2.61 sigma away from ΛCDM)
Log Evidence: -834.32 (Δ logZ 1.34 against ΛCDM)
Degrees of freedom: 1722
"""

"""
Flat w(z) = w0 + wa * z / (1 + z)
TODO: re-run
w0: (prior U(-1.5, 0.0))
wa: (prior U(-3.0, 1.5))
Chi squared: 1638.13  (2.20 sigma away from ΛCDM)
"""
