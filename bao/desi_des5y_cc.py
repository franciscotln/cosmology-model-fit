from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from y2025DESdovekie.data import (
    effective_sample_size as sn_sample,
    get_data as get_sn_data,
)

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, z_sn_hel_vals, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(cov_matrix_bao)
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = c0 / 1000  # km/s

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def rho_de(z, w0):
    cubed = (1.0 + z) ** 3
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2  # wzCDM
    # return 1.0  # ΛCDM
    # return cubed ** (1.0 + w0)  # wCDM
    # return cubed ** (1.0 + w0 + wa) * np.exp(-3 * wa * z / (1.0 + z))  # w0waCDM


@njit
def Ez(z, theta):
    Om, w0 = theta[4], theta[5]
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * rho_de(z, w0))


@njit
def H_z(z, theta):
    return theta[2] * Ez(z, theta)


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
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


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
    dL = (1.0 + z_sn_hel_vals) * DM_z(z_sn_vals, theta)
    return theta[1] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    # much faster than direct inversion for large matrices
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return y @ y


def chi_squared(theta):
    delta_sn = mu_values - mu_theory(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_cc = H_cc_vals - H_z(z_cc_vals, theta)
    chi_cc = theta[0] ** 2 * solve_triang(cho_cc, delta_cc)

    return chi_sn + chi_bao + chi_cc


bounds = np.array(
    [
        (0.5, 2.5),  # f_cc: CC error rescaling (overestimated)
        (-0.55, 0.55),  # ΔM: magnitude offset
        (50.0, 80.0),  # H0: Hubble constant at present
        (110.0, 175.0),  # r_d: sound horizon at drag epoch
        (0.2, 0.7),  # Ωm: matter density parameter at present
        (-1.0, -1 / 3),  # w0: dark energy equation of state at present
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(theta):
    f_cc = theta[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(theta) - 0.5 * normalization_cc


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def main():
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from cosmic_chronometers.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions
    from log_evidence import log_evidence

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2500 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

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
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [dM_16, dM_50, dM_84],
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    deg_of_freedom = sn_sample + bao_data["value"].size + z_cc_vals.size - ndim

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    labels = ["$f_{CCH}$", "ΔM", "$H_0$", "$r_d$", "$Ω_m$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=f"{bao_legend}: $r_d$={rd_50:.1f} Mpc",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_50,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
ΔM: -0.057 +0.070 -0.072 mag
H0: 68.4 +2.2 -2.3 km/s/Mpc
r_d: 147.4 +4.9 -4.6 Mpc
Ωm: 0.307 +0.008 -0.007
f_cc: 1.48 +0.18 -0.17
Chi squared: 1680.55
Log evidence: -979.53
Degrees of freedom: 1758
"""

"""
Flat ΛCDM
Corrections to absolute mag of SNe M(z) = M0 + M'0 * z / (1 + (z / z_c))
where z_c = 0.0395

ΔM: 0.027 +0.077 -0.078 mag
M'0: -2.14 +0.84 -0.81 mag / unity redshift (prior ~ U(-7, 3))
H0: 69.0 +2.3 -2.2 km/s/Mpc
r_d: 147.1 +4.9 -4.6 Mpc
Ωm: 0.298 +0.008 -0.008
f_cc: 1.48 +0.18 -0.17
Chi squared: 1673.57 (2.64 sigma away from constant M)
Log evidence: -977.77 (Δ logZ = 1.76 against constant M)
Degrees of freedom: 1757
"""

"""
Flat wCDM: w(z) = w0
ΔM: -0.060 +0.070 -0.073 mag
H0: 67.7 +2.3 -2.3 km/s/Mpc
r_d: 147.3 +5.0 -4.6 Mpc
Ωm: 0.298 +0.009 -0.008
f_cc: 1.48 +0.18 -0.17
w0: -0.911 +0.037 -0.038 (prior ~ U(-1.5, -0.5))
Chi squared: 1674.85 (2.39 sigma away from ΛCDM)
Log evidence: -979.18 (Δ logZ = 0.35 against ΛCDM)
Degrees of freedom: 1757
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
ΔM: -0.058 +0.070 -0.072 mag
H0: 67.6 +2.2 -2.2 km/s/Mpc
r_d: 147.2 +4.9 -4.6 Mpc
Ωm: 0.306 +0.008 -0.007
f_cc: 1.48 +0.18 -0.17
w0: -0.866 +0.051 -0.052 (prior ~ U(-1.0, -1/3))
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
Chi squared: 1673.83 (2.59 sigma away from ΛCDM)
Log evidence: -978.04 (Δ logZ = 1.49 against ΛCDM)
Degrees of freedom: 1757
"""

"""
Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
ΔM: -0.056 +0.071 -0.072 mag
H0: 67.6 +2.3 -2.2 km/s/Mpc
r_d: 147.2 +4.9 -4.7 Mpc
Ωm: 0.312 +0.014 -0.019
f_cc: 1.47 +0.18 -0.17
w0: -0.859 +0.071 -0.064
wa: -0.424 +0.483 -0.472
Chi squared: 1673.50 (1.9 sigma away from ΛCDM)
Log evidence: -980.78 (Δ logZ = -1.25 in favour of ΛCDM)
Degrees of freedom: 1756
"""
