from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_desi_compression as cmb
from y2024DES.data import effective_sample_size as sn_sample, get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, z_sn_hel_vals, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = c0 / 1000  # km/s
Orh2 = cmb.Omega_r_h2()

# arXiv:1807.06209v4 (Planck 2018 final)
theta_starx100 = 1.04110
theta_starx100_err = 0.00031

z_max = max(np.max(z_sn_vals), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=1200)
dx = np.diff(z_grid)


@njit
def Ez(z, theta):
    h, Om, w0 = theta[2] / 100, theta[4], theta[5]
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    Or = Orh2 / h**2
    Ode = 1 - Om - Or
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


@njit
def DM(theta):
    dh_grid = DH_z(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return cum_dm


@njit
def mu_theory(theta):
    dL = (1 + z_sn_hel_vals) * np.interp(z_sn_vals, z_grid, DM(theta))
    return theta[1] + 25 + 5 * np.log10(dL)


@njit
def H_z(z, theta):
    return theta[2] * Ez(z, theta)


@njit
def DH_z(z, theta):
    return c / H_z(z, theta)


@njit
def DM_z(z, theta):
    return np.interp(z, z_grid, DM(theta))


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, theta):
    H0, Obh2, Om = theta[2], theta[3], theta[4]
    rd = cmb.r_drag(wb=Obh2, wm=Om * (H0 / 100) ** 2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


def theta_starx100_theory(theta):
    H0, Obh2, Om = theta[2], theta[3], theta[4]
    z_star = cmb.z_star(wb=Obh2, wm=Om * (H0 / 100) ** 2)
    rs_star = cmb.rs_z(Ez, z_star, theta, H0, Obh2)
    DA_star = cmb.DA_z(Ez, z_star, theta, H0)
    return 100 * rs_star / ((1 + z_star) * DA_star)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    delta_theta_star = theta_starx100 - theta_starx100_theory(theta)
    chi_theta_star = (delta_theta_star / theta_starx100_err) ** 2

    delta_sn = mu_values - mu_theory(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = solve_triang(cho_bao, delta_bao)

    delta_cc = H_cc_vals - H_z(z_cc_vals, theta)
    chi_cc = solve_triang(cho_cc, delta_cc) * theta[0] ** 2

    return chi_theta_star + chi_sn + chi_bao + chi_cc


bounds = np.array(
    [
        (0.5, 2.5),  # f_cc: CC error rescaling (overestimated)
        (-0.55, 0.55),  # ΔM: magnitude offset
        (50.0, 85.0),  # H0: Hubble constant at present
        (0.005, 0.035),  # Ωb x h^2: baryon density parameter
        (0.1, 0.7),  # Ωm: matter density parameter at present
        (-1.5, 0.0),  # w0: dark energy equation of state at present
    ],
    dtype=np.float64,
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
    from .plot_predictions import plot_bao_predictions
    from log_evidence import log_evidence

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
    ]

    with Pool(8) as pool:
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
        [wb_16, wb_50, wb_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    deg_of_freedom = 1 + sn_sample + bao_data["value"].size + z_cc_vals.size - ndim

    r_d_samples = cmb.r_drag(samples[:, 3], samples[:, 4] * (samples[:, 2] / 100) ** 2)
    r_d_16, r_d_50, r_d_84 = np.percentile(r_d_samples, [15.9, 50, 84.1])

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"Ωm h^2: {wb_50:.5f} +{(wb_84 - wb_50):.5f} -{(wb_50 - wb_16):.5f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {r_d_50:.2f} +{(r_d_84 - r_d_50):.2f} -{(r_d_50 - r_d_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evidence(samples, log_probs, log_probability):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
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
        label=rf"Best fit: $H_0$={h0_50:.1f} km/s/Mpc, $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$f_{CCH}$", "ΔM", "$H_0$", "$ω_b$", "$Ω_m$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
f_cc: 1.48 +0.18 -0.17
H0: 66.6 +1.5 -1.3 km/s/Mpc
Ωm h^2: 0.01979 +0.00210 -0.00168 Mpc
Ωm: 0.307 +0.008 -0.007
r_d: 151.12 +2.34 -2.75 Mpc
Chi squared: 1692.55
Log evidence: -978.57
Degrees of freedom: 1777

===============================

Flat wCDM: w(z) = w0
f_cc: 1.47 +0.18 -0.18
H0: 68.4 +1.9 -1.6 km/s/Mpc
Ωm h^2: 0.02575 +0.00340 -0.00299 Mpc
Ωm: 0.303 +0.006 -0.005
w0: -0.894 +0.028 -0.028 (prior width 1.5: -1.5 to 0.0)
r_d: 144.71 +3.44 -3.73 Mpc
Chi squared: 1681.36
Log evidence: -975.69 (Δ logZ = 2.88 over ΛCDM)
Degrees of freedom: 1776

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 1.47 +0.18 -0.18
H0: 67.9 +1.9 -1.6 km/s/Mpc
Ωm h^2: 0.02477 +0.00332 -0.00276 Mpc
Ωm: 0.309 +0.007 -0.006
w0: -0.843 +0.040 -0.041 (prior width 1.5: -1.5 to 0.0)
r_d: 145.38 +3.34 -3.75 Mpc
Chi squared: 1679.53
Log evidence: -974.24 (Δ logZ = 4.33 over ΛCDM)
Degrees of freedom: 1776

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
f_cc: 1.47 +0.18 -0.18
ΔM: -0.065 +0.063 -0.053 mag
H0: 66.8 +2.1 -1.7 km/s/Mpc
Ωm h^2: 0.02255 +0.00386 -0.00312 Mpc
Ωm: 0.315 +0.010 -0.009
w0: -0.802 +0.066 -0.061 (prior width 1.5: -1.5 to 0.0)
wa: -0.546 +0.328 -0.368 (prior width 5.0: -3.0 to 2.0)
r_d: 147.50 +3.68 -4.23 Mpc
Chi squared: 1678.85
Log evidence: -975.91 (Δ logZ = 2.66 over ΛCDM)
Degrees of freedom: 1775
"""
