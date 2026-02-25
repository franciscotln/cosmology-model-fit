from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2022pantheonSHOES.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

c = c0 / 1000  # Speed of light in km/s

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
legend, z_cmb, z_hel, apparent_mag_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]
cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    cubed = (1.0 + z) ** 3
    return (2 * cubed / ((1.0 + w0) + (1.0 - w0) * cubed)) ** 2


@njit
def Ez(z, params):
    Om = params[3]
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
desi_qty = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


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


@njit
def outflow_correction(params):
    # up to second order in z
    Om, v_100 = params[3], params[4]
    q0 = 1.5 * Om - 1.0
    q_term = (1.0 - q0) * z_cmb
    q_corr = (1.0 + q_term) / (1.0 + 0.5 * q_term)
    v_ratio = 100 * v_100 / (c * z_cmb)
    return v_ratio * (5 / np.log(10)) * q_corr


@njit
def sn_apparent_mag(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[1] + outflow_correction(params) + 25.0 + 5 * np.log10(dL)


bounds = np.array(
    [
        (40.0, 90.0),  # H0
        (-20.0, -19.0),  # M
        (115.0, 170.0),  # r_d
        (0.0, 1.0),  # Ωm
        (-1.3, 3.15),  # v_flow in units of 100 km/s
        (0.4, 2.5),  # f_cc
    ]
)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_sn = apparent_mag_values - sn_apparent_mag(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, params)
    chi_bao = solve_triang(cho_bao, delta_bao)

    f_cc = params[-1]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = solve_triang(cho_cc, delta_cc) * f_cc**2

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
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(6) as pool:
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

    print("Gelman-Rubin R-hat:", gelman_rubin(chains_samples))

    [
        (h0_16, h0_50, h0_84),
        (M_16, M_50, M_84),
        (rd_16, rd_50, rd_84),
        (Om_16, Om_50, Om_84),
        (vf_16, vf_50, vf_84),
        (f_cc_16, f_cc_50, f_cc_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    deg_of_freedom = len(z_cmb) + len(bao_data) + N_cc - len(best_fit)

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"H0: {h0_50:.2f} +{(h0_84 - h0_50):.2f} -{(h0_50 - h0_16):.2f} km/s/Mpc")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"v_flow (x 100 km/s): {vf_50:.3f} +{(vf_84 - vf_50):.3f} -{(vf_50 - vf_16):.3f}"
    )
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    labels = ["$H_0$", "M", "$r_d$", "Ωm", "$v_{flow}$", "$f_{CC}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=legend,
        x=z_cmb,
        y=apparent_mag_values - M_50,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=sn_apparent_mag(best_fit) - M_50,
        label=f"$\Omega_m$={Om_50:.3f}, $H_0$={h0_50:.2f} km/s/Mpc",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
M: -19.406 +0.070 -0.072 mag
H0: 68.5 +2.3 -2.3 km/s/Mpc
r_d: 147.3 +4.9 -4.6 Mpc
Ωm: 0.305 +0.008 -0.008
f_cc: 1.48 +0.18 -0.17
Chi squared: 1451.44
Log Evidence: -866.2
Degrees of freedom: 1634

===============================

Flat ΛCDM: w(z) = -1
Void outflow corrections of SNe M(z) = Minf + v_flow_corr
v_flow_corr = 100 * v_flow * (5 / ln(10)) / (c * z_cmb) with v_flow in units 100 km/s

v_flow: 93 +44 -43 km/s (prior ~ U(-1.30, 3.15))
M_inf: -19.411 +0.070 -0.073 mag
H0: 68.8 +2.3 -2.3 km/s/Mpc
r_d: 147.1 +4.9 -4.6 Mpc
Ωm: 0.301 +0.008 -0.008
f_cc: 1.48 +0.18 -0.17
Chi squared: 1446.60 (2.0 sigma away from no flow velocity correction)
Log Evidence: -865.3
Degrees of freedom: 1633

===============================

Flat wCDM: w(z) = w0
M: -19.421 +0.070 -0.073 mag
H0: 67.7 +2.3 -2.3 km/s/Mpc
r_d: 147.3 +4.9 -4.6 Mpc
Ωm: 0.299 +0.009 -0.009
f_cc: 1.48 +0.18 -0.17
w0: -0.916 +0.040 -0.039 (prior ~ U(-1.5, -0.5))
Chi squared: 1446.82
Log Evidence: -866.3
Degrees of freedom: 1633

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
M: -19.417 +0.069 -0.073 mag
H0: 67.7 +2.2 -2.3 km/s/Mpc
r_d: 147.2 +4.9 -4.6 Mpc
Ωm: 0.306 +0.008 -0.008
f_cc: 1.48 +0.18 -0.17
w0: -0.884 +0.053 -0.052 (prior ~ U(-1.0, 0.0))
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
Chi squared: 1446.78
Log Evidence: -866.0
Degrees of freedom: 1633

===============================

Flat w0waCDM:
TODO
"""
