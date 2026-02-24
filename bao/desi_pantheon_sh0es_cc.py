from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data_shoes import get_data
from y2005cc.data import get_data as get_cc_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
legend, z_cmb, z_hel, mb_vals, ceph_dists, cov_matrix_sn = get_data()

ceph_mask = ceph_dists != -9
z_outflow_cut = 0.0061  # outflow effects start from here on and decays as ~1/z
flow_cut_mask = z_cmb < z_outflow_cut
local_ceph = ceph_mask & flow_cut_mask
z_cut_arr = np.full_like(z_cmb, z_outflow_cut)

"""
z_cut | Chi2 | Log(Z)
---------------------
0.0050 1498.9 -753.3
0.0055 1497.5 -752.8
0.0061 1496.6 -752.4
0.0065 1496.4 -752.4
0.0070 1495.7 -752.5
0.0075 1496.6 -752.6
0.0080 1496.5 -752.6
"""

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_cc = np.linalg.inv(cov_matrix_cc)
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

N_cc = len(z_cc_vals)

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
def v_outflow(v_100, z):
    return 100 * v_100 * (5 / np.log(10)) / (c * z)


@njit
def outflow_correction(theta):
    return np.where(
        local_ceph, v_outflow(theta[4], z_cut_arr), v_outflow(theta[4], z_cmb)
    )


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi2_sn(theta):
    mu_the = np.where(ceph_mask, ceph_dists, mu_theory(theta))
    mb_theory = mu_the + theta[0] + outflow_correction(theta)
    delta_sn = mb_vals - mb_theory
    return solve_triang(cho_sn, delta_sn)


@njit
def chi2_bao(theta):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, theta)
    return delta_bao @ inv_cov_bao @ delta_bao


@njit
def chi2_cc(theta):
    f_cc = theta[-1]
    H_cc_theory = H_z(z_cc_vals, theta)
    delta_cc = H_cc_vals - H_cc_theory
    return delta_cc @ inv_cov_cc @ delta_cc * f_cc**2


def chi_squared(theta):
    return chi2_sn(theta) + chi2_bao(theta) + chi2_cc(theta)


bounds = np.array(
    [
        (-20.0, -18.5),  # M (mag)
        (50.0, 100.0),  # H0 (km/s/Mpc)
        (0.2, 0.7),  # Ωm
        (120.0, 170.0),  # rd (Mpc)
        (-1.5, 3.5),  # v_outflow in units of 100 km/s
        (0.4, 2.5),  # f_cc
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(theta):
    f_cc = theta[-1]
    normalization_cc = -2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(theta) - 0.5 * normalization_cc


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
    from cosmic_chronometers.plot_predictions import plot_cc_predictions

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
        (f_cc_16, f_cc_50, f_cc_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 2] * (samples[:, 1] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    print(f"M0: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωm h^2: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}"
    )
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"v_flow: {v_f_50:.3f} +{(v_f_84 - v_f_50):.3f} -{(v_f_50 - v_f_16):.3f}")
    print(f"f_cc: {f_cc_50:.3f} +{(f_cc_84 - f_cc_50):.3f} -{(f_cc_50 - f_cc_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(samples[np.argmax(log_probs)]):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {len(bao_data) + len(z_cmb) + N_cc - len(best_fit)}")

    labels = ["$M_0$", "$H_0$", "$Ω_m$", "$r_{drag}$", "$v_{flow}$", "$f_{CC}$"]
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
        y=mb_vals - (M_50 + outflow_correction(best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_50,
        label=f"{legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
    )

if __name__ == "__main__":
    main()


"""
Flat ΛCDM
M0: -19.271 +- 0.027 mag
H0: 73.0 +- 0.9 km/s/Mpc
Ωm: 0.302 +0.008 -0.007
Ωm h^2: 0.161 +- 0.005
rd: 138.7 +- 1.9 Mpc
f_cc: 1.43 +- 0.17
Chi2 (MAP): 1502.6
Log evidence: -755.2
Degrees of freedom: 1701
"""

"""
Flat ΛCDM
Void outflow corrections of SNe M(z) = M_inf + v_corr
v_corr = 100 * v_flow * (5 / ln(10)) / (c * z_cmb) with v_flow in units 100 km/s

v_flow: 110 +- 37 km/s (prior ~ U(-150, 350))
M_inf: -19.367 +- 0.042 mag
M0 (computed at z=0.0061): -19.236 +- 0.085 mag
H0: 70.3 +-1.2 km/s/Mpc
Ωm: 0.300 +0.008 -0.007
Ωm h^2: 0.148 +- 0.006
rd: 144.2 +2.7 -2.6 Mpc
f_cc: 1.49 +0.18 -0.17
Chi2 (MAP): 1496.6 (2.8 sigma away from no flow corrections)
Log evidence: -752.4 (Δ logZ = 2.8 against no flow corrections)
Degrees of freedom: 1700
"""

"""
Flat wCDM
w0: -0.927 +- 0.039 (prior ~ U(-1.3, -0.6))
M0: -19.270 +- 0.028 mag
H0: 72.6 +1.0 -0.9 km/s/Mpc
Ωm: 0.296 +- 0.008
Ωm h^2: 0.156 +- 0.0060
rd: 137.9 +2.0 -1.9 Mpc
f_cc: 1.41 +0.17 -0.17
Chi2 (MAP): 1496.7 (2.4 sigma away from ΛCDM)
Log evidence: -755.4 (Δ logZ = -0.2 in favour of ΛCDM)
Degrees of freedom: 1700
"""

"""
Flat w0waCDM
TODO
"""
