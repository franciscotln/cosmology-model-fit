from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data_shoes import get_data
import cmb.data_planck_act_compression as cmb

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
legend, z_cmb, z_hel, mb_vals, ceph_dists, cov_matrix_sn = get_data()

ceph_mask = ceph_dists != -9
z_outflow_cut = 0.0061  # Outflow effects start from here on and decays as ~1/z
flow_cut_mask = z_cmb < z_outflow_cut
local_ceph = ceph_mask & flow_cut_mask
z_cut_arr = np.full_like(z_cmb, z_outflow_cut)


"""
z_cut | Chi2 | Log(Z)
---------------------
0.0040 1472.0 -758.8
0.0045 1470.0 -757.8
0.0050 1468.5 -756.9
0.0055 1467.4 -756.4
0.0061 1466.6 -755.9
0.0065 1466.7 -755.9
0.0070 1467.5 -756.3
0.0075 1468.2 -756.6
0.0080 1468.8 -756.9
0.0085 1469.5 -757.2
"""


cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

c = cmb.c  # km/s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    return (1.0 + z) ** (3 * (1.0 + w0))  # wCDM


@njit
def Ez(z, H0, Obh2, Och2):
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0 = params[1]
    return H0 * Ez(z, H0=H0, Obh2=params[2], Och2=params[3])


cmb.set_HZ(H_z)


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
    Obh2, Och2 = theta[2], theta[3]
    Omh2 = Obh2 + Och2 + Omnu_h2
    rd = cmb.r_drag(wb=Obh2, wm=Omh2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


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


def chi2_cmb(theta):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(theta[2], theta[3], theta)
    return delta_cmb @ cmb.inv_cov_mat @ delta_cmb


@njit
def chi2_bao(theta):
    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], desi_qty, theta)
    return delta_bao @ inv_cov_bao @ delta_bao


def chi_squared(theta):
    return chi2_sn(theta) + chi2_bao(theta) + chi2_cmb(theta)


bounds = np.array(
    [
        (-20.0, -18.5),  # M (mag)
        (50.0, 100.0),  # H0 (km/s/Mpc)
        (0.01, 0.05),  # Ωbh2
        (0.01, 0.25),  # Ωch2
        (-1.0, 3.5),  # v_flow in units of 100 km/s
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
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (v_b_16, v_b_50, v_b_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnu_h2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    rd_samples = cmb.r_drag(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1])

    print(f"M0: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"Ωbh2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"Ωch2: {Och2_50:.5f} +{(Och2_84 - Och2_50):.5f} -{(Och2_50 - Och2_16):.5f}")
    print(f"Ωmh2: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"v_flow: {v_b_50:.3f} +{(v_b_84 - v_b_50):.3f} -{(v_b_50 - v_b_16):.3f}")
    print(f"Chi2 (MAP): {chi_squared(samples[np.argmax(log_probs)]):.1f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {len(bao_data) + len(z_cmb) - len(best_fit)}")

    labels = ["$M_0$", "$H_0$", "$Ω_b h^2$", "$Ω_c h^2$", "$v_{flow}$"]
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


if __name__ == "__main__":
    main()

"""
********************************************
Pantheon + SH0ES (z_cut 0.0041) + BAO + CMB compressed
********************************************
"""

"""
Flat ΛCDM
M0: -19.400 +- 0.008 mag
H0: 68.73 +- 0.26 km/s/Mpc
Ωm: 0.296 +- 0.003
Ωbh2: 0.02264 +- 0.00010
Ωch2: 0.11676 +- 0.00063
Ωmh2: 0.1400 +- 0.0006
rd: 147.67 +- 0.19 Mpc
Chi2 (MAP): 1498.4
Log evidence: -769.9
Degrees of freedom: 1666
"""

"""
Flat ΛCDM
Void outflow corrections of SNe M(z) = M_inf + v_corr
v_corr = 100 * v_flow * (5 / np.log(10)) / (c * z_cmb) with v_flow in units 100 km/s

v_flow: 148 +- 26 km/s (prior ~ U(-1.0, 3.5) in units of 100 km/s)
M_inf: -19.426 +- 0.009 mag
M0 (computed at z_cut=0.0061): -19.251 +- 0.040 mag
H0: 68.54 +- 0.26 km/s/Mpc
Ωm: 0.299 +0.004 -0.003
Ωbh2: 0.02259 +- 0.00010
Ωch2: 0.11715 +- 0.00063
Ωmh2: 0.1404 +- 0.0006
rd: 147.61 +- 0.19 Mpc
Chi2 (MAP): 1466.6 (5.64 sigma away from no outflow case)
Log evidence: -755.9 (Δ logZ = 14.0 in favour of outflow model)
Degrees of freedom: 1665
"""

"""
Flat wCDM
w0: -1.025 +- 0.022 (prior ~ U(-1.3, -0.6))
M0: -19.388 +- 0.013 mag
H0: 69.28 +- 0.53 km/s/Mpc
Ωm: 0.293 +- 0.004
Ωbh2: 0.02261 +- 0.00010
Ωch2: 0.11735 +- 0.00080
Ωmh2: 0.1406 +- 0.0008
rd: 147.54 +0.22 -0.21 Mpc
Chi2 (MAP): 1497.1 (1.147 sigma away from ΛCDM)
Log evidence: -771.8 (Δ logZ = -1.9 in favor of ΛCDM)
Degrees of freedom: 1665
"""
