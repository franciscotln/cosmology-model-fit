from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data_shoes import get_data
import cmb.data_planck_act_compression as cmb

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
legend, z_cmb, z_hel, mb_vals, ceph_dists, cov_matrix_sn = get_data(z_cut_ceph=0.0044)
# At z_ceph < 0.0044 it seems the outflow velocity drops to zero or even becomes slightly negative.

ceph_mask = ceph_dists != -9

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

c = cmb.c  # km/s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

# z grid for DM
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
def v_outflow(v_100):
    return 100 * v_100 * (5 / np.log(10)) / (c * z_cmb)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi2_sn(theta):
    mu_the = np.where(ceph_mask, ceph_dists, mu_theory(theta))
    mb_theory = mu_the + theta[0] + v_outflow(theta[4])
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
    print(f"v_bulk: {v_b_50:.3f} +{(v_b_84 - v_b_50):.3f} -{(v_b_50 - v_b_16):.3f}")
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
        y=mb_vals - M_50 - v_outflow(v_b_50),
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
M0: -19.401 +- 0.008 mag
H0: 68.71 +- 0.27 km/s/Mpc
Ωm: 0.297 +- 0.003
Ωbh2: 0.02263 +- 0.00010
Ωch2: 0.11682 +- 0.00063
Ωmh2: 0.1401 +- 0.0006
rd: 147.66 +0.19 -0.19 Mpc
Chi2 (MAP): 1482.0
Log evidence: -761.7
Degrees of freedom: 1648
"""

"""
Flat ΛCDM
Bulk v_bulk corrections of SNe M(z) = M0 + v_bulk_corr
v_bulk_corr = 100 * v_bulk * (5 / np.log(10)) / (c * z_cmb) with v_bulk in units 100 km/s

v_bulk: 156 +27 - 28 km/s (prior ~ U(-1.0, 3.5) in units of 100 km/s)
M0: -19.427 +- 0.009 mag
H0: 68.54 +- 0.26 km/s/Mpc
Ωm: 0.299 +- 0.003
Ωbh2: 0.02259 +- 0.00010
Ωch2: 0.11713 +- 0.00063
Ωmh2: 0.1404 +- 0.0006
rd: 147.61 +- 0.19 Mpc
Chi2 (MAP): 1449.4 (5.71 sigma away from no bulk flow)
Log evidence: -747.3 (Δ logZ = 14.4 in favor of bulk flow)
Degrees of freedom: 1647
"""

"""
Flat wCDM
M0: -19.390 +- 0.013 mag
H0: 69.19 +- 0.54 km/s/Mpc
w0: -1.02 +- 0.02 (prior ~ U(-1.3, -0.6))
Ωm: 0.294 +- 0.005
Ωbh2: 0.02260 +- 0.00010
Ωch2: 0.1173 +- 0.0008
Ωmh2: 0.1406 +- 0.0008
rd: 147.55 +- 0.22 Mpc
Chi2 (MAP): 1481.0 (1 sigma away from ΛCDM)
Log evidence: -763.8 (Δ logZ = -2.1 in favour of ΛCDM)
Degrees of freedom: 1649
"""


"""
********************************************
Pantheon + SH0ES (z_cut 0.0041) + BAO + θ* CMB
********************************************
"""

"""
Flat ΛCDM
M0: -19.286 +- 0.030 mag
H0: 72.7 +- 1.0 km/s/Mpc
Ωm: 0.282 +- 0.004
Ωbh2: 0.0284 +- 0.0016
Ωch2: 0.1197 +- 0.0016
Ωmh2: 0.149 +- 0.003
rd: 140.95 +- 1.83 Mpc
Chi2 (MAP): 1461.9
Log evidence: -748.5
Degrees of freedom: 1648
"""

"""
Flat ΛCDM
Void outflow corrections of SNe M(z) = M0 + v_corr
v_corr = 100 * v_flow * (5 / np.log(10)) / (c * z_cmb) with v_flow in units 100 km/s

v_flow: 138 +- 33 km/s (prior ~ U(-1.0, 3.5) in units of 100 km/s)
M0: -19.390 +- 0.037 mag
H0: 69.7 +- 1.2 km/s/Mpc
Ωm: 0.291 +- 0.005
Ωbh2: 0.0240 +- 0.0017
Ωch2: 0.117 +- 0.001
Ωmh2: 0.142 +- 0.003
rd: 146.2 +2.0 -2.1 Mpc
Chi2 (MAP): 1444.4 (4.18 sigma away from no outflow)
Log evidence: -741.6 (Δ logZ = 6.9 in favor of outflow)
Degrees of freedom: 1647
"""

"""
Flat wCDM
M0: -19.241 +- 0.032 mag
H0: 73.4 +- 1.1 km/s/Mpc
w0: -0.874 +- 0.032 (prior ~ U(-1.3, -0.6))
Ωm: 0.285 +- 0.004
Ωbh2: 0.0342 +- 0.0023
Ωch2: 0.1188 +- 0.0018
Ωmh2: 0.1536 +- 0.0035
rd: 135.79 +- 2.24 Mpc
Chi2 (MAP): 1446.9 (3.87 sigma away from ΛCDM)
Log evidence: -743.2 (Δ logZ = 5.3 in favor of wCDM)
Degrees of freedom: 1647
"""
