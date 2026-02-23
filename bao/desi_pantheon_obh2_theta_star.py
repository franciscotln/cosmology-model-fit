from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_early_lcdm_compression as cmb
from interpolator import interp_hermite
from y2022pantheonSHOES.data import get_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mag_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(cov_matrix_bao)

"""
Planck compressed priors for θ* and ωb, omitting ωm = Ωm * h^2 (arXiv:2503.14738v2)
"""
cmb_compressed_priors = cmb.DISTANCE_PRIORS[[0, 1]]
cmb_inv_cov = np.linalg.inv(cmb.covariance[[0, 1], :][:, [0, 1]])

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0):
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, theta):
    H0, Obh2, Och2, w0 = theta[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


cmb.set_HZ(H_z)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


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
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, theta):
    Obh2, Och2 = theta[2], theta[3]
    Omh2 = Obh2 + Och2 + Omnuh2
    rd = cmb.r_drag(Obh2, Omh2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


@njit
def apparent_mag_theory(theta):
    dL = (1.0 + z_hel) * DM_z(z_cmb, theta)
    return theta[0] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return y @ y


def chi_squared(theta):
    cmb_observables = cmb.cmb_distances(theta[2], theta[3], theta)[[0, 1]]
    delta_cmb = cmb_compressed_priors - cmb_observables
    chi2_cmb = delta_cmb @ cmb_inv_cov @ delta_cmb

    delta_sn = mag_values - apparent_mag_theory(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao
    return chi_sn + chi_bao + chi2_cmb


bounds = np.array(
    [
        (-20.0, -19.0),  # M
        (50.0, 90.0),  # H0
        (0.0, 0.05),  # ωb = Ωb * h^2
        (0.05, 0.30),  # ωc = Ωc * h^2
        (-1.0, -1 / 3),  # w0
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
    from bao.plot_predictions import plot_bao_predictions
    from log_evidence import log_evidence

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 3500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
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
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    zd_samples = cmb.z_drag(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])
    zd_16, zd_50, zd_84 = np.percentile(zd_samples, [15.9, 50, 84.1])

    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.5f} +{(Och2_84 - Och2_50):.5f} -{(Och2_50 - Och2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z_d: {zd_50:.2f} +{(zd_84 - zd_50):.2f} -{(zd_50 - zd_16):.2f}")
    print(f"r_d: {cmb.rs_z(zd_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {2 + len(bao_data['z']) + len(z_cmb) - len(best_fit)}")

    labels = ["$M$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
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
        y=mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag_theory(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
DESI DR2
Pantheon+
(θ*, ωb = Ωb x h^2)CMB - Early time ΛCDM

Priors:
U(-20.0, -19.0),  # M
U(50.0, 90.0),  # H0
U(0.010, 0.030),  # ωb = Ωb * h^2
U(0.05, 0.30),  # ωc = Ωc * h^2

wCDM:
U(-1.5, -0.5),  # w0

w0waCDM: (w0 + wa < 0 enforced)
U(-1.5, 0.0),  # w0
U(-2.0, 1.5),  # wa

wzCDM:
U(-1.0, -1/3),  # w0

V_pec:
U(-1.30, 3.15),  # vpec (in units of 100 km/s)
"""

"""
Flat ΛCDM  w(z) = -1
M: -19.412 +0.009 -0.009 mag
H0: 68.41 +0.30 -0.30 km/s/Mpc
Ωm: 0.298 +0.004 -0.004
ωm: 0.1395 +0.0009 -0.0008
ωb: 0.02223 +0.00015 -0.00015
ωc: 0.11663 +0.00082 -0.00081
z_d: 1059.45 +0.36 -0.36
r_d: 148.15 Mpc
Chi squared: 1416.94
Log Evidence: -728.18
Degs of freedom: 1601

===============================

Flat ΛCDM
Outflow corrections of SNe M(z) = M_inf + v_flow_corr
v_flow_corr = 100 * v_flow * (5 / np.log(10)) / (c * z_cmb) with v_flow in units 100 km/s

M_inf: -19.423 +0.010 -0.010 mag
v_flow: 99 +42 -43 km/s
H0: 68.48 +0.30 -0.30 km/s/Mpc
Ωm: 0.297 +0.004 -0.004
ωm: 0.1393 +0.0009 -0.0008
ωb: 0.02223 +0.00015 -0.00015
ωc: 0.11640 +0.00083 -0.00081
z_d: 1059.45 +0.35 -0.36
r_d: 148.22 Mpc
Chi squared: 1411.56 (2.32 sigma away from no v_flow corrections)
Log Evidence: -726.92 (Δ ln(Z) = 1.26 in favor of v_flow corrections)
Degs of freedom: 1600

===============================

Flat wCDM w(z) = w0
M: -19.438 +0.015 -0.015 mag
H0: 67.26 +0.63 -0.62 km/s/Mpc
Ωm: 0.304 +0.005 -0.005
ωm: 0.1376 +0.0013 -0.0013
ωb: 0.02223 +0.00015 -0.00015
ωc: 0.11472 +0.00125 -0.00127
w0: -0.943 +0.027 -0.028
z_d: 1059.34 +0.36 -0.36
r_d: 148.66 Mpc
Chi squared: 1412.51 (2.10 sigma away from ΛCDM)
Log Evidence: -728.69
Degs of freedom: 1600

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
M: -19.434 +0.013 -0.013 mag
H0: 67.21 +0.58 -0.59 km/s/Mpc
Ωm: 0.307 +0.006 -0.006
ωm: 0.1385 +0.0009 -0.0009
ωb: 0.02223 +0.00015 -0.00015
ωc: 0.11563 +0.00092 -0.00092
w0: -0.893 +0.045 -0.045
wa: d w(z)/dz at z=0 = -(3 / 2) * (1 - w0^2)
z_d: 1059.40 +0.35 -0.36
r_d: 148.41 Mpc
Chi squared: 1411.65 (2.30 sigma away from ΛCDM)
Log Evidence: -727.33

===============================

Flat w(z) = w0 + wa * z / (1 + z)
M: -19.430 +0.016 -0.017 mag
H0: 67.30 +0.62 -0.62 km/s/Mpc
Ωm: 0.308 +0.006 -0.006
ωm: 0.1396 +0.0018 -0.0021
ωb: 0.02223 +0.00015 -0.00015
ωc: 0.11670 +0.00184 -0.00207
w0: -0.882 +0.060 -0.058
wa: -0.306 +0.261 -0.277
z_d: 1059.53 +0.37 -0.38
r_d: 148.12 Mpc
Chi squared: 1411.33 (1.88 sigma away from ΛCDM)
Log Evidence: -728.98
Degs of freedom: 1599
"""
