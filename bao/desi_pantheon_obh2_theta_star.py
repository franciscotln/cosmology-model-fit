from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_desi_compression as cmb
from y2022pantheonSHOES.data import get_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Omega_r_h2(2.044)
Omnuh2 = cmb.Omnu_h2
z_nr = cmb.z_nr

sn_legend, z_cmb, z_hel, mag_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_bao = cho_factor(cov_matrix_bao, lower=True)[0]

"""
Planck compressed priors for θ* and ωb, without ωm = Ωm * h^2 (arXiv:2503.14738v2)
This way we allow for the ratio ωb / ωm to vary freely independently from Planck.
"""
cmb_compressed_priors = cmb.DISTANCE_PRIORS[[0, 1]]
cmb_inv_cov = np.linalg.inv(cmb.covariance[[0, 1], :][:, [0, 1]])

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=2000)
dx = np.diff(z_grid)


@njit
def Omnu_z(z):
    """
    Computes the appox. evolution of one massive
    neutrino species energy density with redshift
    """
    return (
        (1 + z) ** 4
        * (1 + ((1 + z_nr) / (1 + z)) ** 2) ** 0.5
        * (1 + (1 + z_nr) ** 2) ** -0.5
    )


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * Omnu_z(z)
    dark_energy_term = Ode * (4 * zp1**3 / (1 + 3 * zp1**3)) ** (4 * (1 + w0))

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, theta):
    H0, Obh2, Och2, w0 = theta[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


def bao_theory(z, qty, theta):
    Obh2, Och2 = theta[2], theta[3]
    Omh2 = Obh2 + Och2 + Omnuh2
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
def theory_apparent_mag(theta):
    dL = (1 + z_hel) * DM_z(z_cmb, theta)
    return theta[0] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(theta):
    cmb_observables = cmb.cmb_distances(Ez, *theta[1:])[[0, 1]]
    delta_cmb = cmb_compressed_priors - cmb_observables
    chi2_cmb = delta_cmb @ cmb_inv_cov @ delta_cmb

    delta_sn = mag_values - theory_apparent_mag(theta)
    chi_sn = solve_triang(cho_sn, delta_sn)

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = solve_triang(cho_bao, delta_bao)
    return chi_sn + chi_bao + chi2_cmb


bounds = np.array(
    [
        (-20.0, -19.0),  # M
        (50.0, 90.0),  # H0
        (0.0, 0.05),  # ωb = Ωb * h^2
        (0.05, 0.30),  # ωc = Ωc * h^2
        (-1.5, 0.0),  # w0
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
    from .plot_predictions import plot_bao_predictions
    from log_evidence import log_evidence

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 350
    nsteps = 3500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

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
    print(f"r_d: {cmb.rs_z(Ez, zd_50, *best_fit[1:]):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {2 + len(bao_data['z']) + len(z_cmb) - len(best_fit)}")

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
        y_model=theory_apparent_mag(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$M$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
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
U(-1.5, 0.0),  # w0
U(-2.0, 1.5),  # wa
"""

"""
Flat ΛCDM  w(z) = -1
M: -19.412 +0.009 -0.009 mag
H0: 68.40 +0.30 -0.30 km/s/Mpc
Ωm: 0.298 +0.004 -0.004
ωm: 0.1395 +0.0008 -0.0008
ωb: 0.02222 +0.00015 -0.00015
ωc: 0.11661 +0.00082 -0.00081
w0: -1
wa: 0
z_d: 1059.52 +0.36 -0.35
r_d: 148.15 Mpc
Chi squared: 1416.96
Log Evidence: -727.27
Degs of freedom: 1601

===============================

Flat wCDM w(z) = w0
M: -19.438 +0.015 -0.015 mag
H0: 67.26 +0.62 -0.62 km/s/Mpc
Ωm: 0.304 +0.005 -0.005
ωm: 0.1376 +0.0013 -0.0013
ωb: 0.02223 +0.00015 -0.00015
ωc: 0.11468 +0.00125 -0.00126
w0: -0.943 +0.027 -0.027 (prior width 1.5: -1.5 to 0.0)
wa: 0
z_d: 1059.40 +0.36 -0.36
r_d: 148.67 Mpc
Chi squared: 1412.50
Log Evidence: -728.18
Degs of freedom: 1600

===============================

Flat w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)
M: -19.435 +0.013 -0.013 mag
H0: 67.20 +0.60 -0.59 km/s/Mpc
Ωm: 0.307 +0.006 -0.006
ωm: 0.1384 +0.0010 -0.0010
ωb: 0.02223 +0.00014 -0.00015
ωc: 0.11556 +0.00093 -0.00094
w0: -0.900 +0.044 -0.044 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9 / 4) * (1 + w0)
z_d: 1059.46 +0.35 -0.35
r_d: 148.43 Mpc
Chi squared: 1411.63
Log Evidence: -727.28
Degs of freedom: 1600

===============================

Flat w(z) = w0 + wa * z / (1 + z)
M: -19.430 +0.016 -0.017 mag
H0: 67.30 +0.62 -0.62 km/s/Mpc
Ωm: 0.308 +0.006 -0.006
ωm: 0.1396 +0.0018 -0.0021
ωb: 0.02223 +0.00015 -0.00015
ωc: 0.11670 +0.00184 -0.00207
w0: -0.882 +0.060 -0.058 (prior width 1.5: -1.5 to 0.0)
wa: -0.306 +0.261 -0.277 (prior width 3.5: -2.0 to 1.5)
z_d: 1059.53 +0.37 -0.38
r_d: 148.12 Mpc
Chi squared: 1411.33
Log Evidence: -728.98
Degs of freedom: 1599
"""

"""
DESI DR2
Pantheon+
(θ*, ωm = Ωm x h^2)CMB - Early time ΛCDM

Priors:
U(-20.0, -19.0)  # M
U(50.0, 90.0)  # H0
U(0.0, 0.05)  # ωb = Ωb * h^2
U(0.05, 0.30)  # ωc = Ωc * h^2
U(-1.5, 0.0)  # w0
U(-3.0, 3.0)  # wa
"""

"""
Flat ΛCDM: w(z) = -1
M: -19.391 +0.024 -0.024 mag
H0: 69.10 +0.82 -0.82 km/s/Mpc
Ωm: 0.296 +0.006 -0.006
ωm: 0.1415 +0.0011 -0.0011
ωb: 0.02338 +0.00095 -0.00096
ωc: 0.11744 +0.00064 -0.00064
w0: -1
wa: 0
z_d: 1062.21 +2.09 -2.18
r_d: 146.65 Mpc
Chi squared: 1418.40
Log Evidence: -727.24
Degs of freedom: 1601

===============================

Flat wCDM: w(z) = w0
M: -19.375 +0.023 -0.023 mag
H0: 69.13 +0.76 -0.76 km/s/Mpc
Ωm: 0.297 +0.005 -0.005
ωm: 0.1421 +0.0012 -0.0012
ωb: 0.02604 +0.00140 -0.00137
ωc: 0.11538 +0.00104 -0.00107
w0: -0.913 +0.031 -0.032 (prior width 1.5: -1.5 to 0.0)
wa: 0
z_d: 1067.81 +2.79 -2.84
r_d: 144.36 Mpc
Chi squared: 1411.52
Log Evidence: -726.77
Degs of freedom: 1600

===============================

Flat wzCDM: w(z) = -1 + 4 * (1 + w0) / (1 + 3 * (1 + z)^3)
M: -19.390 +0.023 -0.023 mag
H0: 68.56 +0.80 -0.79 km/s/Mpc
Ωm: 0.302 +0.006 -0.006
ωm: 0.1419 +0.0012 -0.0012
ωb: 0.02475 +0.00105 -0.00106
ωc: 0.11656 +0.00072 -0.00072
w0: -0.878 +0.045 -0.046 (prior width 1.5: -1.5 to 0.0)
wa: d w(z)/dz at z=0 = -(9 / 4) * (1 + w0)
z_d: 1065.17 +2.20 -2.31
r_d: 145.40 Mpc
Chi squared: 1411.51
Log Evidence: -726.41
Degs of freedom: 1600

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
M: -19.387 +0.039 -0.035 mag
H0: 68.68 +1.38 -1.25 km/s/Mpc
Ωm: 0.301 +0.010 -0.011
ωm: 0.1419 +0.0012 -0.0012
ωb: 0.02496 +0.00307 -0.00244
ωc: 0.11643 +0.00201 -0.00286
w0: -0.891 +0.060 -0.055 (prior width 1.5: -1.5 to 0.0)
wa: -0.160 +0.343 -0.357 (prior width 6.0: -3.0 to 3.0)
z_d: 1065.60 +6.11 -5.28
r_d: 145.22 Mpc
Chi squared: 1411.37
Log Evidence: -728.15
Degs of freedom: 1599
"""
