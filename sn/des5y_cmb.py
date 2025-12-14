from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from y2025DESdovekie.data import get_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Omega_r_h2(2.044)
Onuh2 = cmb.Omnu_h2
z_nr = cmb.z_nr

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
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
def Ode_z(z, w0, wa):
    a3 = 1 / (1 + z) ** 3
    return 4 / ((1 + w0) * a3 + (1 - w0)) ** 2


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2
    Onu = Onuh2 / h**2
    Or = Orh2 / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0, wa)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def DM_z(z, theta):
    H0, Obh2, Och2, w0 = theta[1:]
    dh_grid = (c / H0) / Ez(z_grid, H0, Obh2, Och2, w0)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return np.interp(z, z_grid, cum_dm)


@njit
def theory_mu(params):
    dL = (1 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, *params[1:])
    chi2_cmb = np.dot(delta, np.dot(cmb.inv_cov_mat, delta))

    delta_sn = mu_vals - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (-0.7, 0.7),  # ΔM
        (55.0, 75.0),  # H0
        (0.010, 0.030),  # Ωb * h^2
        (0.01, 0.25),  # Ωc * h^2
        (-1.0, 0.0),  # w0
    ],
    dtype=np.float64,
)


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee
    from multiprocessing import Pool
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from gelman_rubin import gelman_rubin
    from sn.plotting import plot_predictions as plot_sn_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 250
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.70),
    ]

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", nwalkers * nsteps * ndim / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)
    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    [
        (dM_16, dM_50, dM_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Onuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    zst_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    zdr_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(zst_samples, [15.9, 50, 84.1])
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(zdr_samples, [15.9, 50, 84.1])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"zd: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, *best_fit[1:]):.2f} Mpc")
    print(f"r_d: {cmb.rs_z(Ez, z_dr_50, *best_fit[1:]):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")

    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_corner_and_chains(
        labels=["$ΔM$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
H0: 67.38 +0.45 -0.46 km/s/Mpc
Ωm: 0.315 +0.007 -0.006
ωb: 0.02247 +0.00011 -0.00011
ωc: 0.1199 +0.0011 -0.0011
ωm: 0.1430 +0.0011 -0.0011
w0: -1
wa: 0
z*: 1089.77 +0.20 -0.20
zd: 1060.15 +0.23 -0.23
r*: 144.39 Mpc
r_d: 147.02 Mpc
Chi squared: 1632.68
Log evidence: -834.5
"""

"""
Flat wCDM w(z) = w0
H0: 66.67 +0.72 -0.72 km/s/Mpc
Ωm: 0.320 +0.008 -0.008
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1193 +0.0012 -0.0012
ωm: 0.1424 +0.0012 -0.0012
w0: -0.967 +0.026 -0.026 (prior width 1.5: -1.5 - 0.0)
wa: 0
z*: 1089.68 +0.21 -0.21
zd: 1060.17 +0.23 -0.24
r*: 144.53 Mpc
r_d: 147.15 Mpc
Chi squared: 1631.01
Log evidence: -836.9
"""

"""
Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
H0: 66.67 +0.58 -0.61 km/s/Mpc
Ωm: 0.320 +0.007 -0.007
ωb: 0.02251 +0.00011 -0.00011
ωc: 0.1191 +0.0012 -0.0012
ωm: 0.1423 +0.0011 -0.0011
w0: -0.927 +0.044 -0.040 (prior width 1.0: -1.0 - 0.0)
wa: d w(z)/dz at z=0 = -(3/2) * (1 - w0^2)
z*: 1089.65 +0.21 -0.21
zd: 1060.18 +0.23 -0.24
r*: 144.56 Mpc
r_d: 147.18 Mpc
Chi squared: 1630.48
Log evidence: -835.6
"""

"""
Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.79 +0.96 -1.06 km/s/Mpc
Ωm: 0.310 +0.011 -0.009
ωb: 0.02249 +0.00011 -0.00011
ωc: 0.1194 +0.0012 -0.0012
ωm: 0.1426 +0.0012 -0.0012
w0: -0.810 +0.111 -0.116 (prior width 1.5: -1.5 to 0.0)
wa: -0.769 +0.557 -0.550 (prior width 6.5: -4.0 to 2.5)
z*: 1089.70 +0.22 -0.21
zd: 1060.17 +0.23 -0.24
r*: 144.50 Mpc
r_d: 147.12 Mpc
Chi squared: 1629.53
Log evidence: -837.1
"""
