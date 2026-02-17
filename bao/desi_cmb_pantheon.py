from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import cmb.data_planck_act_compression as cmb
from interpolator import interp_hermite
from y2022pantheonSHOES.data import get_data
from y2025BAO.data import get_data as get_bao_data

c = cmb.c  # km/s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mb_values, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=3000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / ((1.0 + w0) + (1.0 - w0) * zp1**3)) ** 2  # wzCDM
    # return 1  # ΛCDM
    # return zp1 ** (3 * (1.0 + w0))  # wCDM
    # return zp1 ** (3 * (1.0 + w0 + wa)) * np.exp(-3 * wa * z / zp1)  # w0waCDM


@njit
def Ez(z, H0, Obh2, Och2, w0):
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0 = params[1]
    return H0 * Ez(z, H0=H0, Obh2=params[2], Och2=params[3], w0=params[4])


cmb.set_HZ(H_z)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    Obh2, Och2 = params[2], params[3]
    Omh2 = Obh2 + Och2 + Omnu_h2
    rd = cmb.r_drag(wb=Obh2, wm=Omh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


@njit
def apparent_mag(params):
    M = params[0]
    return M + 25.0 + 5 * np.log10((1.0 + z_hel) * DM_z(z_cmb, params))


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_sn = mb_values - apparent_mag(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_bao + chi_sn


bounds = np.array(
    [
        (-20.0, -19.0),  # M
        (60.0, 75.0),  # H0
        (0.019, 0.025),  # ωb = Ωb * h^2
        (0.01, 0.25),  # ωc = Ωc * h^2
        (-1.0, -1 / 3),  # w0
    ]
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
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 100
    burn_in = 500
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.30),
        (emcee.moves.DEMove(), 0.70),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    try:
        tau = sampler.get_autocorr_time()
        print("Auto-correlation time", tau)
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("Effective samples:", nwalkers * ndim * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    MAP_index = np.argmax(log_probs)
    print("MAP chi^2:", chi_squared(samples[MAP_index]))

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    [
        (M_16, M_50, M_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)
    degs_of_freedom = (
        len(z_cmb) + len(bao_data["z"]) + len(cmb.DISTANCE_PRIORS) - len(best_fit)
    )

    omh2_samples = samples[:, 2] + samples[:, 3] + Omnu_h2
    om_samples = omh2_samples / (samples[:, 1] / 100) ** 2
    zst_samples = cmb.z_star(samples[:, 2], omh2_samples)
    zdr_samples = cmb.z_drag(samples[:, 2], omh2_samples)
    rdr_samples = cmb.r_drag(samples[:, 2], omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(om_samples, [15.9, 50, 84.1])
    zst_16, zst_50, zst_84 = np.percentile(zst_samples, [15.9, 50, 84.1])
    zdr_16, zdr_50, zdr_84 = np.percentile(zdr_samples, [15.9, 50, 84.1])
    rd_16, rd_50, rd_84 = np.percentile(rdr_samples, [15.9, 50, 84.1])

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z*: {zst_50:.2f} +{(zst_84 - zst_50):.2f} -{(zst_50 - zst_16):.2f}")
    print(f"z_d: {zdr_50:.2f} +{(zdr_84 - zdr_50):.2f} -{(zdr_50 - zdr_16):.2f}")
    print(f"r*: {cmb.rs_z(zst_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degrees of freedom: {degs_of_freedom}")

    labels = ["M", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    plot_corner_and_chains(
        labels=labels,
        flat_samples=samples,
        samples=chains_samples,
    )
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mb_values - M_50,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag(best_fit) - M_50,
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
Priors:

M  U(-20.0, -19.0)
H0 U(60.0, 75.0)
ωb U(0.019, 0.025)
ωc U(0.01, 0.25)

wCDM:
w0 U(-1.5, -0.5)

wzCDM (thawing quintessence):
w0 U(-1.0, -1/3)

w0waCDM:
w0 U(-1.5, 0.0)
wa U(-2.0, 1.0)
with w0 + wa < 0 enforced

Evolving absolute magnitude of SNe:
p U(-1.0, 2.0)
"""

"""
Flat ΛCDM w(z) = -1
H0: 68.37 +0.27 -0.27 km/s/Mpc
ωb: 0.02257 +0.00010 -0.00010
ωc: 0.1175 +0.0006 -0.0006
ωm: 0.1407 +0.0006 -0.0006
Ωm: 0.301 +0.004 -0.004
M: -19.412 +0.008 -0.008 mag
z*: 1089.43 +0.15 -0.15
z_d: 1060.19 +0.23 -0.23
r*: 144.94 Mpc
rd: 147.54 +0.19 -0.19 Mpc
MAP chi^2: 1420.34
Log evidence: -727.31
Degrees of freedom: 1602

===============================

Flat ΛCDM
Evolving absolute mag of SNe M(z) = M_max + p * [1 - (z / (0.1 + z))^0.05]

H0: 68.45 +0.27 -0.27 km/s/Mpc
ωb: 0.02258 +0.00010 -0.00010
ωc: 0.1173 +0.0007 -0.0006
ωm: 0.1405 +0.0007 -0.0006
Ωm: 0.300 +0.004 -0.004
M_max: -19.426 +0.011 -0.011 mag
p: 0.414 +0.203 -0.205
z*: 1089.40 +0.15 -0.15
z_d: 1060.20 +0.23 -0.23
r*: 144.98 Mpc
rd: 147.58 +0.19 -0.19 Mpc
MAP chi^2: 1416.27 (2.02 sigma away from no evolution in magnitude)
Log evidence: -727.04 (Δ logZ = 0.27 against no evolution in magnitude)
Degrees of freedom: 1601

===============================

Flat wCDM w(z) = w0
H0: 67.87 +0.59 -0.58 km/s/Mpc
ωb: 0.02259 +0.00011 -0.00010
ωc: 0.1170 +0.0008 -0.0008
ωm: 0.1403 +0.0008 -0.0008
Ωm: 0.304 +0.005 -0.005
M: -19.423 +0.014 -0.014 mag
w0: -0.978 +0.023 -0.023
z*: 1089.35 +0.17 -0.17
z_d: 1060.21 +0.23 -0.23
r*: 145.05 Mpc
rd: 147.65 +0.22 -0.22 Mpc
MAP chi^2: 1419.38 (0.98 sigma away from ΛCDM)
Log evidence: -729.68
Degrees of freedom: 1601

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
H0: 67.41 +0.55 -0.57 km/s/Mpc
ωb: 0.02259 +0.00010 -0.00010
ωc: 0.1170 +0.0007 -0.0007
ωm: 0.1402 +0.0007 -0.0007
Ωm: 0.309 +0.006 -0.005
M: -19.430 +0.012 -0.013 mag
w0: -0.919 +0.043 -0.041 (truncated at 1.88 sigma to the left of the mean)
z*: 1089.35 +0.16 -0.16
z_d: 1060.21 +0.23 -0.23
r*: 145.06 Mpc
rd: 147.65 +0.20 -0.20 Mpc
MAP chi^2: 1417.15 (1.79 sigma away from ΛCDM)
Log evidence: -727.54
Degrees of freedom: 1601

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.47 +0.60 -0.59 km/s/Mpc
ωb: 0.02229 +0.00013 -0.00013
ωc: 0.1184 +0.0010 -0.0010
ωm: 0.1413 +0.0009 -0.0009
Ωm: 0.310 +0.006 -0.006
w0: -0.856 +0.056 -0.055
wa: -0.495 +0.208 -0.224
M: -19.422 +0.015 -0.015
z*: 1089.90 +0.24 -0.23
z_d: 1059.80 +0.28 -0.28
r*: 144.92 Mpc
rd: 147.62 +0.23 -0.23 Mpc
Chi squared: 1412.63 (2.30 sigma away from ΛCDM)
Log evidence: -728.13
Degrees of freedom: 1600
"""
