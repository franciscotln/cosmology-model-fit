from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2025DESdovekie.data import get_data as get_sn_data
from y2025BAO.data import get_data as get_bao_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
inv_cov_bao = np.linalg.inv(bao_cov_matrix)

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1 + w0 + (1 - w0) * zp1**3)) ** 2  # wzCDM
    # return 1  # ΛCDM
    # return zp1 ** (3 * (1 + w0))  # wCDM
    # return zp1 ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / zp1)  # w0waCDM


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
def H_z(z, params):
    H0, Obh2, Och2, w0 = params[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


cmb.set_HZ(H_z)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
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
    Omh2 = Obh2 + Och2 + Omnuh2
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
def theory_mu(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta @ cmb.inv_cov_mat @ delta

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_sn = mu_values - theory_mu(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_bao + chi_sn


bounds = np.array(
    [
        (-0.5, 0.5),  # ΔM
        (60.0, 75.0),  # H0
        (0.010, 0.030),  # ωb
        (0.01, 0.25),  # ωc
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
    from gelman_rubin import gelman_rubin
    from corner_plot import plot_corner_and_chains
    from sn.plotting import plot_predictions as plot_sn_predictions
    from bao.plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 250
    nsteps = 2500 + burn_in
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
        print("Auto-correlation time", tau)
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("Effective samples:", nwalkers * ndim * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)
    print(f"Gelman-Rubin: {gelman_rubin(chains_samples)}")

    one_sigma_contours = [15.9, 50, 84.1]

    pct = np.percentile(samples, one_sigma_contours, axis=0).T
    [
        (dM_16, dM_50, dM_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    wa_samples = -1.5 * (1 - samples[:, 4] ** 2)  # wzCDM
    wa_16, wa_50, wa_84 = np.percentile(wa_samples, one_sigma_contours)

    omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    om_samples = omh2_samples / (samples[:, 1] / 100) ** 2
    z_star_samples = cmb.z_star(wb=samples[:, 2], wm=omh2_samples)
    z_drag_samples = cmb.z_drag(wb=samples[:, 2], wm=omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(omh2_samples, one_sigma_contours)
    Om_16, Om_50, Om_84 = np.percentile(om_samples, one_sigma_contours)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_contours)
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(z_drag_samples, one_sigma_contours)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"wa: {wa_50:.3f} +{(wa_84 - wa_50):.3f} -{(wa_50 - wa_16):.3f}")
    print(f"r*: {cmb.rs_z(z_st_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r_d: {cmb.rs_z(z_dr_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"z_d: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")

    labels = ["$Δ_M$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
*******************************
DESI DR2 + DES5Y + (R, π/θ*, ωb)CMB
*******************************
"""

"""
Flat ΛCDM: w(z) = -1

** Early time ΛCDM **
H0: 68.25 +0.29 -0.29 km/s/Mpc
Ωm: 0.3016 +0.0037 -0.0037
ωb: 0.02235 +0.00012 -0.00012
ωc: 0.1175 +0.0006 -0.0006
ωm: 0.1405 +0.0006 -0.0006
w0: -1
wa: 0
r*: 145.09 Mpc
z*: 1089.82 +0.17 -0.17
r_d: 147.78 Mpc
z_d: 1059.80 +0.27 -0.27
Chi squared: 1648.82
Log evidence: -842.4

** ACT DR6 + Planck **
H0: 68.34 +0.27 -0.27 km/s/Mpc
Ωm: 0.3015 +0.0036 -0.0036
ωb: 0.02256 +0.00010 -0.00010
ωc: 0.1176 +0.0006 -0.0007
ωm: 0.1408 +0.0006 -0.0006
w0: -1
wa: 0
r*: 144.91 Mpc
z*: 1089.44 +0.15 -0.15
r_d: 147.52 Mpc
z_d: 1060.19 +0.23 -0.23
Chi squared: 1649.70
Log evidence: -843.0
"""


"""
Flat wCDM: w(z) = w0

** Early time ΛCDM **
H0: 67.61 +0.54 -0.53 km/s/Mpc
Ωm: 0.3059 +0.0049 -0.0048
ωb: 0.02241 +0.00013 -0.00013
ωc: 0.1168 +0.0008 -0.0008
ωm: 0.1398 +0.0008 -0.0008
w0: -0.969 +0.022 -0.022 (prior width 2/3: -4/3 to -2/3)
wa: 0
r*: 145.25 Mpc
z*: 1089.68 +0.20 -0.20
r_d: 147.91 Mpc
z_d: 1059.88 +0.28 -0.28
Chi squared: 1646.78
Log evidence: -843.9 (Δ logZ = -1.5 in favour of ΛCDM)

** ACT DR6 + Planck **
H0: 67.73 +0.54 -0.53 km/s/Mpc
Ωm: 0.3056 +0.0048 -0.0047
ωb: 0.02259 +0.00011 -0.00011
ωc: 0.1170 +0.0008 -0.0008
ωm: 0.1402 +0.0008 -0.0008
w0: -0.972 +0.021 -0.022 (prior width 2/3: -4/3 to -2/3)
wa: -0.083 +0.064 -0.062
r*: 145.06 Mpc
z*: 1089.34 +0.17 -0.17
r_d: 147.66 Mpc
z_d: 1060.21 +0.23 -0.23
Chi squared: 1647.95
Log evidence: -844.6 (Δ logZ = -1.6 in favour of ΛCDM)
"""


"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)

** Early time ΛCDM **
H0: 67.17 +0.53 -0.54 km/s/Mpc
Ωm: 0.3100 +0.0053 -0.0051
ωb: 0.02241 +0.00012 -0.00012
ωc: 0.1168 +0.0007 -0.0007
ωm: 0.1399 +0.0007 -0.0007
w0: -0.903 +0.041 -0.041 (prior width 2/3: -1 to -1/3)
wa: -0.276 +0.114 -0.108
r*: 145.24 Mpc
z*: 1089.69 +0.18 -0.18
r_d: 147.91 Mpc
z_d: 1059.89 +0.27 -0.27
Chi squared: 1643.70
Log evidence: -841.8 (Δ logZ = 0.6 against ΛCDM)

** ACT DR6 + Planck **
H0: 67.26 +0.54 -0.54 km/s/Mpc
Ωm: 0.3099 +0.0053 -0.0051
ωb: 0.02260 +0.00010 -0.00010
ωc: 0.1169 +0.0007 -0.0007
ωm: 0.1402 +0.0007 -0.0007
w0: -0.906 +0.041 -0.041 (prior width 2/3: -1 to -1/3)
wa: -0.268 +0.113 -0.110
r*: 145.07 Mpc
z*: 1089.33 +0.16 -0.16
r_d: 147.67 Mpc
z_d: 1060.22 +0.23 -0.23
Chi squared: 1644.75
Log evidence: -842.4 (Δ logZ = 0.6 against ΛCDM)
"""


"""
Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)

** Early time ΛCDM **
H0: 67.31 +0.55 -0.54 km/s/Mpc
Ωm: 0.3123 +0.0055 -0.0054
ωb: 0.02227 +0.00014 -0.00013
ωc: 0.1186 +0.0010 -0.0010
ωm: 0.1415 +0.0009 -0.0010
w0: -0.824 +0.058 -0.056 (prior width 1.5: -1.5 to 0.0)
wa: -0.613 +0.224 -0.239 (prior width 3.5: -2.5 to 1.0)
r*: 144.88 Mpc
z*: 1089.94 +0.24 -0.24
r_d: 147.57 Mpc
z_d: 1059.77 +0.28 -0.28
Chi squared: 1638.92
Log evidence: -842.4 (Δ logZ = 0.1 compared to ΛCDM)

** ACT DR6 + Planck **
H0: 67.43 +0.55 -0.54 km/s/Mpc
Ωm: 0.3120 +0.0054 -0.0053
ωb: 0.02252 +0.00011 -0.00011
ωc: 0.1187 +0.0010 -0.0010
ωm: 0.1419 +0.0009 -0.0009
w0: -0.822 +0.057 -0.055 (prior width 1.5: -1.5 to 0.0)
wa: -0.620 +0.216 -0.231 (prior width 3.5: -2.5 to 1.0)
r*: 144.66 Mpc
z*: 1089.60 +0.19 -0.19
r_d: 147.28 Mpc
z_d: 1060.18 +0.23 -0.24
Chi squared: 1639.18
Log evidence: -842.7 (Δ logZ = 0.3 compared to ΛCDM)
"""
