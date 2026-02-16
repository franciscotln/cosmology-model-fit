from numba import njit
import numpy as np
from interpolator import interp_hermite
from y2026union3_1.data import get_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_data()
inv_cov_sn = np.linalg.inv(cov_matrix_sn)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=2000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


@njit
def Ez(z, H0, Obh2, Och2):
    h = H0 / 100
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def Hz(z, params):
    H0 = params[1]
    return H0 * Ez(z, H0=H0, Obh2=params[2], Och2=params[3])


cmb.set_HZ(Hz)


@njit
def DM_z(z, params):
    dh_grid = c / Hz(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def mu_theory(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    Mz = params[0] + 1.0 - (z_cmb / (0.1 + z_cmb)) ** (0.1 * params[4])
    return Mz + 25.0 + 5 * np.log10(dL)


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_sn = mu_vals - mu_theory(params)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (-1.0, 1.0),  # ΔM
        (60.0, 75.0),  # H0
        (0.010, 0.030),  # ωb
        (0.010, 0.250),  # ωc
        (-1.0, 2.0),  # p
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


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
    from sn.plotting import plot_predictions
    from corner_plot import plot_corner_and_chains
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 2500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
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
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    print("Gelman-Rubin:", gelman_rubin(chains_samples))

    one_sigma_percentiles = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_percentiles, axis=0).T
    [
        (dM_16, dM_50, dM_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (p_16, p_50, p_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)
    degrees_of_freedom = len(mu_vals) + len(cmb.DISTANCE_PRIORS) - len(best_fit)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_percentiles)
    z_d_16, z_d_50, z_d_84 = np.percentile(z_drag_samples, one_sigma_percentiles)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"p: {p_50:.3f} +{(p_84 - p_50):.3f} -{(p_50 - p_16):.3f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r*: {cmb.rs_z(z_st_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"r_d: {cmb.rs_z(z_d_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log Evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {degrees_of_freedom}")

    labels = ["$Δ_M$", "$H_0$", "$ω_b$", "$ω_c$", "$p$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: Union 3.1 (22 bins)
CMB(R, lA = π / θ*, ωb) ACT+Planck compressed
z range: 0.050 - 2.262
*******************************
"""

"""
Flat ΛCDM w(z) = -1
ΔM: -0.069 +0.011 -0.011
H0: 67.49 +0.48 -0.48 km/s/Mpc
Ωm: 0.313 +0.007 -0.007
ωm: 0.14275 +0.00113 -0.00113
ωb: 0.02248 +0.00011 -0.00011
ωc: 0.1196 +0.0012 -0.0012
z*: 1089.73 +0.21 -0.21
z_drag: 1060.16 +0.23 -0.23
r*: 144.46 Mpc
r_d: 147.08 Mpc
Chi squared: 29.5
Log Evidence: -33.1
Degrees of freedom: 21
"""

"""
Flat ΛCDM w(z) = -1, varying absolute magnitude
M(z) = ΔM_max + 1 - (z / (0.1 + z))^(0.1 * p)

ΔM_max: -0.078 +0.012 -0.012
p: 0.485 +0.269 -0.261 (prior U(-1.0, +2.0))
H0: 67.68 +0.49 -0.49 km/s/Mpc
Ωm: 0.311 +0.007 -0.007
ωm: 0.14231 +0.00115 -0.00114
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1192 +0.0012 -0.0012
z*: 1089.66 +0.21 -0.21
z_drag: 1060.18 +0.23 -0.23
r*: 144.56 Mpc
r_d: 147.18 Mpc
Chi squared: 26.1 (1.84 sigma away from no evolution)
Log Evidence: -32.9
Degrees of freedom: 20
"""

"""
Flat wCDM w(z) = w0
ΔM: -0.086 +0.018 -0.019
H0: 66.35 +1.15 -1.14 km/s/Mpc
Ωm: 0.324 +0.012 -0.012
ωm: 0.14241 +0.00118 -0.00117
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1193 +0.0012 -0.0012
w0: -0.957 +0.040 -0.039 (prior U(-1.5, -0.5))
z*: 1089.68 +0.21 -0.21
z_drag: 1060.17 +0.23 -0.23
r*: 144.53 Mpc
r_d: 147.15 Mpc
Chi squared: 28.3
Log Evidence: -34.9
Degrees of freedom: 20
"""

"""
Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
ΔM: -0.081 +0.012 -0.013
H0: 66.19 +0.85 -0.94 km/s/Mpc
Ωm: 0.325 +0.010 -0.009
ωm: 0.14232 +0.00116 -0.00115
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1192 +0.0012 -0.0012
w0: -0.895 +0.068 -0.060 (prior U(-1.0, 0.0))
wa: d w(z)/d z at z=0 = -1.5 * (1 - w0^2) = -0.447
z*: 1089.66 +0.21 -0.21
z_drag: 1060.18 +0.23 -0.23
r*: 144.56 Mpc
r_d: 147.17 Mpc
Chi squared: 27.7
Log Evidence: -33.9
Degrees of freedom: 20
"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
TODO
w0: (prior U(-1.5, 0.0))
wa: (prior U(-5.5, 3.0))
"""
