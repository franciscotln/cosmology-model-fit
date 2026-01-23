from numba import njit
import numpy as np
from interpolator import interp_hermite
from y2023union3.data import get_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
inv_cov_sn = np.linalg.inv(cov_matrix_sn)

z_grid = np.linspace(0, np.max(z_sn_vals) + 0.1, num=2000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


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
def Hz(z, params):
    H0, Obh2, Och2, w0 = params[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


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
    dL = (1.0 + z_sn_vals) * DM_z(z_sn_vals, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


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
        (-1.0, 0.0),  # w0
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

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff7f0e"}
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
        (w0_16, w0_50, w0_84),
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
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r*: {cmb.rs_z(z_st_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"r_d: {cmb.rs_z(z_d_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log Evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {degrees_of_freedom}")

    labels = ["$Δ_M$", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_predictions(
        legend=sn_legend,
        x=z_sn_vals,
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
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*******************************
"""

"""
Flat ΛCDM w(z) = -1
H0: 67.40 +0.48 -0.48 km/s/Mpc
Ωm: 0.315 +0.007 -0.007
ωm: 0.14298 +0.00114 -0.00113
ωb: 0.02247 +0.00011 -0.00011
ωc: 0.1199 +0.0012 -0.0012
w0: -1
wa: 0
z*: 1089.76 +0.21 -0.21
z_drag: 1060.16 +0.23 -0.23
r*: 144.40 Mpc
r_d: 147.03 Mpc
Chi squared: 26.6
Log Evidence: -28.6
Degrees of freedom: 21
"""

"""
Flat wCDM w(z) = w0
H0: 65.31 +1.23 -1.23 km/s/Mpc
Ωm: 0.334 +0.013 -0.013
ωm: 0.14242 +0.00118 -0.00117
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1193 +0.0012 -0.0012
w0: -0.922 +0.043 -0.043 (prior width 1.0: -1.5 to -0.5)
wa: 0
z*: 1089.68 +0.22 -0.21
z_drag: 1060.17 +0.23 -0.23
r*: 144.53 Mpc
r_d: 147.15 Mpc
Chi squared: 23.1
Log Evidence: -29.1
Degrees of freedom: 20
"""

"""
Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
H0: 65.38 +1.02 -1.03 km/s/Mpc
Ωm: 0.333 +0.012 -0.011
ωm: 0.14238 +0.00116 -0.00116
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.1192 +0.0012 -0.0012
w0: -0.837 +0.074 -0.074 (prior width 1.0: -1.0 to 0.0)
wa: d w(z)/d z at z=0 = -1.5 * (1 - w0^2) = -0.447
z*: 1089.67 +0.21 -0.21
z_drag: 1060.18 +0.23 -0.23
r*: 144.54 Mpc
r_d: 147.16 Mpc
Chi squared: 22.2
Log Evidence: -28.1
Degrees of freedom: 20
"""

"""
Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 66.62 +1.35 -1.42 km/s/Mpc
Ωm: 0.321 +0.015 -0.013
ωm: 0.14255 +0.00121 -0.00118
ωb: 0.02249 +0.00011 -0.00011
ωc: 0.1194 +0.0012 -0.0012
w0: -0.686 +0.160 -0.165 (prior width 1.5: -1.5 to 0.0)
wa: -1.102 +0.753 -0.761 (prior width 8.5: -5.5 to 3.0)
z*: 1089.70 +0.22 -0.21
z_drag: 1060.17 +0.24 -0.23
r*: 144.50 Mpc
r_d: 147.12 Mpc
Chi squared: 21.8
Log Evidence: -29.7
Degrees of freedom: 19
"""
