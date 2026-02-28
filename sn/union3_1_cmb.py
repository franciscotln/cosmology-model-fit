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
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence with w(z) ranging from -1 to 1
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
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


correction_mask = z_cmb <= 0.2


@njit
def mu_corr(params):
    z_pec = 100 * params[4] / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)

    return np.where(
        correction_mask,
        5.0 * np.log10(DM_z(z_cosmo, params) / DM_z(z_cmb, params)),
        0.0,
    )


@njit
def mu_theory(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


def chi_squared(params):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[2], params[3], params)
    chi2_cmb = delta_cmb @ cmb.inv_cov_mat @ delta_cmb

    delta_sn = mu_vals - mu_theory(params) - mu_corr(params)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (-1.0, 1.0),  # ΔM
        (60.0, 75.0),  # H0
        (0.010, 0.030),  # ωb
        (0.010, 0.250),  # ωc
        (-12.0, 5.0),  # v x 100 km/s
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
    from getdist import plots, MCSamples
    from matplotlib import pyplot as plt
    from sn.plotting import plot_predictions
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
    log_probs_chains = sampler.get_log_prob(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    chain_list = [chains_samples[:, i, :] for i in range(chains_samples.shape[1])]
    loglikes = [-log_probs_chains[:, i] for i in range(log_probs_chains.shape[1])]

    gd_samples = MCSamples(
        samples=chain_list,
        loglikes=loglikes,
        names=["dM", "H0", "obh2", "och2", "v"],
        labels=["Δ_M", "H_0", "ω_b", "ω_c", "v{100}"],
        label="Union3.1 + CMB(R, lA, ωb)",
    )
    gd_samples.addDerived(
        Omnuh2 + gd_samples["obh2"] + gd_samples["och2"], name="omh2", label="ω_m"
    )
    gd_samples.addDerived(
        gd_samples["omh2"] / (gd_samples["H0"] / 100) ** 2, name="om", label="Ω_m"
    )

    g = plots.get_subplot_plotter()
    g.triangle_plot(
        gd_samples,
        params=["dM", "v", "H0", "om"],
        title_limit=1,
        filled=True,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    print(f"Gelman-Rubin: {gd_samples.getGelmanRubin():.1e}")

    best_fit = np.percentile(samples, 50, axis=0)
    degs_of_freedom = len(mu_vals) + len(cmb.DISTANCE_PRIORS) - len(best_fit)

    MAP_index = np.argmax(log_probs)
    print(f"Chi2 (MAP): {chi_squared(samples[MAP_index]):.1f}")
    print(f"Log Evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {degs_of_freedom}")

    plot_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"ΛCDM",
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
ΔM: -0.069 +- 0.011
H0: 67.50 +- 0.48 km/s/Mpc
Ωm: 0.3134 +- 0.0069
Chi2 (MAP): 29.6
Log Evidence: -33.1
Degrees of freedom: 21
"""

"""
Flat ΛCDM w(z) = -1
Isotropic velocity SNe observed redshifts (limit to z <= 0.2)
z_cosmo = -1 + (1 + z) / (1 + v/c)

ΔM: -0.071 ± 0.011 mag
v: -3.4 ± 1.4 (prior ~ U(-12, 5)) x 100 km/s
v / (z_cut=0.2): -1700 ± 700 km/s
H0: 67.69 ± 0.49 km/s/Mpc
Ωm: 0.3107 ± 0.0069
Chi2 (MAP): 23.6 (2.45 sigma away from no flow)
Log Evidence: -31.8 (delta logZ = 1.3 in favour of flow)
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
