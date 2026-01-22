from numba import njit
import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_pchip
from y2022pantheonSHOES.data import get_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mb_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dx = np.diff(z_grid)


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
    dark_energy_term = Ode * (2 * zp1**3 / ((1.0 + w0) + (1.0 - w0) * zp1**3)) ** 2

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0, Obh2, Och2, w0 = params[1:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


@njit
def DM_z(z, params):
    dh_grid = c / H_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_pchip(z, z_grid, cum_dm)


@njit
def apparent_mag(params):
    dL = (1.0 + z_hel) * DM_z(z_cmb, params)
    return params[0] + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(H_z, params[2], params[3], params)
    chi2_cmb = delta @ cmb.inv_cov_mat @ delta

    delta_sn = mb_values - apparent_mag(params)
    chi_sn = solve_triang(cho_sn, delta_sn)

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (-20.0, -19.0),  # M
        (60.0, 75.0),  # H0
        (0.010, 0.030),  # Ωb * h^2
        (0.010, 0.25),  # Ωc * h^2
        (-1.0, 0.0),  # w0
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
    from corner_plot import plot_corner_and_chains
    from gelman_rubin import gelman_rubin
    from sn.plotting import plot_predictions as plot_sn_predictions

    ndim = len(bounds)
    nwalkers = 150
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
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", nwalkers * ndim * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains = sampler.get_chain(discard=burn_in, flat=False)

    print("Gelman-Rubin R^:", gelman_rubin(chains))

    one_sigma_conf_int = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_conf_int, axis=0).T
    [
        (M_16, M_50, M_84),
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 1] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    r_drag_samples = cmb.r_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_conf_int)
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, one_sigma_conf_int)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_conf_int)
    z_d_16, z_d_50, z_d_84 = np.percentile(z_drag_samples, one_sigma_conf_int)
    r_d_16, r_d_50, r_d_84 = np.percentile(r_drag_samples, one_sigma_conf_int)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.5f} +{(Och2_84 - Och2_50):.5f} -{(Och2_50 - Och2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_d: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r* = {cmb.rs_z(H_z, z_st_50, Obh2_50, best_fit):.2f} Mpc")
    print(f"rd: {r_d_50:.2f} +{(r_d_84 - r_d_50):.2f} -{(r_d_50 - r_d_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    labels = ["M", "$H_0$", "$ω_b$", "$ω_c$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains)
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
Flat ΛCDM w(z) = -1
H0: 67.43 +0.46 -0.46 km/s/Mpc
Ωm: 0.314 +0.007 -0.007
ωm: 0.14291 +0.00111 -0.00109
ωb: 0.02248 +0.00011 -0.00011
ωc: 0.11979 +0.00114 -0.00112
w0: -1
wa: 0
M: -19.438 +0.013 -0.013 mag
z*: 1089.75 +0.20 -0.20
z_d: 1060.16 +0.23 -0.23
r* = 144.42 Mpc
rd: 147.04 +0.28 -0.28 Mpc
Chi squared: 1403.98

===============================

Flat wCDM w(z) = w0
H0: 66.67 +0.83 -0.82 km/s/Mpc
Ωm: 0.320 +0.009 -0.009
ωm: 0.14245 +0.00118 -0.00117
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.11931 +0.00122 -0.00121
w0: -0.968 +0.029 -0.029 (prior width 1.0: from -1.5 to -0.5)
wa: 0
M: -19.456 +0.021 -0.021 mag
z*: 1089.68 +0.22 -0.21
z_d: 1060.17 +0.23 -0.23
r* = 144.52 Mpc
rd: 147.14 +0.29 -0.29 Mpc
Chi squared: 1402.70

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
H0: 66.73 +0.60 -0.64 km/s/Mpc
Ωm: 0.320 +0.008 -0.007
ωm: 0.14236 +0.00115 -0.00113
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.11921 +0.00119 -0.00117
w0: -0.935 +0.045 -0.038 (prior width 1.0: from -1.0 to 0.0)
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0**2)
M: -19.451 +0.015 -0.016 mag
z*: 1089.67 +0.21 -0.21
z_d: 1060.18 +0.23 -0.23
r* = 144.55 Mpc
rd: 147.16 +0.29 -0.29 Mpc
Chi squared: 1402.80

===============================

Flat w(z) = w0 + wa * z / (1 + z)
TODO
"""
