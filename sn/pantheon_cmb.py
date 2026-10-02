from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2022pantheonSHOES.data import get_data
import cmb.data_spt_planck_act_compression as cmb

c = cmb.c_km_per_s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

sn_legend, z_cmb, z_hel, mb_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
logdet_sn = 2 * np.sum(np.log(np.diag(cho_sn)))
N_sn = len(z_cmb)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0, wa):
    # w1w2CDM
    zp1 = 1.0 + z
    return zp1**(3 * (1.0 + w0 + wa)) * ((2 * zp1**2) / (1.0 + zp1**2))**(-3 * wa)


@njit
def H_z(z, params):
    h, Obh2, Och2 = params[1], params[2], params[3]
    zp1 = 1.0 + z

    radiation_term = Orh2 * zp1**4
    matter_term = (Obh2 + Och2) * zp1**3
    neutrino_term = Omnuh2 * cmb.Omnu_z(z)
    lambda_term = h**2 - Orh2 - Omnuh2 - Obh2 - Och2

    return 100 * np.sqrt(radiation_term + matter_term + neutrino_term + lambda_term)


cmb.set_HZ(H_z)


@njit
def DM_z(z, params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def get_z_cosmo(params):
    # Heaviside step at z = 0.15
    v_km_s = 100 * params[4] * np.where(z_cmb <= 0.15, 1, -1)
    z_pec = v_km_s / c
    return (1.0 + z_cmb) / (1.0 + z_pec) - 1.0


def mu_corr(params, DM_ref):
    # For plotting purposes only
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_ref)


@njit
def mu_theory(DM):
    return 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi2_sn(params):
    M = params[0]
    z_cosmo = get_z_cosmo(params)
    delta_sn = mb_values - M - mu_theory(DM_z(z_cosmo, params))
    y = solve_triangular(cho_sn, delta_sn)
    return np.dot(y, y)


@njit
def log_likelihood_sn(params):
    return -0.5 * (chi2_sn(params) + logdet_sn + N_sn * np.log(2 * np.pi))


labels = ["M", "h", "$ω_b$", "$ω_c$", "$v_{100}$"]
bounds = np.array(
    [
        (-20.0, -19.0),  # M
        (0.60, 0.75),  # h
        (0.010, 0.030),  # Ωb * h^2
        (0.010, 0.25),  # Ωc * h^2
        (-2.5, 2.5),  # v 100 km/s
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


@njit
def log_likelihood(params):
    return log_likelihood_sn(params) + cmb.log_likelihood(params[2], params[3], params)


@njit
def log_probability_njit(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def log_probability(params):
    return log_probability_njit(params)


def main():
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains
    from gelman_rubin import gelman_rubin
    from sn.plotting import plot_predictions as plot_sn_predictions

    ndim = len(bounds)
    nwalkers = 100
    burn_in = 500
    nsteps = 3500 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [(emcee.moves.KDEMove(), 0.2), (emcee.moves.DEMove(), 0.8)]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"})

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", nwalkers * ndim * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    print("Gelman-Rubin R^:", gelman_rubin(chains))

    one_sigma_conf_int = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_conf_int, axis=0).T
    [
        (M_16, M_50, M_84),
        (h_16, h_50, h_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (vf_16, vf_50, vf_84),
    ] = pct

    Omh2_samples = samples[:, 2] + samples[:, 3] + Omnuh2
    Om_samples = Omh2_samples / samples[:, 1] ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    r_drag_samples = cmb.r_drag(samples[:, 2], Omh2_samples)
    r_star_samples = [cmb.rs_z(z_star_samples[i], samples[i, 2], samples[i]) for i in range(samples.shape[0])]

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_conf_int)
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, one_sigma_conf_int)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, one_sigma_conf_int)
    z_d_16, z_d_50, z_d_84 = np.percentile(z_drag_samples, one_sigma_conf_int)
    r_d_16, r_d_50, r_d_84 = np.percentile(r_drag_samples, one_sigma_conf_int)
    rs_16, rs_50, rs_84 = np.percentile(r_star_samples, one_sigma_conf_int)

    print(f"h: {h_50:.4f} +{(h_84 - h_50):.4f} -{(h_50 - h_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.5f} +{(Och2_84 - Och2_50):.5f} -{(Och2_50 - Och2_16):.5f}")
    print(f"v: {vf_50:.3f} +{(vf_84 - vf_50):.3f} -{(vf_50 - vf_16):.3f} x 100 km/s")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f} mag")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_d: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r* = {rs_50:.2f} +{(rs_84 - rs_50):.2f} -{(rs_50 - rs_16):.2f} Mpc")
    print(f"rd: {r_d_50:.2f} +{(r_d_84 - r_d_50):.2f} -{(r_d_50 - r_d_16):.2f} Mpc")

    best_fit = samples[np.argmax(log_probs)]
    print(f"Chi2 (MAP): {chi2_sn(best_fit) + cmb.chi2(best_fit[2], best_fit[3], best_fit):.2f}")

    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains)
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mb_values - M_50 - mu_corr(best_fit, DM_z(z_cmb, best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(DM_z(z_cmb, best_fit)),
        label=f"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# h: 0.6711 +0.0037 -0.0036
# Ωm: 0.319 +0.005 -0.005
# ωm: 0.14351 +0.00088 -0.00088
# ωb: 0.02239 +0.00009 -0.00009
# ωc: 0.12048 +0.00090 -0.00090
# M: -19.447 +0.011 -0.011 mag
# z*: 1088.80 +0.13 -0.13
# z_d: 1060.00 +0.21 -0.21
# r* = 144.40 +0.22 -0.22 Mpc
# rd: 146.95 +0.23 -0.23 Mpc
# Chi2 (MAP): 1403.48
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.15 inflow z > 0.15 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# h: 0.6721 +0.0037 -0.0037
# Ωm: 0.3172 +0.0053 -0.0053
# ωm: 0.14328 +0.00088 -0.00089
# ωb: 0.02240 +0.00009 -0.00009
# ωc: 0.12024 +0.00090 -0.00091
# v: -0.655 +0.360 -0.361 x 100 km/s (prior ~ U[-2.5, 2.5])
# M: -19.451 +0.011 -0.011 mag
# z*: 1088.77 +0.13 -0.13
# z_d: 1060.00 +0.21 -0.21
# r* = 144.46 +0.22 -0.22 Mpc
# rd: 147.00 +0.23 -0.23 Mpc
# Chi2 (MAP): 1400.23 (1.8 sigma significance)
# ---------------------------------


# ----------- Flat wCDM -----------
# h: 0.6651 +0.0081 -0.0080
# Ωm: 0.324 +0.009 -0.008
# ωm: 0.14332 +0.00091 -0.00092
# ωb: 0.02240 +0.00009 -0.00009
# ωc: 0.12027 +0.00093 -0.00094
# w0: -0.977 +0.028 -0.029 (prior ~ U[-1.5, -0.5])
# M: -19.461 +0.020 -0.020 mag
# z*: 1088.77 +0.13 -0.13
# z_d: 1060.01 +0.21 -0.21
# r* = 144.45 +0.23 -0.23 Mpc
# rd: 147.00 +0.24 -0.23 Mpc
# Chi2 (MAP): 1402.78
# ---------------------------------


# ----------- Flat w1w2CDM --------
# w(z) = w1 + w2 * ((1 + z)^2 - 1) / ((1 + z)^2 + 1)
# w1 + w2 < 0 enforced in the likelihood
#
# h: 0.6717 +0.0134 -0.0144
# Ωm: 0.318 +0.014 -0.013
# ωm: 0.14336 +0.00092 -0.00091
# ωb: 0.02240 +0.00009 -0.00010
# ωc: 0.12031 +0.00094 -0.00092
# w1: -0.917 +0.107 -0.109 (prior ~ U[-2, 0])
# w2: -0.27 +0.47 -0.50 (prior ~ U[-3, 3])
# M: -19.436 +0.045 -0.050 mag
# z*: 1088.78 +0.13 -0.13
# z_d: 1060.01 +0.21 -0.21
# r* = 144.44 +0.23 -0.23 Mpc
# rd: 146.99 +0.24 -0.24 Mpc
# Chi2 (MAP): 1402.71
# ---------------------------------
