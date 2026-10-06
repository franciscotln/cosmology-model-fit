from numba import njit
import numpy as np
from interpolator import interp_hermite, interp_pchip
from solve_triangular import solve_triangular
import cmb.data_spt_planck_act_compression as cmb
from y2025BAO.data_fs_lya import get_data
import y2024BBN.prior_lcdm_schoneberg as bbn

c = cmb.c_km_per_s
omnu_h2 = cmb.Omnu_h2
or_h2 = cmb.Or_h2

legend, bao, cov_matrix = get_data()
L_cov = np.linalg.cholesky(cov_matrix)

z_grid = np.linspace(0, np.max(bao["z"]) + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]

z_piv = 0.406


@njit
def Ode_z(z, wp, wa):
    zp1 = 1. + z
    return zp1**(3 * (1 + wp + (wa / (1. + z_piv)))) * np.exp(-3 * wa * z / zp1)


@njit
def H_z(z, params):
    h, om, w0, wa = params[0] / 100, params[1], params[3], params[4]
    h2 = h**2
    om_h2 = om * h2
    zp1 = 1. + z
    rad_term = or_h2 * zp1**4
    obcdm_term = (om_h2 - omnu_h2) * zp1**3
    nu_term = omnu_h2 * cmb.Omnu_z(z)
    lambda_term = (h2 - or_h2 - om_h2) * Ode_z(z, w0, wa)

    return 100 * np.sqrt(rad_term + obcdm_term + nu_term + lambda_term)


@njit
def DM_DH_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dh)
    return (cum_dm, dh_grid)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    h, Om, Obh2 = params[0] / 100, params[1], params[2]
    inv_rd = 1 / cmb.r_drag(Obh2, Om * h**2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    F_mask = qty == 3

    DM_vals, DH_vals = DM_DH_grid(params)
    DM = interp_hermite(z, z_grid, DM_vals, DH_vals)
    DH = interp_pchip(z, z_grid, DH_vals)

    results[DH_mask] = DH[DH_mask] * inv_rd
    results[DM_mask] = DM[DM_mask] * inv_rd
    results[DV_mask] = (z[DV_mask] * DH[DV_mask] * DM[DV_mask] ** 2) ** (1 / 3) * inv_rd
    results[F_mask] = DM[F_mask] / DH[F_mask]
    return results


@njit
def chi_squared(params):
    delta = bao["value"] - bao_theory(bao["z"], bao_qty, params)
    y = solve_triangular(L_cov, delta)
    return np.dot(y, y)


params = ["H0", "om", "obh2", "wp", "wa"]
labels=["H_0", "Ω_m", "ω_b", "w_{piv}", "w_a"]
bounds = np.array([
    (55.0, 75.0),  # H0
    (0.17, 0.50),  # Ωm
    (0.016, 0.030),  # Ωb h^2
    (-1.5, -0.5),  # w_pivot
    (-8.0, 1.0),  # wa
])

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    if params[3] + (params[4] / (1. + z_piv)) > -1 / 3:
        return -np.inf

    bbn_chi2 = ((bbn.Obh2 - params[2]) / bbn.Obh2_sigma) ** 2
    return normalization - 0.5 * bbn_chi2


@njit
def log_likelihood(params):
    return -0.5 * chi_squared(params)


@njit
def log_probability_jit(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def log_probability(params):
    return log_probability_jit(params)


def main():
    import emcee
    from multiprocessing import Pool
    from getdist import MCSamples, plots
    import matplotlib.pyplot as plt
    from bao.plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 100
    burn_in = 500
    nsteps = 5000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [(emcee.moves.KDEMove(), 0.2), (emcee.moves.DEMove(), 0.8)]

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"})

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=False)
    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    flat_log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    chain_list = np.moveaxis(samples, 1, 0)
    loglikes_list = np.moveaxis(log_probs, 1, 0)

    gd_samples = MCSamples(
        samples=chain_list,
        loglikes=-loglikes_list,
        names=params,
        labels=labels,
    )
    gd_samples.addDerived(
        cmb.r_drag(gd_samples["obh2"], gd_samples["om"] * (gd_samples["H0"] / 100)**2),
        name="rd",
        label="r_{drag}",
    )
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    print(gd_samples.corr(["om", "wp", "wa"]))

    best_fit = flat_samples[np.argmax(flat_log_probs)]

    print(f"Chi squared (MAP): {chi_squared(best_fit):.2f}")
    print(f"log likelihood (MAP): {log_likelihood(best_fit):.2f}")
    print(f"DOF: {len(bao)  - len(best_fit)}")

    plots.get_subplot_plotter().triangle_plot(
        gd_samples,
        params=params,
        title_limit=1,
        filled=True,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend}: $Ω_m$={gd_samples['om'].mean():.3f}",
    )


if __name__ == "__main__":
    main()


# *********************************
# Data set: DESI DR2 BAO + FS Lya
# Gaussian prior on Obh2 from BBN
# *********************************


# Flat ΛCDM:
# H0: 68.52 +- 0.59 km/s/Mpc
# ωb: 0.02219 +- 0.00055
# Ωm: 0.3015 +- 0.0077
# rd: 147.7 +- 1.5 Mpc
# Chi squared: 12.80
# log likelihood (MAP): -6.40
# DOF: 11
# ---------------------------------


# Flat wCDM:
# H_0 = 67.7 +- 2.0 km/s/Mpc
# Ωm = 0.3020 +- 0.0081
# ωb = 0.02219 +- 0.00055
# rd: 148.5 +2.2 -2.5 Mpc
# w: -0.969 +- 0.073 (prior ~U[-1.5, -0.5])
# Chi squared: 12.56
# log likelihood (MAP): -6.28
# DOF: 10
# ---------------------------------


# Flat w0waCDM at z_pivot = 0.406
# wp + wa / (1+z_pivot) <= -1/3 enforced in the likelihood
# 
# H0 = 62.9 +2.4 -2.9 km/s/Mpc
# Ωm = 0.402 +- 0.042
# ωb = 0.02219 +- 0.00055
# wp = -0.992 +0.073 -0.065 (prior ~U[-1.5, -0.5])
# wa = -3.3 +- 1.4 (prior ~U[-8, 1])
# rd: 143.4 +1.6 -2.2 Mpc
# Chi squared (MAP): 7.20
# log likelihood (MAP): -3.60
# DOF: 9
#
# Correlation matrix
#      om          wp          wa
# om   1.          0.20764926 -0.95852006
# wp   0.20764926  1.         -0.00843565
# wa  -0.95852006 -0.00843565  1.
# ---------------------------------
