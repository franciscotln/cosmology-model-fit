from numba import njit
import numpy as np
from interpolator import interp_hermite, interp_pchip
from y2025BAO.data_fs_lya import get_data
import y2024BBN.prior_lcdm_schoneberg as bbn
from cmb.data_early_lcdm_compression import r_drag, c

legend, bao, cov_matrix = get_data()
inv_cov = np.linalg.inv(cov_matrix)

z_grid = np.linspace(0, np.max(bao["z"]) + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]

z_piv = 0.406


@njit
def Ode_z(z, wp, wa):
    zp1 = 1. + z
    return zp1**(3 * (1 + wp + (wa / (1. + z_piv)))) * np.exp(-3 * wa * z / zp1)


@njit
def H_z(z, params):
    H0, Om, w0, wa = params[0], params[1], params[3], params[4]
    return H0 * np.sqrt(Om * (1. + z) ** 3 + (1. - Om) * Ode_z(z, w0, wa))


@njit
def DM_grid(params):
    dh_grid = c / H_z(z_grid, params)
    n = z_grid.size
    cum_dm = np.zeros(n, dtype=np.float64)

    # Compute local derivatives d(dh)/dz using central differences
    d_dh = np.empty(n, dtype=np.float64)

    # Central difference for internal points
    d_dh[1:-1] = (dh_grid[2:] - dh_grid[:-2]) / (2 * dz)
    # Forward/Backward difference at boundaries
    d_dh[0] = (dh_grid[1] - dh_grid[0]) / dz
    d_dh[-1] = (dh_grid[-1] - dh_grid[-2]) / dz

    # Integrate with 4th-order cubic correction per interval
    dz_sq_over_12 = (dz ** 2) / 12
    acc = 0.0

    for i in range(n - 1):
        # Trapezoidal area + 1st-derivative endpoint correction
        trap = 0.5 * dz * (dh_grid[i] + dh_grid[i + 1])
        corr = dz_sq_over_12 * (d_dh[i] - d_dh[i + 1])
        acc += trap + corr
        cum_dm[i + 1] = acc

    return (cum_dm, dh_grid)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in bao["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    h, Om, Obh2 = params[0] / 100, params[1], params[2]
    rd = r_drag(Obh2, Om * h**2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    F_mask = qty == 3

    DM_vals, DH_vals = DM_grid(params)
    DM = interp_hermite(z, z_grid, DM_vals, DH_vals)
    DH = interp_pchip(z, z_grid, DH_vals)

    results[DH_mask] = DH[DH_mask] / rd
    results[DM_mask] = DM[DM_mask] / rd
    results[DV_mask] = (z[DV_mask] * DH[DV_mask] * DM[DV_mask] ** 2) ** (1 / 3) / rd
    results[F_mask] = DM[F_mask] / DH[F_mask]
    return results


@njit
def chi_squared(params):
    delta = bao["value"] - bao_theory(bao["z"], bao_qty, params)
    return delta @ inv_cov @ delta


bounds = np.array(
    [
        (55.0, 75.0),  # H0
        (0.17, 0.50),  # Ωm
        (0.016, 0.030),  # Ωb h^2
        (-1.5, -0.5),  # w_pivot
        (-8.0, 1.0),  # wa
    ]
)

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
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.2),
        (emcee.moves.DEMove(), 0.8),
    ]

    with Pool(5) as pool:
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

    samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=False)
    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    flat_log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    chain_list = np.moveaxis(samples, 1, 0)
    loglikes_list = np.moveaxis(log_probs, 1, 0)

    params = ["H0", "om", "obh2", "wp", "wa"]
    labels=["H_0", "Ω_m", "ω_b", "w_{piv}", "w_a"]
    gd_samples = MCSamples(
        samples=chain_list,
        loglikes=-loglikes_list,
        names=params,
        labels=labels,
    )
    gd_samples.addDerived(
        r_drag(gd_samples["obh2"], gd_samples["om"] * (gd_samples["H0"] / 100)**2),
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
# H0: 68.55 +- 0.59 km/s/Mpc
# ωb: 0.02218 +- 0.00055
# Ωm: 0.3019 +- 0.0077
# rd: 147.6 +- 1.5 Mpc
# Chi squared: 12.81
# log likelihood (MAP): -6.40
# DOF: 11
# ---------------------------------


# Flat wCDM:
# H_0 = 67.7 +- 2.0 km/s/Mpc
# Ωm = 0.3023 +- 0.0082
# ωb = 0.02218 +- 0.00055
# rd: 148.4 +2.4 -2.3 Mpc
# w0: -0.970 +- 0.073 (prior ~U[-1.5, -0.5])
# Chi squared: 12.57
# log likelihood (MAP): -6.28
# DOF: 10
# ---------------------------------


# Flat w0waCDM at z_pivot = 0.406
# wp + wa / (1+z_pivot) <= -1/3 enforced in the likelihood
# 
# H0 = 63.0 +2.5 -2.9 km/s/Mpc
# Ωm = 0.402 +- 0.042
# ωb = 0.02219 +- 0.00055
# wp = -0.993 +0.072 -0.066 (prior ~U[-1.5, -0.5])
# wa = -3.3 +- 1.4 (prior ~U[-8, 1])
# rd: 143.4 +1.6 -2.2 Mpc
# Chi squared (MAP): 7.23
# log likelihood (MAP): -3.61
# DOF: 9
#
# Correlation matrix
#      om          wp          wa
# om   1.          0.19835095 -0.95887650
# wp   0.19835095  1.         -0.00209782
# wa  -0.95887650 -0.00209782  1.
# ---------------------------------
