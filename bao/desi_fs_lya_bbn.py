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


@njit
def Ode_z(z, w0):
    cubic = (1.0 + z) ** 3
    return (2 * cubic / (1.0 + w0 + (1.0 - w0) * cubic)) ** 2


@njit
def H_z(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0))


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
        (-1.0, -1 / 3),  # w0
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
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
    from corner_plot import plot_corner_and_chains
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

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (w0_16, w0_50, w0_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)

    h_samples = samples[:, 0] / 100
    Omh2_samples = samples[:, 1] * h_samples**2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    rd_samples = r_drag(wb=samples[:, 2], wm=Omh2_samples)
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1])

    residuals = bao["value"] - bao_theory(bao["z"], bao_qty, best_fit)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((bao["value"] - np.mean(bao["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"Ωm: {Om_50:.4f} +{Om_84-Om_50:.4f} -{Om_50-Om_16:.4f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {len(bao)  - len(best_fit)}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    labels = ["$H_0$", "$Ω_m$", "$ω_b$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend}: $Ω_m$={Om_50:.4f}",
    )


if __name__ == "__main__":
    main()


# *********************************
# Data set: DESI DR2 BAO + FS Lya
# Gaussian prior on Obh2 from BBN
# *********************************


# Flat ΛCDM:
# H0: 68.55 +0.60 -0.58 km/s/Mpc
# ωb: 0.02218 +- 0.00055
# ωm: 0.1417 +0.0047 -0.0045
# Ωm: 0.3017 +0.0077 -0.0075
# r_d: 147.59 +- 1.47 Mpc
# Chi squared: 12.81
# Degs of freedom: 11
# R^2: 0.9987
# RMSD: 0.298
# ---------------------------------


# Flat wCDM:
# H0: 67.7 +2.0 -2.0 km/s/Mpc
# ωb: 0.02219 +0.00055 -0.00055
# ωm: 0.1389 +0.0081 -0.0082
# Ωm: 0.3022 +0.0083 -0.0081
# r_d: 148.4 +2.4 -2.3 Mpc
# w0: -0.990 +0.024 -0.025 (prior ~U[-1.5, -1/3])
# wa: 0
# Chi squared: 12.57
# Degs of freedom: 10
# R^2: 0.9988
# RMSD: 0.288
# ---------------------------------


# Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
# H0: 66.27 +1.52 -1.82 km/s/Mpc
# ωb: 0.02218 +0.00055 -0.00054
# ωm: 0.1376 +0.0053 -0.0053
# Ωm: 0.314 +0.012 -0.011
# r_d: 148.73 +1.69 -1.64 Mpc
# w0: -0.84 +0.12 -0.11 (prior ~U[-1.0, -1/3])
# wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
# Chi squared: 12.08
# Degs of freedom: 10
# R^2: 0.9990
# RMSD: 0.268
# ---------------------------------
