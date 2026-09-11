from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import block_diag
from interpolator import interp_pchip, interp_hermite
from solve_triangular import solve_triangular
from y2025BAO.data_fs_lya import get_data as get_desi_data
from y2024DESBAO.data import get_data as get_des_data
from y20116dFBAO.data import get_data as get_6dF_data

c = c0 / 1000  # Speed of light in km/s
rd = 147.09  # Mpc, fixed

legend_desi, data_desi, cov_desi = get_desi_data()
legend_des, data_des, cov_des = get_des_data()
legend_6dF, data_6dF, cov_6dF = get_6dF_data()

data = np.concatenate((data_desi, data_des, data_6dF))
cov_matrix = block_diag(cov_desi, cov_des, cov_6dF)

Lcc = np.linalg.cholesky(cov_matrix)

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]

# ----- PARAMS -----
names = ["h", "om", "wp", "wa"]
labels = ["h", "Ω_m", "w_p", "w_a"]
bounds = np.array(
    [
        (0.50, 0.80),  # h
        (0.1, 0.6),  # Ωm
        (-3.0, 1.0),  # wp
        (-8.0, 8.0),  # wa
    ]
)
# ------------------

z_piv = 0.38


@njit
def Ode_z(z, wp, wa):
    zp1 = 1 + z
    return zp1**(3 * (1 + wp + wa / (1 + z_piv))) * np.exp(-3 * wa * z / zp1)


@njit
def H_z(z, params):
    h, om, wp, wa = params
    return 100 * h * np.sqrt(om * (1 + z) ** 3 + (1 - om) * Ode_z(z, wp, wa))


@njit
def bao_theory(z, qty, theta):
    DH_grid = c / H_z(z_grid, theta)
    dh = (DH_grid[:-1] + DH_grid[1:]) / 2
    DM_grid = np.zeros(z_grid.size, dtype=np.float64)
    DM_grid[1:] = np.cumsum(dz * dh)

    DH = interp_pchip(z, x=z_grid, y=DH_grid)
    DM = interp_hermite(z, x=z_grid, y=DM_grid, y_prime=DH_grid)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    FAP_mask = qty == 3

    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH[DH_mask] / rd
    results[DM_mask] = DM[DM_mask] / rd
    results[DV_mask] = (z[DV_mask] * DH[DV_mask] * DM[DV_mask] ** 2) ** (1 / 3) / rd
    results[FAP_mask] = DM[FAP_mask] / DH[FAP_mask]
    return results


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
bao_qty = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def chi_squared(theta):
    delta_bao = data["value"] - bao_theory(data["z"], bao_qty, theta)
    y = solve_triangular(Lcc, delta_bao)
    return np.dot(y, y)


normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    if params[2] + (params[3] / (1 + z_piv)) >= 0.0:
        return -np.inf
    
    return normalization


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
    from multiprocessing import Pool
    import emcee
    from getdist import MCSamples, plots
    import matplotlib.pyplot as plt
    from bao.plot_predictions import plot_bao_predictions, plot_bao_residuals
    from log_evidence import log_evidence

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 1000
    nsteps = 6000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, moves=moves, pool=pool
        )
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    samples = sampler.get_chain(discard=burn_in, flat=False)
    flat_log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=False)

    log_evd = log_evidence(flat_samples, flat_log_probs, log_probability, bounds)

    # reshape for getdist
    chain_list = np.moveaxis(samples, 1, 0)
    loglike_list = np.moveaxis(log_probs, 1, 0)

    gd_samples = MCSamples(
        samples=chain_list,
        loglikes=-loglike_list,
        names=names,
        labels=labels,
        label='DESI + DES6Y + 6dF'
    )
    gd_samples.addDerived(gd_samples["h"] * rd, name="hrd", label="h \\cdot r_d")

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = flat_samples[np.argmax(flat_log_probs)]
    DOF = len(data) - len(best_fit)

    residuals = data["value"] - bao_theory(data["z"], bao_qty, best_fit)
    chi2 = chi_squared(best_fit)

    print(f"Chi2: {chi2:.2f}")
    print(f"DOF: {DOF}")
    print(f"Chi2/DOF: {chi2 / DOF:.2f}")
    print(f"Log evidence: {log_evd:.2f}")

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=["hrd", "om", "wp", "wa"],
        title_limit=1,
        contour_colors=["C0"],
        filled=True,
    )
    plt.show()
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend_des} + {legend_desi}",
    )
    plot_bao_residuals(data, residuals, np.sqrt(np.diag(cov_matrix)))


if __name__ == "__main__":
    main()


# *********************************
# Data sets:
# - DESI DR2 (BAO+FS Lya)
# - DES6Y BAO
# - 6dF BAO
# *********************************


# ----------- Flat ΛCDM -----------
# Ωm = 0.3009 ± 0.0076
# h x r_d = 101.26 ± 0.66 Mpc
# Chi2: 13.59
# DOF: 14
# Chi2/DOF: 13.59 / 14 ≈ 0.97
# Log evidence: -14.00
# ---------------------------------


# ----------- Flat wCDM -----------
# Ωm = 0.3011 ± 0.0080
# h x r_d = 100.8 ± 1.7 Mpc
# w0 = -0.978 ± 0.072 (prior U[-2, 0])
# Chi2: 13.47
# DOF: 13
# Chi2/DOF: 13.47 / 13 ≈ 1.04
# Log evidence: -16.35
# ---------------------------------


# ---------- Flat w0waCDM ---------
# z_pivot = 0.38 (wp and wa uncorrelated)
# wp + wa/(1 + z_pivot) < 0 enforced in the likelihood
#
# Ωm = 0.366 +0.040 -0.036
# wp = -0.987 ± 0.070 (0.2 sigma from -1) (prior U[-3, 1])
# wa = -2.2 ± 1.3 (1.7 sigma from 0) (prior U[-8, 8])
# h x r_d = 94.1 +3.7 -4.4 Mpc
# Chi2: 10.22
# DOF: 12
# Chi2/DOF: 10.22 / 12 ≈ 0.85
# Log evidence: -17.41
# ---------------------------------
