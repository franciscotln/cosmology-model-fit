from numba import njit
import numpy as np
from interpolator import interp_hermite
from y2025BAO.data import get_data
import y2024BBN.prior_lcdm_schoneberg as bbn
from cmb.data_planck_compression import r_drag, c

legend, data, cov_matrix = get_data()
inv_cov = np.linalg.inv(cov_matrix)

z_grid = np.linspace(0, np.max(data["z"]) + 0.1, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    cubic = (1.0 + z) ** 3
    return (2 * cubic / (1.0 + w0 + (1.0 - w0) * cubic)) ** 2


@njit
def Ez(z, params):
    Om = params[1]
    cubic = (1.0 + z) ** 3
    return np.sqrt(Om * cubic + (1.0 - Om) * Ode_z(z, w0=params[3]))


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = DH_z(z_grid, params)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dz * dy)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    h, Om, Obh2 = params[0] / 100, params[1], params[2]
    rd = r_drag(Obh2, Om * h**2)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


@njit
def chi_squared(params):
    bbn_delta = bbn.Obh2 - params[2]
    bbn_chi2 = (bbn_delta / bbn.Obh2_sigma) ** 2

    delta = data["value"] - bao_theory(data["z"], quantities, params)
    bao_chi2 = delta @ inv_cov @ delta
    return bao_chi2 + bbn_chi2


bounds = np.array(
    [
        (55, 75),  # H0
        (0.17, 0.50),  # Ωm
        (0.016, 0.030),  # Ωb h^2
        (-1.0, -1 / 3),  # w0
    ],
    dtype=np.float64,
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
    from bao.plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 100
    burn_in = 500
    nsteps = 5000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.25),
        (emcee.moves.DEMove(), 0.75),
    ]

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

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

    residuals = data["value"] - bao_theory(data["z"], quantities, best_fit)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"Ωm: {Om_50:.4f} +{Om_84-Om_50:.4f} -{Om_50-Om_16:.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {1 + len(data['z'])  - len(best_fit)}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    labels = ["$H_0$", "$Ω_m$", "$ω_b$", "$w_0$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend}: $Ω_m$={Om_50:.4f}",
    )


if __name__ == "__main__":
    main()


"""
*******************************
Dataset: DESI DR2 2025
*******************************

Flat ΛCDM:
H0: 68.58 +0.60 -0.59 km/s/Mpc
ωb: 0.02219 +0.00055 -0.00055
ωm: 0.14007 +0.00518 -0.00492
Ωm: 0.2979 +0.0087 -0.0085
w0: -1
wa: 0
r_d: 148.03 +1.57 -1.59 Mpc
Chi squared: 10.27
Degs of freedom: 11
R^2: 0.9987
RMSD: 0.305

===============================

Flat wCDM:
H0: 66.38 +2.22 -2.20 km/s/Mpc
ωb: 0.02218 +0.00055 -0.00055
ωm: 0.13138 +0.00984 -0.01004
Ωm: 0.2973 +0.0089 -0.0087
w0: -0.919 +0.076 -0.080
wa: 0
r_d: 150.44 +3.06 -2.84 Mpc
Chi squared: 9.05
Degs of freedom: 10
R^2: 0.9989
RMSD: 0.281

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
H0: 65.33 +1.91 -2.03 km/s/Mpc
ωb: 0.02218 +0.00055 -0.00055
ωm: 0.13308 +0.00627 -0.00634
Ωm: 0.3123 +0.0122 -0.0115
w0: -0.770 +0.132 -0.130
wa: d w(z)/dz at z=0 = -(3/2) * (1 - w0^2)
r_d: 149.97 +1.99 -1.92 Mpc
Chi squared: 8.29
Degs of freedom: 10
R^2: 0.9991
RMSD: 0.262
"""
