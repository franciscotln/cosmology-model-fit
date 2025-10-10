from numba import njit
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from y2025BAO.data import get_data
import y2024BBN.prior_lcdm_cyril as bbn
from cmb.data_desi_compression import c, z_drag, rs_z, Omega_r_h2

legend, data, cov_matrix = get_data()
cho = cho_factor(cov_matrix)

Orh2 = Omega_r_h2()


def rd(params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    h = H0 / 100
    Omh2 = Om * h**2
    zdrag = z_drag(wb=Obh2, wm=Omh2)
    return rs_z(Ez, zdrag, params, H0=H0, Ob_h2=Obh2)


@njit
def Ez(z, params):
    h, Om, w0 = params[0] / 100, params[1], params[3]
    Or = Orh2 / h**2
    OL = 1 - Om - Or
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(Or * one_plus_z**4 + Om * cubic + OL * rho_de)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    result = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        zp = z[i]
        x = np.linspace(0, zp, num=max(250, int(250 * zp)))
        y = DH_z(x, params)
        result[i] = np.trapz(y=y, x=x)
    return result


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

quantities = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd(params)


bounds = np.array(
    [
        (55, 75),  # H0
        (0.17, 0.50),  # Ωm
        (0.016, 0.030),  # Ωb h^2
        (-2.0, 0.0),  # w0
    ],
    dtype=np.float64,
)


def chi_squared(params):
    bbn_delta = (bbn.Obh2 - params[2]) / bbn.Obh2_sigma
    bbn_chi2 = bbn_delta**2

    delta = data["value"] - bao_theory(data["z"], quantities, params)
    bao_chi2 = delta.dot(cho_solve(cho, delta, check_finite=False))
    return bao_chi2 + bbn_chi2


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return 0.0


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee, corner
    from multiprocessing import Pool
    import matplotlib.pyplot as plt
    from .plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))
    np.random.seed(42)

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.30),
                (emcee.moves.DEMove(), 0.56),
                (emcee.moves.DESnookerMove(), 0.14),
            ],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50], dtype=np.float64)

    h_samples = samples[:, 0] / 100
    Omh2_samples = samples[:, 1] * h_samples**2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    z_d_samples = z_drag(wb=samples[:, 2], wm=Omh2_samples)
    z_d_16, z_d_50, z_d_84 = np.percentile(z_d_samples, [15.9, 50, 84.1])

    residuals = data["value"] - bao_theory(data["z"], quantities, best_fit)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"Ωm: {Om_50:.4f} +{Om_84-Om_50:.4f} -{Om_50-Om_16:.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {rd(best_fit):.2f} Mpc")
    print(f"z_d: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {1 + data['value'].size  - len(best_fit)}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend}: $H_0$={H0_50:.2f} km/s/Mpc, $Ω_m$={Om_50:.4f}",
    )
    labels = ["$H_0$", "$Ω_m$", "$ω_b$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()

    plt.figure(figsize=(16, 1.5 * ndim))
    for n in range(ndim):
        plt.subplot2grid((ndim, 1), (n, 0))
        plt.plot(chains_samples[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


"""
*******************************
Dataset: DESI DR2 2025
*******************************

Flat ΛCDM:
H0: 68.45 +0.44 -0.43 km/s/Mpc
ωb: 0.02190 +0.00024 -0.00024
ωm: 0.13943 +0.00476 -0.00463
Ωm: 0.2976 +0.0086 -0.0084
w0: -1
r_d: 148.31 Mpc
z_d: 1058.58 +0.70 -0.70
Chi squared: 10.29
Degs of freedom: 11
R^2: 0.9987
RMSD: 0.305

===============================

Flat wCDM:
H0: 66.25 +2.17 -2.16 km/s/Mpc
ωb: 0.02190 +0.00025 -0.00025
ωm: 0.13068 +0.00970 -0.00990
Ωm: 0.2970 +0.0089 -0.0088
w0: -0.918 +0.076 -0.079
r_d: 150.83 Mpc
z_d: 1057.91 +0.96 -0.99
Chi squared: 9.05
Degs of freedom: 10
R^2: 0.9989
RMSD: 0.281

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 65.65 +2.18 -2.05 km/s/Mpc
ωb: 0.02190 +0.00025 -0.00025
ωm: 0.13271 +0.00703 -0.00669
Ωm: 0.3076 +0.0118 -0.0115
w0: -0.834 +0.121 -0.127
r_d: 150.20 Mpc
z_d: 1058.08 +0.81 -0.81
Chi squared: 8.42
Degs of freedom: 10
R^2: 0.9990
RMSD: 0.266
"""
