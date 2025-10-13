from numba import njit
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_bao = cho_factor(cov_matrix_bao)

c = 299792.458  # Speed of light in km/s

grid = np.linspace(0, np.max(z_cmb), num=1000)
zhel_plus1 = 1 + z_hel


@njit
def Ez(z, params):
    Om, w0 = params[3], params[4]
    z_plus_1 = 1 + z
    cubed = z_plus_1**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


def theory_mu(params):
    y = 1 / Ez(grid, params)
    I = np.interp(z_cmb, grid, cumulative_trapezoid(y=y, x=grid, initial=0))
    return params[0] + 25 + 5 * np.log10(zhel_plus1 * c * I / params[2])


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


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

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[1]


# Planck prior
Omh2_planck = 0.1430
Omh2_planck_sigma = 0.0011


def chi_squared(params):
    Omh2 = params[3] * (params[2] / 100) ** 2
    chi2_prior = ((Omh2_planck - Omh2) / Omh2_planck_sigma) ** 2

    delta_sn = mu_values - theory_mu(params)
    chi_sn = delta_sn.dot(cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao.dot(cho_solve(cho_bao, delta_bao, check_finite=False))
    return chi_sn + chi_bao + chi2_prior


bounds = np.array(
    [
        (-0.6, 0.6),  # ΔM
        (120.0, 160.0),  # r_d
        (50.0, 90.0),  # H0
        (0.0, 1.0),  # Ωm
        (-2.0, 0.0),  # w0
    ],
    dtype=np.float64,
)


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee, corner
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from log_evidence import log_evidence
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(6) as pool:
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
        print("acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    dM_16, dM_50, dM_84 = pct[0]
    rd_16, rd_50, rd_84 = pct[1]
    H0_16, H0_50, H0_84 = pct[2]
    Om_16, Om_50, Om_84 = pct[3]
    w0_16, w0_50, w0_84 = pct[4]

    best_fit = np.array([dM_50, rd_50, H0_50, Om_50, w0_50], dtype=np.float64)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.1f}")
    print(f"Log evidence: {log_evidence(samples, log_probs, log_probability):.1f}")
    print(f"Degrees of freedom: {1 + bao_data['value'].size + sn_size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$Δ_M$", "$r_d$", "$H_0$", "$Ω_M$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
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
Flat ΛCDM
ΔM: -0.071 +0.025 -0.024 mag
r_d: 148.13 +1.23 -1.21 Mpc
H0: 67.87 +0.91 -0.89 km/s/Mpc
Ωm: 0.310 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1659.0
Log evidence: -838.4
Degrees of freedom: 1745

===============================

Flat wCDM
ΔM: 0.003 +0.037 -0.035 mag
r_d: 142.71 +2.16 -2.25 Mpc
H0: 69.29 +1.11 -1.05 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.871 +0.038 -0.038
Chi squared: 1648.1 (Δ chi2 10.9)
Log evidence: -835.3 (Δ logZ 3.1)
Degrees of freedom: 1744

===============================

Flat w0 - (1 + w0) * (((1 + z)**3 - 1) / ((1 + z)**3 + 1))
ΔM: -0.025 +0.028 -0.028 mag
r_d: 144.66 +1.57 -1.58 Mpc
H0: 68.19 +0.92 -0.89 km/s/Mpc
Ωm: 0.308 +0.008 -0.008
w0: -0.835 +0.045 -0.046
Chi squared: 1646.5 (Δ chi2 12.5)
Log evidence: -834.4 (Δ logZ 4.0)
Degrees of freedom: 1744

Flat w0 + wa * (((1 + z)**3 - 1) / ((1 + z)**3 + 1))
ΔM: -0.060 +0.046 -0.038 mag
r_d: 147.35 +2.57 -3.18 Mpc
H0: 66.88 +1.63 -1.33 km/s/Mpc
Ωm: 0.320 +0.013 -0.015
w0: -0.784 +0.071 -0.067
wa: -0.432 +0.274 -0.278
Chi squared: 1645.5 (Δ chi2 13.5)
Log evidence: -834.5 (Δ logZ 3.9)
Degrees of freedom: 1743

===============================

Flat w(z) = w0 + wa * z / (1 + z)
ΔM: -0.065 +0.046 -0.038 mag
r_d: 147.69 +2.53 -3.20 Mpc
H0: 66.73 +1.67 -1.32 km/s/Mpc
Ωm: 0.321 +0.013 -0.015
w0: -0.783 +0.072 -0.068
wa: -0.726 +0.456 -0.456
Chi squared: 1645.5 (Δ chi2 13.5)
Log evidence: -834.0 (Δ logZ 4.4)
Degrees of freedom: 1743
"""
