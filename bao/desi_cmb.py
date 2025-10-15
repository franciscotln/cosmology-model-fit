from numba import njit
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from y2025BAO.data import get_data as get_bao_data
import cmb.data_desi_compression as cmb

c = cmb.c  # speed of light in km/s

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)

Orh2 = cmb.Omega_r_h2()


@njit
def Ez(z, params):
    h, Om, w0 = params[0] / 100, params[1], params[3]
    Or = Orh2 / h**2
    Ode = 1 - Om - Or

    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


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

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


def bao_predictions(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    z_drag = cmb.z_drag(wb=Obh2, wm=Om * (H0 / 100) ** 2)
    rd = cmb.rs_z(Ez, z_drag, params, H0, Obh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def chi_squared(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]

    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = np.dot(delta_cmb, np.dot(cmb.inv_cov_mat, delta_cmb))

    delta_bao = bao_data["value"] - bao_predictions(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    return chi2_cmb + chi_bao


bounds = np.array(
    [
        (55, 75),  # H0
        (0.15, 0.50),  # Ωm
        (0.021, 0.023),  # Ωb * h^2
        (-1.5, 0.0),  # w0
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
    from .plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2200 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))

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
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]

    best_fit = np.percentile(samples, 50, axis=0)

    Om_h2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_st_samples = cmb.z_star(samples[:, 2], Om_h2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Om_h2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Om_h2_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, [15.9, 50, 84.1])
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(z_drag_samples, [15.9, 50, 84.1])

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r_d: {cmb.rs_z(Ez, z_dr_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"z_d: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_predictions(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
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
Dataset: DESI DR2 2024 + (θ∗,ωb,ωbc)CMB
*******************************

Flat ΛCDM w(z) = -1
H0: 68.48 +0.29 -0.29 km/s/Mpc
Ωm: 0.2987 +0.0037 -0.0037
ωm: 0.14009 +0.00061 -0.00061
ωb: 0.02239 +0.00012 -0.00012
w0: -1
r*: 145.13 Mpc
z*: 1088.63 +0.14 -0.14
r_d: 147.72 Mpc
z_d: 1059.72 +0.26 -0.26
Chi squared: 14.03
Degs of freedom: 15

===============================

Flat wCDM w(z) = w0
H0: 68.91 +0.96 -0.92 km/s/Mpc
Ωm: 0.2956 +0.0074 -0.0073
ωm: 0.14038 +0.00087 -0.00088
ωb: 0.02236 +0.00013 -0.00013
w0: -1.019 +0.038 -0.040
r*: 145.07 Mpc
z*: 1088.67 +0.17 -0.17
r_d: 147.66 Mpc
z_d: 1059.69 +0.27 -0.27
Chi squared: 13.85
Degs of freedom: 14

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 68.21 +1.38 -1.34 km/s/Mpc
Ωm: 0.3009 +0.0115 -0.0112
ωm: 0.14000 +0.00077 -0.00078
ωb: 0.02239 +0.00012 -0.00013
w0: -0.983 +0.086 -0.087
r*: 145.15 Mpc
z*: 1088.61 +0.16 -0.16
r_d: 147.74 Mpc
z_d: 1059.73 +0.27 -0.27
Chi squared: 13.98
Degs of freedom: 14

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 63.78 +2.02 -1.90 km/s/Mpc
Ωm: 0.3489 +0.0231 -0.0220
ωm: 0.14198 +0.00092 -0.00099
ωb: 0.02223 +0.00013 -0.00013
w0: -0.453 +0.228 -0.223
wa: -1.594 +0.635 -0.682 (unconstrained)
r*: 144.74 Mpc
z*: 1088.93 +0.18 -0.19
r_d: 147.35 Mpc
z_d: 1059.52 +0.28 -0.28
Chi squared: 7.29
Degs of freedom: 13
"""
