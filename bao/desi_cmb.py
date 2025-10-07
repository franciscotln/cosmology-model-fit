from numba import njit
import numpy as np
import emcee
import corner
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
import cmb.data_chen_compression as cmb
from .plot_predictions import plot_bao_predictions

c = cmb.c  # speed of light in km/s

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)

Orh2 = cmb.Omega_r_h2()


@njit
def Ez(z, params):
    H0, Om, w0 = params[1], params[2], params[4]
    h = H0 / 100
    Or = Orh2 / h**2
    Ode = 1 - Om - Or

    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


@njit
def H_z(z, params):
    return params[1] * Ez(z, params)


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


@njit
def bao_predictions(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[0]


def chi_squared(params):
    H0, Om, Ob_h2 = params[1], params[2], params[3]

    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = np.dot(delta_cmb, np.dot(cmb.inv_cov_mat, delta_cmb))

    delta_bao = bao_data["value"] - bao_predictions(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    return chi2_cmb + chi_bao


bounds = np.array(
    [
        (120, 160),  # r_d
        (55, 75),  # H0
        (0.25, 0.45),  # Ωm
        (0.021, 0.023),  # Ωb * h^2
        (-1.5, 0),  # w0
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
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2200 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

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

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    r_d_16, r_d_50, r_d_84 = pct[0]
    H0_16, H0_50, H0_84 = pct[1]
    Om_16, Om_50, Om_84 = pct[2]
    Obh2_16, Obh2_50, Obh2_84 = pct[3]
    w0_16, w0_50, w0_84 = pct[4]

    best_fit = np.array([r_d_50, H0_50, Om_50, Obh2_50, w0_50], dtype=np.float64)

    Om_h2_samples = samples[:, 2] * (samples[:, 1] / 100) ** 2
    z_st_samples = cmb.z_star(samples[:, 3], Om_h2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Om_h2_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, [15.9, 50, 84.1])

    print(f"r_d: {r_d_50:.2f} +{(r_d_84 - r_d_50):.2f} -{(r_d_50 - r_d_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_predictions(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    labels = ["$r_d$", "$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$"]
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
r_d: 148.41 +0.51 -0.52 Mpc
H0: 68.01 +0.43 -0.44 km/s/Mpc
Ωm: 0.3051 +0.0058 -0.0057
ωm: 0.14112 +0.00092 -0.00090
ωb: 0.02231 +0.00013 -0.00013
w0: -1
r*: 144.90 Mpc
z*: 1091.83 +0.23 -0.23
Chi squared: 11.76
Degs of freedom: 14

CHEN:
r_d: 148.00 +0.52 -0.52 Mpc
H0: 68.07 +0.44 -0.45 km/s/Mpc
Ωm: 0.3076 +0.0060 -0.0058
ωm: 0.14251 +0.00093 -0.00092
ωb: 0.02246 +0.00013 -0.00013
w0: -1
r*: 144.46 Mpc
z*: 1088.71 +0.17 -0.17
Chi squared: 12.83

===============================

Flat wCDM w(z) = w0
r_d: 148.36 +0.52 -0.51 Mpc
H0: 68.50 +1.00 -0.96 km/s/Mpc
Ωm: 0.3016 +0.0085 -0.0084
ωm: 0.14153 +0.00112 -0.00113
ωb: 0.02228 +0.00014 -0.00014
w0: -1.023 +0.038 -0.041
r*: 144.82 Mpc
z*: 1091.92 +0.27 -0.26
Chi squared: 11.51
Degs of freedom: 13

CHEN:
r_d: 147.95 +0.53 -0.52 Mpc
H0: 68.83 +1.02 -0.98 km/s/Mpc
Ωm: 0.3021 +0.0087 -0.0086
ωm: 0.14316 +0.00116 -0.00117
ωb: 0.02241 +0.00014 -0.00014
w0: -1.036 +0.040 -0.042
r*: 144.33 Mpc
z*: 1088.81 +0.20 -0.21
Chi squared: 12.17

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
r_d: 148.42 +0.51 -0.52 Mpc
H0: 68.03 +1.42 -1.35 km/s/Mpc
Ωm: 0.3050 +0.0121 -0.0120
ωm: 0.14116 +0.00108 -0.00108
ωb: 0.02231 +0.00014 -0.00014
w0: -1.002 +0.089 -0.091
r*: 144.89 Mpc
z*: 1091.84 +0.26 -0.25
Chi squared: 11.77
Degs of freedom: 13

CHEN:
r_d: 148.00 +0.52 -0.53 Mpc
H0: 68.45 +1.46 -1.38 km/s/Mpc
Ωm: 0.3045 +0.0123 -0.0122
ωm: 0.14269 +0.00111 -0.00111
ωb: 0.02245 +0.00014 -0.00014
w0: -1.027 +0.090 -0.095
r*: 144.42 Mpc
z*: 1088.73 +0.20 -0.20
Chi squared: 12.77

===============================

Flat w(z) = w0 + wa * z / (1 + z)
r_d: 147.71 +0.57 -0.55 Mpc
H0: 63.97 +2.11 -2.08 km/s/Mpc
Ωm: 0.3482 +0.0250 -0.0227
ωm: 0.14252 +0.00116 -0.00117
ωb: 0.02219 +0.00014 -0.00014
w0: -0.50 +0.26 -0.23
wa: -1.49 +0.67 -0.75
r*: 144.60 Mpc
z*: 1092.12 +0.27 -0.28
Chi squared: 6.81
Degs of freedom: 12

CHEN:
r_d: 147.30 +0.55 -0.55 Mpc
H0: 63.88 +2.11 -1.93 km/s/Mpc
Ωm: 0.3535 +0.0235 -0.0233
ωm: 0.14424 +0.00120 -0.00122
ωb: 0.02233 +0.00014 -0.00014
w0: -0.455 +0.240 -0.241
wa: -1.658 +0.698 -0.714
r*: 144.08 Mpc
z*: 1088.97 +0.21 -0.21
Chi squared: 6.45
"""
