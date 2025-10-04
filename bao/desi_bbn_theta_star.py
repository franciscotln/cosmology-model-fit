from numba import njit
import numpy as np
import emcee
import corner
from scipy.constants import c as c0
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
import cmb.data_chen_compression as cmb
import y2024BBN.prior_lcdm as bbn
from .plot_predictions import plot_bao_predictions

c = c0 / 1000  # speed of light in km/s
Or_h2 = cmb.Omega_r_h2()

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)

# arXiv:2503.14738v2
theta_100 = 1.04110
theta_100_err = 0.00053


@njit
def Ez(z, params):
    H0, Om, exp_w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or

    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + np.log(exp_w0)))
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


@njit
def bao_predictions(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    rd = cmb.r_drag(wb=Obh2, wm=Omh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def theta_100_theory(params):
    H0, Om, wb = params[0], params[1], params[2]
    wm = Om * (H0 / 100) ** 2
    zstar = cmb.z_star(wb, wm)
    rs_star = cmb.rs_z(Ez, zstar, params, H0, wb)
    DA_star = cmb.DA_z(Ez, zstar, params, H0)
    return 100 * rs_star / ((1 + zstar) * DA_star)


def chi_squared(params):
    delta_bbn = bbn.Obh2 - params[2]
    chi2_bbn = (delta_bbn / bbn.Obh2_sigma) ** 2

    delta_theta_100 = theta_100 - theta_100_theory(params)
    chi2_theta_100 = (delta_theta_100 / theta_100_err) ** 2

    delta_bao = bao_data["value"] - bao_predictions(bao_data["z"], quantities, params)
    chi_bao = delta_bao.dot(cho_solve(cho_bao, delta_bao, check_finite=False))

    return chi2_bbn + chi2_theta_100 + chi_bao


bounds = np.array(
    [
        (55, 75),  # H0
        (0.20, 0.50),  # Ωm
        (0.020, 0.025),  # Ωb * h^2
        (0.15, 0.70),  # e^w0
    ],
    dtype=np.float64,
)


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.log(params[3])  # flat prior in w0
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
    nsteps = 2000 + burn_in
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
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    exp_w0_16, exp_w0_50, exp_w0_84 = pct[3]

    best_fit = np.array([H0_50, Om_50, Obh2_50, exp_w0_50], dtype=np.float64)

    w0_samples = np.log(samples[:, 3])
    w0_16, w0_50, w0_84 = np.percentile(w0_samples, [15.9, 50, 84.1])
    Om_h2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_st_samples = cmb.z_star(samples[:, 2], Om_h2_samples)
    z_dr_samples = cmb.z_drag(samples[:, 2], Om_h2_samples)
    r_drag_samples = cmb.r_drag(samples[:, 2], Om_h2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Om_h2_samples, [15.9, 50, 84.1])
    r_d_16, r_d_50, r_d_84 = np.percentile(r_drag_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, [15.9, 50, 84.1])
    z_d_16, z_d_50, z_d_84 = np.percentile(z_dr_samples, [15.9, 50, 84.1])

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r_s(z*) = {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(
        f"r_s(z_drag) = {r_d_50:.2f} +{(r_d_84 - r_d_50):.2f} -{(r_d_50 - r_d_16):.2f} Mpc"
    )
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_predictions(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$e^{w_0}$"]
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
Dataset: DESI DR2 2024 + θ∗ + BBN
*******************************

Flat ΛCDM w(z) = -1
H0: 68.60 +0.46 -0.47 km/s/Mpc
Ωm: 0.295 +0.004 -0.004
Ωm h^2: 0.13898 +0.00119 -0.00118
Ωb h^2: 0.02216 +0.00052 -0.00053
w0: -0.752 +0.504 -0.503
z*: 1088.80 +0.55 -0.52
z_drag: 1059.13 +1.24 -1.26
r_s(z*) = 145.53 Mpc
r_s(z_drag) = 148.19 +0.71 -0.71 Mpc
Chi squared: 10.35
Degs of freedom: 12

===============================

Flat wCDM w(z) = w0
H0: 67.75 +1.10 -1.06 km/s/Mpc
Ωm: 0.301 +0.007 -0.008
Ωm h^2: 0.13794 +0.00171 -0.00172
Ωb h^2: 0.02223 +0.00054 -0.00054
w0: -0.960 +0.046 -0.048
z*: 1088.66 +0.58 -0.56
z_drag: 1059.20 +1.26 -1.29
r_s(z*) = 145.77 Mpc
r_s(z_drag) = 148.42 +0.77 -0.77 Mpc
Chi squared: 9.59
Degs of freedom: 11

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.96 +1.48 -1.43 km/s/Mpc
Ωm: 0.308 +0.012 -0.012
Ωm h^2: 0.13811 +0.00143 -0.00141
Ωb h^2: 0.02224 +0.00054 -0.00054
w0: -0.887 +0.096 -0.098
z*: 1088.66 +0.57 -0.54
z_drag: 1059.24 +1.26 -1.29
r_s(z*) = 145.71 Mpc
r_s(z_drag) = 148.35 +0.74 -0.73 Mpc
Degs of freedom: 11
"""
