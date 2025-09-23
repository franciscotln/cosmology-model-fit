import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data
import cmb.data_cmb_act_compression as cmb
from hubble.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions


c = cmb.c  # km/s

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

sn_grid = np.linspace(0, np.max(z_sn_vals), num=3000)


def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = cmb.Omega_r_h2() / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode * rho_de)


def mu_theory(Ez_func, params):
    H0, mag_offset = params[0], params[-1]
    integral_vals = cumulative_trapezoid(1 / Ez_func(sn_grid), sn_grid, initial=0)
    I = np.interp(z_sn_vals, sn_grid, integral_vals)
    return mag_offset + 25 + 5 * np.log10((1 + z_sn_vals) * I * c / H0)


def H_z(z, params):
    return params[0] * Ez(z, params)


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


bao_funcs = {
    "DV_over_rs": DV_z,
    "DM_over_rs": DM_z,
    "DH_over_rs": DH_z,
}


def bao_theory(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    rd = cmb.r_drag(wb=Obh2, wm=Omh2)

    return np.array([bao_funcs[q](zi, params) / rd for zi, q in zip(z, qty)])


def chi_squared(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]
    Ez_func = lambda z: Ez(z, params)

    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez_func, H0, Om, Ob_h2)
    chi2_cmb = delta @ cmb.inv_cov_mat @ delta

    delta_bao = bao_data["value"] - bao_theory(
        bao_data["z"], bao_data["quantity"], params
    )
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))

    delta_sn = mu_vals - mu_theory(Ez_func, params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    return chi2_cmb + chi_bao + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-2, 0),  # w0
        (-0.7, 0.7),  # ΔM
    ]
)


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0
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
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 10000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]
    dM_16, dM_50, dM_84 = pct[4]

    best_fit = [H0_50, Om_50, Obh2_50, w0_50, dM_50]

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    r_drag_samples = cmb.r_drag(samples[:, 2], Omh2_samples)

    one_sigma_contours = [15.9, 50, 84.1]

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_contours)
    z_star_16, z_star_50, z_star_84 = np.percentile(z_star_samples, one_sigma_contours)
    z_drag_16, z_drag_50, z_drag_84 = np.percentile(z_drag_samples, one_sigma_contours)
    r_drag_16, r_drag_50, r_drag_84 = np.percentile(r_drag_samples, one_sigma_contours)
    Ez_func = lambda z: Ez(z, best_fit)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(
        f"z*: {z_star_50:.2f} +{(z_star_84 - z_star_50):.2f} -{(z_star_50 - z_star_16):.2f}"
    )
    print(
        f"z_drag: {z_drag_50:.2f} +{(z_drag_84 - z_drag_50):.2f} -{(z_drag_50 - z_drag_16):.2f}"
    )
    print(f"r_s(z*) = {cmb.rs_z(Ez_func, z_star_50, H0_50, Obh2_50):.2f} Mpc")
    print(
        f"r_s(z_drag) = {r_drag_50:.2f} +{(r_drag_84 - r_drag_50):.2f} -{(r_drag_50 - r_drag_16):.2f} Mpc"
    )
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(Ez_func, best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "$Δ_M$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
        levels=(0.393, 0.864),
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3, lw=0.4)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
H0: 68.4 ± 0.3 km/s/Mpc
Ωm: 0.302 ± 0.004
Ωb h^2: 0.02255 ± 0.00013
Ωm h^2: 0.14097 ± 0.00064
w0: -1
ΔM: -0.129 +0.089 -0.089
z*: 1088.51 ± 0.15
z_drag: 1060.14 +0.27 -0.28
r_s(z*) = 144.81 Mpc
r_s(z_drag) = 147.33 +0.18 -0.19 Mpc
Chi squared: 46.62
Degs of freedom: 35

===============================

Flat wCDM w(z) = w0
H0: 68.0 ± 0.7 km/s/Mpc
Ωm: 0.304 ± 0.006
Ωb h^2: 0.02257 ± 0.00013
Ωm h^2: 0.14067 +0.00082 -0.00083
w0: -0.984 ± 0.028
ΔM: -0.139 ± 0.090
z*: 1088.46 ± 0.17
z_drag: 1060.18 ± 0.28
r_s(z*) = 144.87 Mpc
r_s(z_drag) = 147.38 ± 0.21 Mpc
Chi squared: 46.28 (Δ chi2 0.34)
Degs of freedom: 34

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 67.0 ± 0.8 km/s/Mpc
Ωm: 0.313 ± 0.007
Ωb h^2: 0.02260 ± 0.00013
Ωm h^2: 0.14034 ± 0.00073
w0: -0.904 +0.052 -0.053
ΔM: -0.164 ± 0.091
z*: 1088.41 ± 0.16
z_drag: 1060.22 ± 0.28
r_s(z*) = 144.94 Mpc
r_s(z_drag) = 147.44 ± 0.20 Mpc
Chi squared: 43.29 (Δ chi2 3.33)
Degs of freedom: 34

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 66.0 ± 0.8 km/s/Mpc
Ωm: 0.327 ± 0.009
Ωb h^2: 0.02242 ± 0.00013
Ωm h^2: 0.14259 +0.00091 -0.00093
w0: -0.665 +0.091 -0.088
wa: -1.082 +0.296 -0.317
ΔM: -0.171 ± 0.091
z*: 1088.76 ± 0.18
z_drag: 1059.98 ± 0.28
r_s(z*) = 144.45 Mpc
r_s(z_drag) = 147.00 ± 0.22 Mpc
Chi squared: 30.85 (Δ chi2 15.77)
Degs of freedom: 33
"""
