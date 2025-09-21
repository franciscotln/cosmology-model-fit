import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data
import cmb.data as cmb
from hubble.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions


c = 299792.458  # km/s

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


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(sn_grid, params), sn_grid, initial=0)
    return np.interp(z_sn_vals, sn_grid, integral_values)


def distance_modulus(params):
    H0, offset_mag = params[0], params[-1]
    dL = (1 + z_sn_vals) * integral_Ez(params) * c / H0
    return offset_mag + 25 + 5 * np.log10(dL)


def rs_z(z, params):
    H0, Ob_h2 = params[0], params[2]
    Rb = 3 * Ob_h2 / (4 * cmb.O_GAMMA_H2)

    def integrand(zp):
        denom = Ez(zp, params) * np.sqrt(3 * (1 + Rb / (1 + zp)))
        return 1 / denom

    z_lower = z
    z_upper = np.inf
    I = quad(integrand, z_lower, z_upper, limit=100)[0]
    return (c / H0) * I


def DA_z(z, params):
    return DM_z(z, params) / (1 + z)


def cmb_distances(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]
    Om_h2 = Om * (H0 / 100) ** 2
    zstar = cmb.z_star(Ob_h2, Om_h2)
    rs_star = rs_z(zstar, params)
    DA_star = DA_z(zstar, params)

    lA = np.pi * (1 + zstar) * DA_star / rs_star
    R = np.sqrt(Om) * H0 * (1 + zstar) * DA_star / c
    return np.array([R, lA, Ob_h2])


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
    rd = rs_z(cmb.z_drag(Obh2, Omh2), params)

    return np.array([bao_funcs[q](zi, params) / rd for zi, q in zip(z, qty)])


def chi_squared(params):
    delta = cmb.DISTANCE_PRIORS - cmb_distances(params)
    chi2_cmb = delta @ cmb.inv_cov_mat @ delta

    delta_bao = bao_data["value"] - bao_theory(
        bao_data["z"], bao_data["quantity"], params
    )
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))

    delta_sn = mu_vals - distance_modulus(params)
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
    (H0_16, H0_50, H0_84) = pct[0]
    (Om_16, Om_50, Om_84) = pct[1]
    (Obh2_16, Obh2_50, Obh2_84) = pct[2]
    (w0_16, w0_50, w0_84) = pct[3]
    (dM_16, dM_50, dM_84) = pct[4]

    best_fit = [H0_50, Om_50, Obh2_50, w0_50, dM_50]

    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    z_st = cmb.z_star(Obh2_50, Omh2_50)
    z_dr = cmb.z_drag(Obh2_50, Omh2_50)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"z*: {z_st:.2f}")
    print(f"z_drag: {z_dr:.2f}")
    print(f"r_s(z*) = {rs_z(z_st, best_fit):.2f} Mpc")
    print(f"r_s(z_drag) = {rs_z(z_dr, best_fit):.2f} Mpc")
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
        y_model=distance_modulus(best_fit),
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
H0: 68.52 ± 0.30 km/s/Mpc
Ωm: 0.302 ± 0.004
Ωb h^2: 0.02253 ± 0.00012
w0: -1
wa: 0
ΔM: -0.125 ± 0.089
z*: 1088.58
z_drag: 1060.15
r_s(z*) = 144.64 Mpc
r_s(z_drag) = 147.17 Mpc
Chi squared: 43.73
Degs of freedom: 35

===============================

Flat wCDM w(z) = w0
H0: 67.96 +0.71 -0.70 km/s/Mpc
Ωm: 0.306 ± 0.006
Ωb h^2: 0.02256 ± 0.00013
w0: -0.975 ± 0.029
wa: 0
ΔM: -0.139 +0.091 -0.090
z*: 1088.51
z_drag: 1060.20
r_s(z*) = 144.75 Mpc
r_s(z_drag) = 147.26 Mpc
Chi squared: 42.88 (Δ chi2 0.85)
Degs of freedom: 34

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.97 +0.82 -0.80 km/s/Mpc
Ωm: 0.314 ± 0.007
Ωb h^2: 0.02259 ± 0.00013
w0: -0.893 ± 0.053
wa: -(1 + w0) = -0.107 ± 0.053
ΔM: -0.161 ± 0.091
z*: 1088.47
z_drag: 1060.22
r_s(z*) = 144.80 Mpc
r_s(z_drag) = 147.31 Mpc
Chi squared: 39.68 (Δ chi2 4.05)
Degs of freedom: 34

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 66.17 +0.85 -0.84 km/s/Mpc
Ωm: 0.327 +0.009 -0.009
Ωb h^2: 0.02241 +0.00014 -0.00013
w0: -0.677 +0.093 -0.090
wa: -1.031 +0.305 -0.328
ΔM: -0.167 +0.091 -0.090
z*: 1088.81
z_drag: 1059.99
r_s(z*) = 144.31 Mpc
r_s(z_drag) = 146.87 Mpc
Chi squared: 29.95 (Δ chi2 13.78)
Degs of freedom: 33
"""
