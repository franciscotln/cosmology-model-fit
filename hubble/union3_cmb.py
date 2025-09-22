import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
import cmb.data as cmb
from .plotting import plot_predictions


c = cmb.c  # km/s

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)

sn_grid = np.linspace(0, np.max(z_sn_vals), num=3000)


def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = cmb.Omega_r_h2() / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode * rho_de)


def mu_theory(params):
    H0, offset_mag = params[0], params[-1]
    integral_vals = cumulative_trapezoid(1 / Ez(sn_grid, params), sn_grid, initial=0)
    I = np.interp(z_sn_vals, sn_grid, integral_vals)
    dL = (1 + z_sn_vals) * I * c / H0
    return offset_mag + 25 + 5 * np.log10(dL)


def rs_z(z, params):
    H0, Ob_h2 = params[0], params[2]
    Rb = 3 * Ob_h2 / (4 * cmb.O_GAMMA_H2)

    def integrand(zp):
        denominator = Ez(zp, params) * np.sqrt(3 * (1 + Rb / (1 + zp)))
        return 1 / denominator

    I = quad(integrand, z, np.inf, limit=100)[0]
    return (c / H0) * I


def DA_z(z, params):
    integral = quad(lambda zp: 1 / Ez(zp, params), 0, z)[0]
    return (c / params[0]) * integral / (1 + z)


def cmb_distances(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]
    Om_h2 = Om * (H0 / 100) ** 2
    zstar = cmb.z_star(Ob_h2, Om_h2)
    rs_star = rs_z(zstar, params)
    DA_star = DA_z(zstar, params)

    lA = np.pi * (1 + zstar) * DA_star / rs_star
    R = np.sqrt(Om) * H0 * (1 + zstar) * DA_star / c
    return np.array([R, lA, Ob_h2])


def chi_squared(params):
    delta = cmb.DISTANCE_PRIORS - cmb_distances(params)
    chi2_cmb = delta @ cmb.inv_cov_mat @ delta

    delta_sn = mu_vals - mu_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.45),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-1.7, -0.3),  # w0
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

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, [15.9, 50, 84.1])
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(z_drag_samples, [15.9, 50, 84.1])

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(
        f"z_drag: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}"
    )
    print(f"r_s(z*) = {rs_z(z_st_50, best_fit):.2f} Mpc")
    print(f"r_s(z_drag) = {rs_z(z_dr_50, best_fit):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
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
*******************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*******************************

Flat ΛCDM w(z) = -1
H0: 67.13 +0.59 -0.58 km/s/Mpc
Ωm: 0.321 ± 0.008
Ωm h^2: 0.14446 +0.00122 -0.00123
Ωb h^2: 0.02231 ± 0.00015
w0: -1
ΔM: -0.167 +0.090 -0.091
z*: 1089.00 +0.22 -0.21
z_drag: 1059.88 +0.29 -0.30
r_s(z*) = 144.03 Mpc
r_s(z_drag) = 146.61 Mpc
Chi squared: 26.0
Degrees of freedom: 21

===============================

Flat wCDM w(z) = w0
H0: 65.30 +1.25 -1.23 km/s/Mpc
Ωm: 0.337 +0.014 -0.013
Ωm h^2: 0.14383 ± 0.00129
Ωb h^2: 0.02236 ± 0.00015
w0: -0.927 ± 0.044
ΔM: -0.216 +0.095 -0.096
z*: 1088.91 ± 0.22
z_drag: 1059.94 ± 0.30
r_s(z*) = 144.17 Mpc
r_s(z_drag) = 146.73 Mpc
Chi squared: 23.2 (Δ chi2 2.8)
Degrees of freedom: 20

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)^3)
H0: 65.41 ± 1.08 km/s/Mpc
Ωm: 0.336 ± 0.012
Ωm h^2: 0.14380 +0.00127 -0.00128
Ωb h^2: 0.02236 ± 0.00015
w0: -0.878 ± 0.067
ΔM: -0.209 +0.093 -0.094
z*: 1088.90 ± 0.22
z_drag: 1059.94 ± 0.30
r_s(z*) = 144.18 Mpc
r_s(z_drag) = 146.74 Mpc
Chi squared: 22.6 (Δ chi2 3.4)
Degrees of freedom: 20

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 66.58 +1.34 -1.38 km/s/Mpc
Ωm: 0.325 +0.015 -0.013
Ωm h^2: 0.14396 +0.00131 -0.00129
Ωb h^2: 0.02235 ± 0.00015
w0: -0.689 +0.154 -0.159
wa: -1.122 +0.732 -0.734
ΔM: -0.154 +0.100 -0.101
z*: 1088.93 ± 0.22
z_drag: 1059.93 ± 0.30
r_s(z*) = 144.13 Mpc
r_s(z_drag) = 146.69 Mpc
Chi squared: 21.80 (Δ chi2 4.2)
Degrees of freedom: 19
"""
