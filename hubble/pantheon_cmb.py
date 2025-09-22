import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2022pantheonSHOES.data import get_data
import cmb.data as cmb
from hubble.plotting import plot_predictions as plot_sn_predictions

c = cmb.c  # Speed of light in km/s

sn_legend, z_cmb, z_hel, mb_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)

sn_grid = np.linspace(0, np.max(z_cmb), num=3000)


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
    return np.interp(z_cmb, sn_grid, integral_values)


def apparent_mag(params):
    H0, M = params[0], params[-1]
    dL = (1 + z_hel) * (c / H0) * integral_Ez(params)
    return M + 25 + 5 * np.log10(dL)


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

    delta_sn = mb_values - apparent_mag(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.15, 0.55),  # Ωm
        (0.020, 0.025),  # Ωb * h^2
        (-1.5, -0.5),  # w0
        (-20, -19),  # M
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
    nwalkers = 10 * ndim
    burn_in = 500
    nsteps = 8000 + burn_in
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

    samples = sampler.get_chain(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]
    M_16, M_50, M_84 = pct[5]

    best_fit = [H0_50, Om_50, Obh2_50, w0_50, M_50]

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_star_samples, [15.9, 50, 84.1])
    z_dr_16, z_dr_50, z_dr_84 = np.percentile(z_drag_samples, [15.9, 50, 84.1])

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(
        f"z_drag: {z_dr_50:.2f} +{(z_dr_84 - z_dr_50):.2f} -{(z_dr_50 - z_dr_16):.2f}"
    )
    print(f"r_s(z*) = {rs_z(z_st_50, best_fit):.2f} Mpc")
    print(f"r_s(z_drag) = {rs_z(z_dr_50, best_fit):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mb_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag(best_fit),
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "M"]
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


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
H0: 67.21 ± 0.55 km/s/Mpc
Ωm: 0.319 ± 0.008
Ωb h^2: 0.02233 +0.00014 -0.00014
Ωm h^2: 0.14429 +0.00118 -0.00116
w0: -1
M: -19.444 ± 0.016
z*: 1088.98 +0.21 -0.20
z_drag: 1059.89 +0.29 -0.30
r_s(z*) = 144.07 Mpc
r_s(z_drag) = 146.64 Mpc
Chi squared: 1403.48
Degrees of freedom: 1591

===============================

Flat wCDM w(z) = w0
H0: 66.68 +0.82 -0.83 km/s/Mpc
Ωm: 0.324 ± 0.009
Ωb h^2: 0.02236 ± 0.00015
Ωm h^2: 0.14386 +0.00126 -0.00127
w0: -0.975 ± 0.030
M: -19.456 ± 0.021
z*: 1088.92 ± 0.22
z_drag: 1059.93 ± 0.30
r_s(z*) = 144.16 Mpc
r_s(z_drag) = 146.73 Mpc
Chi squared: 1402.76
Degrees of freedom: 1590

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.77 +0.74 -0.71 km/s/Mpc
Ωm: 0.323 +0.009 -0.008
Ωb h^2: 0.02236 ± 0.00015
Ωm h^2: 0.14384 +0.00129 -0.00128
w0: -0.962 ± 0.043
M: -19.452 ± 0.018
z*: 1088.91 ± 0.22
z_drag: 1059.93 ± 0.30
r_s(z*) = 144.17 Mpc
r_s(z_drag) = 146.73 Mpc
Chi squared: 1402.67
Degrees of freedom: 1590

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.22 +1.29 -1.43 km/s/Mpc
Ωm: 0.319 +0.014 -0.013
Ωb h^2: 0.02236 ± 0.00015
Ωm h^2: 0.14391 +0.00129 -0.00126
w0: -0.919 +0.107 -0.114
wa: -0.287 +0.551 -0.563
M: -19.434 +0.043 -0.049
z*: 1088.92 ± 0.22
z_drag: 1059.93 ± 0.30
r_s(z*) = 144.13 Mpc
r_s(z_drag) = 146.69 Mpc
Chi squared: 1402.58
Degrees of freedom: 1589
"""
