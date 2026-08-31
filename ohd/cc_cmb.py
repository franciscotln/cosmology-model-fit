from numba import njit
import numpy as np
from scipy.linalg import cho_factor
import cmb.data_planck_act_compression as cmb
from solve_triangular import solve_triangular
from y2005cc.data import get_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Onuh2 = cmb.Omnu_h2

legend, z_values, H_values, cov_matrix_cc = get_data()
L_cc = cho_factor(cov_matrix_cc, lower=True)[0]
logdet_base = 2.0 * np.log(np.diag(L_cc)).sum()
N = len(z_values)


@njit
def H_z(z, params):
    H0, Obh2, Och2 = params[0], params[1], params[2]
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2
    Onu = Onuh2 / h**2
    Or = Orh2 / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return H0 * np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


cmb.set_HZ(H_z)


@njit
def chi_squared(params, f_cc_arr):
    delta_cc = H_values - H_z(z_values, params)
    chi2_cc = solve_triangular(L_cc, f_cc_arr * delta_cc)

    delta_cm = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[1], params[2], params)
    chi2_cmb = delta_cm @ cmb.inv_cov_mat @ delta_cm
    return chi2_cc + chi2_cmb


@njit
def log_likelihood(params):
    f_cc_arr = params[3] + params[4] * z_values / (1. + z_values)
    if np.any(f_cc_arr <= 1e-4):
        return -np.inf

    logdet = logdet_base - 2.0 * np.log(f_cc_arr).sum()
    normalization = N * np.log(2 * np.pi) + logdet
    return -0.5 * (chi_squared(params, f_cc_arr) + normalization)


def main():
    from multiprocessing import Pool
    from nautilus import Sampler, Prior
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from ohd.plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(63.0, 73.0))
    prior.add_parameter("obh2", dist=(0.0210, 0.0235))
    prior.add_parameter("och2", dist=(0.05, 0.30))
    prior.add_parameter("f0", dist=(0.1, 6.0))
    prior.add_parameter("fa", dist=(-9.0, 9.0))

    with Pool(5) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    labels=["H_0", "\\Omega_b h^2", "\\Omega_c h^2", "f_{0,CC}", "f_{a,CC}"]

    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(
        (gd_samples["obh2"] + gd_samples["och2"] + Onuh2) / (gd_samples["H0"] / 100)**2,
        name="om",
        label="\\Omega_m",
    )
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    DOF = len(cmb.DISTANCE_PRIORS) + N - len(best_fit)

    f_cc_arr = best_fit[3] + best_fit[4] * z_values / (1. + z_values)

    print(f"Chi squared: {chi_squared(best_fit, f_cc_arr):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"Degs of freedom: {DOF}")

    plots.getSubplotPlotter().triangle_plot(
        gd_samples, filled=True, title_limit=1, contour_colors=["C0"], color=["C0"],
    )
    plt.show()

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_arr,
        label=f"{legend} $H_0$: {best_fit[0]:.1f} km/s/Mpc",
    )


if __name__ == "__main__":
    main()


# *****************************************************************
# CMB (ACT+Planck) + Cosmic Chronometers (CC)
# Model: Flat ΛCDM
# *****************************************************************


# ------ Overestimation factor f(z) = f0 + fa * z / (1 + z) -------
# H0: 67.66 +- 0.49 km/s/Mpc
# Ωm: 0.3111 +- 0.0069
# ωb: 0.02250 +- 0.00011
# ωc: 0.1192 +- 0.0012
# f0: 3.04 +- 0.56 (prior ~ U[0.1, 6])
# fa: -3.4 +- 1.1 (prior ~ U[-9, 9])
# Chi squared: 38.50
# Log likelihood: -149.54 (4.05 sigma significance)
# Log evidence: -164.60 (Δ logZ = 5.38 compared to no scaling)
# Degs of freedom: 37


# ------ Fixed factor f0 = 1, fa = 0 ------------------------------
# f0: 1, fa: 0 (assuming no overestimaded errors in CCH sample)
# H0: 67.61 +- 0.50 km/s/Mpc
# Ωm: 0.3118 +- 0.0071
# ωb: 0.02250 +- 0.00011
# ωc: 0.1193 +- 0.0012
# Chi squared: 16.67
# Log likelihood: -159.41
# Log evidence: -169.98
# Degs of freedom: 39
