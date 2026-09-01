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
    y = solve_triangular(L_cc, f_cc_arr * delta_cc)
    chi2_cc = np.dot(y, y)

    delta_cm = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[1], params[2], params)
    chi2_cmb = delta_cm @ cmb.inv_cov_mat @ delta_cm
    return chi2_cc + chi2_cmb


@njit
def log_likelihood(params):
    f_cc_arr = np.exp(params[3]) * (1.0 + z_values)**(params[4])
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
    prior.add_parameter("ln_f0", dist=(-0.5, 2.5))
    prior.add_parameter("n", dist=(-4.0, 4.0))

    with Pool(5) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    labels=["H_0", "\\Omega_b h^2", "\\Omega_c h^2", "\\ln (f_0)", "n_{CCH}"]

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

    f_cc_arr = np.exp(best_fit[3]) * (1.0 + z_values)**(best_fit[4])

    print(f"Chi squared (MAP): {chi_squared(best_fit, f_cc_arr):.2f}")
    print(f"Log likelihood (MAP): {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"DOF: {DOF}")

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
# *****************************************************************


# Model: Flat ΛCDM
# ------ Fixed factor ln(f0) = 0, n = 1 ------------------------------
# ln(f0): 0, n: 1 (assuming no overestimaded errors in CCH sample)
# H0: 67.61 +- 0.50 km/s/Mpc
# Ωm: 0.3118 +- 0.0071
# ωb: 0.02250 +- 0.00011
# ωc: 0.1193 +- 0.0012
# Chi squared (MAP): 16.67
# Log likelihood (MAP): -159.41
# Log evidence: -169.98
# DOF: 39
# -----------------------------------------------------------------


# Model: Flat ΛCDM
# ------ Overestimation factor f(z) = f0 * (1 + z)^n --------------
# H0: 67.67 +- 0.49 km/s/Mpc
# Ωm: 0.3109 +- 0.0069
# ωb: 0.02250 +- 0.00011
# ωc: 0.1192 +- 0.0012
# ln(f0): 1.15 +0.26 -0.23 (prior ~ U[-0.5, 2.5])
# n: -1.35 +- 0.46 (prior ~ U[-4, 4])
# Chi squared (MAP): 39.68
# Log likelihood (MAP): -149.47 (4.06 sigma significance)
# Log evidence: -164.39 (Δ logZ = 5.59 compared to no scaling)
# DOF: 37
# -----------------------------------------------------------------
