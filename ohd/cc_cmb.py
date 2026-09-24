from numba import njit
import numpy as np
import cmb.data_spt_planck_act_compression as cmb
from solve_triangular import solve_triangular
from y2005cc.data import get_data, method

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Onuh2 = cmb.Omnu_h2

legend, z_values, H_values, diag_stat, cov_mat_sys = get_data(split_sys=True)
N_cc = z_values.size
non_d = method != "D"


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
def chi2_cmb(params):
    delta_cm = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[1], params[2], params)
    chi2_cmb = delta_cm @ cmb.inv_cov_mat @ delta_cm
    return chi2_cmb


@njit
def chi2_cc(params, L_cc):
    delta_cc = H_values - H_z(z_values, params)
    y = solve_triangular(L_cc, delta_cc)
    return np.dot(y, y)


@njit
def chi_squared(params, L_cc):
    return chi2_cc(params, L_cc) + chi2_cmb(params)


@njit
def get_fz(params):
    z_pivot = 0.62
    fp, n = np.exp(params[3]), params[4]
    fz = np.full_like(z_values, fp)
    fz[non_d] *= ((1.0 + z_values[non_d]) / (1.0 + z_pivot)) ** n
    return fz


@njit
def log_likelihood(params):
    cov_mat = cov_mat_sys + np.diag(diag_stat**2 * get_fz(params)**2)
    L_cc = np.linalg.cholesky(cov_mat)
    logdet = 2 * np.sum(np.log(np.diag(L_cc)))
    normalization = N_cc * np.log(2 * np.pi) + logdet

    return -0.5 * (chi_squared(params, L_cc) + normalization)


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
    prior.add_parameter("ln_fp", dist=(-1.5, 0.5))
    prior.add_parameter("n", dist=(-2.0, 4.0))

    with Pool(5) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    labels=["H_0", "Ω_b h^2", "Ω_c h^2", "ln(f_{piv})", "n_{cc}"]

    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(
        (gd_samples["obh2"] + gd_samples["och2"] + Onuh2) / (gd_samples["H0"] / 100)**2,
        name="om",
        label="\\Omega_m",
    )
    gd_samples.addDerived(np.exp(gd_samples["ln_fp"]), name="fp", label="f_{piv}")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    DOF = len(cmb.DISTANCE_PRIORS) + N_cc - len(best_fit)

    fz_cc = get_fz(best_fit)
    cov_mat = np.diag(diag_stat**2 * fz_cc**2) + cov_mat_sys
    L_cc = np.linalg.cholesky(cov_mat)

    print(f"Chi squared (MAP): {chi_squared(best_fit, L_cc):.2f}")
    print(f"Log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"DOF: {DOF}")

    plots.getSubplotPlotter().triangle_plot(
        gd_samples,
        params=prior.keys,
        filled=True,
        title_limit=1,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_mat_sys) + diag_stat**2),
        label=f"{legend} $H_0$: {best_fit[0]:.1f} km/s/Mpc",
        err_scaling=1 / fz_cc,
    )


if __name__ == "__main__":
    main()


# *****************************************************************
# CMB(θ*, ωb, ωm) SPT + Planck PR4 + ACT DR6 + Cosmic Chronometers
# *****************************************************************


# Model: Flat ΛCDM
# ------ Fixed factor ln(fp) = 0, n = 1 -----------------------------
# ln(fp): 0, n: 1 (assuming no overestimated errors in CCH sample)
# H0: 67.19 +- 0.38 km/s/Mpc
# Ωm: 0.3175 +- 0.0055
# ωb = 0.022399 +- 0.000095
# ωc = 0.12026 +- 0.00093
#
# Chi squared (MAP): 20.30
# Log likelihood (MAP): -159.80
# Log evidence: -170.83
# DOF: 39
# -------------------------------------------------------------------


# Model: Flat ΛCDM
# --- Overestimation factor f(z) = fp * [(1 + z) * H(z) / ((1 + z_piv) * H(z_piv))]^n ---
# H0 = 67.28 ± 0.38 km/s/Mpc
# Ωb h^2 = 0.022412 ± 0.000095
# Ωc h^2 = 0.12007 ± 0.00093
#
# n_cc = 1.19 ± 0.50 (prior ~ U[-2, 4])
# ln(fp) = -0.47 +0.14 -0.16 (prior ~ U[-1.5, 0.5])
# fp = 0.631 +0.073 -0.110
#
# Chi squared (MAP): 37.65
# Log likelihood (MAP): -153.85
# Log evidence: -168.14 (Δ logZ = 2.69 compared to no scaling)
# DOF: 37
# -------------------------------------------------------------------
