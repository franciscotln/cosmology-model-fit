from numba import njit
import numpy as np
import cmb.data_spt_planck_act_compression as cmb
from solve_triangular import solve_triangular
from y2005cc.data import get_data, method

c = cmb.c_km_per_s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

legend, z_values, H_values, diag_stat, cov_mat_sys = get_data(split_sys=True)
N_cc = z_values.size


@njit
def H_z(z, params):
    h, Obh2, Och2 = params[0], params[1], params[2]
    zp1 = 1.0 + z

    radiation_term = Orh2 * zp1**4
    matter_term = (Obh2 + Och2) * zp1**3
    neutrino_term = Omnuh2 * cmb.Omnu_z(z)
    lambda_term = h**2 - Orh2 - Omnuh2 - Obh2 - Och2

    return 100 * np.sqrt(radiation_term + matter_term + neutrino_term + lambda_term)


cmb.set_HZ(H_z)


@njit
def chi2_cc(params, L_cc):
    delta_cc = H_values - H_z(z_values, params)
    y = solve_triangular(L_cc, delta_cc)
    return np.dot(y, y)


@njit
def chi_squared(params, L_cc):
    return chi2_cc(params, L_cc) + cmb.chi2(params[1], params[2], params)


method_f = method == "F"
z_pivot = 1.198
shape = np.ones_like(z_values, dtype=np.float64)
shape[method_f] = ((1 + z_values[method_f]) / (1 + z_pivot))**4
# statistical error in H is proportional to (1+z) * H^2


@njit
def get_fz(params):
    fp, n = np.exp(params[3]), params[4]
    return fp * shape**n


@njit
def log_likelihood_jit(params):
    cov_mat = cov_mat_sys + np.diag(diag_stat**2 * get_fz(params)**2)
    L_cc = np.linalg.cholesky(cov_mat)
    logdet = 2 * np.sum(np.log(np.diag(L_cc)))
    normalization = N_cc * np.log(2 * np.pi) + logdet

    return -0.5 * (chi_squared(params, L_cc) + normalization + cmb.prob_norm)


def log_likelihood(params):
    return log_likelihood_jit(params)


def main():
    from multiprocessing import Pool
    from nautilus import Sampler, Prior
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from ohd.plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("h", dist=(0.63, 0.73))
    prior.add_parameter("obh2", dist=(0.0210, 0.0235))
    prior.add_parameter("och2", dist=(0.05, 0.30))
    prior.add_parameter("ln_fp", dist=(-2.0, 1.0))
    prior.add_parameter("n", dist=(-2.5, 2.5))

    with Pool(5) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    labels=["h", "Ω_b h^2", "Ω_c h^2", "ln(f_{piv})", "n_{cc}"]

    gd_samples = MCSamples(samples=samples, weights=weights, names=prior.keys, labels=labels)
    gd_samples.addDerived(
        (gd_samples["obh2"] + gd_samples["och2"] + Omnuh2) / gd_samples["h"]**2,
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
        roots=gd_samples,
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
        label=f"{legend} $H_0$: {100 * best_fit[0]:.1f} km/s/Mpc",
        method=method,
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
# H0: 67.20 ± 0.38 km/s/Mpc
# Ωm: 0.3174 ± 0.0055
# ωb = 0.022399 ± 0.000095
# ωc = 0.12027 ± 0.00093
#
# Chi squared (MAP): 20.30
# Log likelihood (MAP): -137.88
# Log evidence: -148.91
# DOF: 39
# -------------------------------------------------------------------


# Model: Flat ΛCDM
# --- Overestimation factor f(z) = fp * [(1 + z) * H(z)^2 / ((1 + z_piv) * H(z_piv)^2)]^n ---
# H0 = 67.28 ± 0.38 km/s/Mpc
# Ωm = 0.3163 ± 0.0055
# Ωb h^2 = 0.022411 ± 0.000094
# Ωc h^2 = 0.12008 ± 0.00093
#
# ln(fp) = -0.44 +0.26 -0.23 (prior ~ U[-2, 1])
# n_cc = 1.02 +0.32 -0.57 (prior ~ U[-2.5, 2.5])
#
# Chi squared (MAP): 41.09
# Log likelihood (MAP): -130.01
# Log evidence: -144.22 (Δ logZ = 4.69 compared to no scaling)
# DOF: 37
# -------------------------------------------------------------------
