from numba import njit
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from y2023union3.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from sn.plotting import plot_predictions as plot_sn_predictions
from cosmic_chronometers.plot_predictions import plot_cc_predictions
from .plot_predictions import plot_bao_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, sn_mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(cov_matrix_bao)
cho_cc = cho_factor(cov_matrix_cc)
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = 299792.458  # Speed of light in km/s

z_grid = np.linspace(0, np.max(z_sn_vals), num=1000)
one_plus_z_sn = 1 + z_sn_vals


@njit
def Ez(z, params):
    O_m, w0, wa = params[4], params[5], params[6]
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    # rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    rho_de = one_plus_z ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / one_plus_z)
    return np.sqrt(O_m * cubed + (1 - O_m) * rho_de)


def mu_theory(params):
    dM, h0 = params[1], params[2]
    y = 1 / Ez(z_grid, params)
    integral_values = cumulative_trapezoid(y=y, x=z_grid, initial=0)
    I = np.interp(z_sn_vals, z_grid, integral_values)
    return dM + 25 + 5 * np.log10(one_plus_z_sn * (c / h0) * I)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


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
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[3]


bounds = np.array(
    [
        (0.1, 1.5),  # f_cc
        (-0.7, 0.7),  # ΔM
        (55, 80),  # H0
        (125, 170),  # r_d
        (0.2, 0.7),  # Ωm
        (-2.0, 1.0),  # w0
        (-3.0, 2.0),
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f_cc = params[0]
    delta_sn = sn_mu_vals - mu_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = f_cc**-2 * np.dot(
        delta_cc, cho_solve(cho_cc, delta_cc, check_finite=False)
    )
    return chi_sn + chi_bao + chi_cc


@njit
def log_prior(params):
    if params[5] + params[6] > 0:
        return -np.inf
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc + 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee, corner
    import matplotlib.pyplot as plt
    from multiprocessing import Pool

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

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
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    print("Correlation matrix:")
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [dM_16, dM_50, dM_84],
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([f_cc_50, dM_50, h0_50, rd_50, Om_50, w0_50, wa_50], dtype=np.float64)

    Omh2_samples = samples[:, 4] * samples[:, 2] ** 2 / 100**2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    deg_of_freedom = (
        z_sn_vals.size + bao_data["value"].size + z_cc_vals.size - len(best_fit)
    )

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"wa: {wa_50:.3f} +{(wa_84 - wa_50):.3f} -{(wa_50 - wa_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=sn_mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f}, $\Omega_m$={Om_50:.3f}",
        x_scale="log",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) * f_cc_50,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )

    labels = ["$f_{CCH}$", "ΔM", "$H_0$", "$r_d$", "Ωm", "$w_0$", "$w_a$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
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
Flat ΛCDM: w(z) = -1
f_cc: 0.70 +0.10 -0.08
ΔM: -0.118 +0.114 -0.117 mag
H0: 68.7 +2.3 -2.4 km/s/Mpc
r_d: 147.0 +5.1 -4.7 Mpc
Ωm: 0.304 +0.008 -0.008
ωm: 0.1436 +0.0097 -0.0096
w0: -1
Chi squared: 69.1
Degrees of freedom: 63

==============================

Flat wCDM: w(z) = w0
f_cc: 0.70 +0.10 -0.08
ΔM: -0.158 +0.115 -0.118 mag
H0: 67.1 +2.4 -2.4 km/s/Mpc
r_d: 147.3 +5.1 -4.9 Mpc
Ωm: 0.298 +0.009 -0.009
ωm: 0.1342 +0.0103 -0.0097
w0: -0.870 +0.051 -0.051
Chi squared: 62.6 (Δ chi2 = 6.5 compared to ΛCDM)
Degrees of freedom: 62

==============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 0.71 +0.10 -0.08
ΔM: -0.165 +0.117 -0.119 mag
H0: 66.7 +2.4 -2.4 km/s/Mpc
r_d: 147.2 +5.2 -4.8 Mpc
Ωm: 0.310 +0.009 -0.008
ωm: 0.1378 +0.0099 -0.0095
w0: -0.811 +0.065 -0.066
Chi squared: 60.7 (Δ chi2 = 8.4 compared to ΛCDM)
Degrees of freedom: 62

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
f_cc: 0.71 +0.10 -0.08
ΔM: -0.167 +0.116 -0.117 mag
H0: 66.3 +2.4 -2.4 km/s/Mpc
r_d: 147.1 +5.2 -4.8 Mpc
Ωm: 0.329 +0.016 -0.018
ωm: 0.1441 +0.0115 -0.0113
w0: -0.722 +0.113 -0.106
wa: -0.901 +0.554 -0.561
Chi squared: 59.1 (Δ chi2 = 10.0 compared to ΛCDM)
Degrees of freedom: 61
Correlation matrix:
[[ 1.      -0.01544 -0.02956  0.03362  0.01976  0.04016 -0.03784]
 [-0.01544  1.       0.65363 -0.63626 -0.08695 -0.09796  0.03614]
 [-0.02956  0.65363  1.      -0.93058 -0.25339 -0.28492  0.18545]
 [ 0.03362 -0.63626 -0.93058  1.       0.00176 -0.01002  0.02232]
 [ 0.01976 -0.08695 -0.25339  0.00176  1.       0.77201 -0.8719 ]
 [ 0.04016 -0.09796 -0.28492 -0.01002  0.77201  1.      -0.89085]
 [-0.03784  0.03614  0.18545  0.02232 -0.8719  -0.89085  1.     ]]
"""
