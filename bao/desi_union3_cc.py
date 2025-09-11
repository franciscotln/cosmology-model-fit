import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, sn_mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(cov_matrix_bao)
inv_cov_cc = np.linalg.inv(cov_matrix_cc)
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = 299792.458  # Speed of light in km/s


def Ez(z, O_m, w0):
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubed + (1 - O_m) * rho_de)


z_grid = np.linspace(0, np.max(z_sn_vals), num=3000)


def integral_Ez(params):
    y = 1 / Ez(z_grid, *params[4:])
    integral_values = cumulative_trapezoid(y=y, x=z_grid, initial=0)
    return np.interp(z_sn_vals, z_grid, integral_values)


def mu_theory(params):
    dM, h0 = params[1], params[2]
    dL = (1 + z_sn_vals) * (c / h0) * integral_Ez(params)
    return dM + 25 + 5 * np.log10(dL)


def plot_bao_predictions(params):
    observed_values = bao_data["value"]
    z_values = bao_data["z"]
    quantity_types = bao_data["quantity"]
    errors = np.sqrt(np.diag(cov_matrix_bao))

    unique_quantities = set(quantity_types)
    colors = {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"}

    f_cc, h0, r_d = params[0], params[2], params[3]
    z_smooth = np.linspace(0, max(z_values), 100)
    plt.figure(figsize=(8, 6))
    for q in unique_quantities:
        mask = quantity_types == q
        plt.errorbar(
            x=z_values[mask],
            y=observed_values[mask],
            yerr=errors[mask],
            fmt=".",
            color=colors[q],
            label=q,
            capsize=2,
            linestyle="None",
        )
        model_smooth = []
        for z in z_smooth:
            if q == "DV_over_rs":
                model_smooth.append(DV_z(z, params) / r_d)
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, params) / r_d)
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, params)) / r_d)
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(bao_legend)
    plt.show()

    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=np.sqrt(np.diag(cov_matrix_cc)) / f_cc,
        fmt=".",
        color="blue",
        alpha=0.4,
        label="CCH data",
        capsize=2,
        linestyle="None",
    )
    plt.plot(z_smooth, H_z(z_smooth, params), color="green", alpha=0.5, label="Model")
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z_cc_vals) + 0.2)
    plt.legend()
    plt.title(f"{cc_legend}: $H_0$={h0:.1f} km/s/Mpc")
    plt.show()


def H_z(z, params):
    return params[2] * Ez(z, *params[4:])


def DM_z(z, params):
    return quad(lambda zp: c / H_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


def bao_theory(params):
    r_d = params[3]
    predictions = []
    for z, _, quantity in bao_data:
        if quantity == "DV_over_rs":
            predictions.append(DV_z(z, params) / r_d)
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params) / r_d)
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params)) / r_d)
    return np.array(predictions)


bounds = np.array(
    [
        (0.4, 2.5),  # f_cc
        (-0.7, 0.7),  # ΔM
        (55, 80),  # H0
        (125, 170),  # r_d
        (0.2, 0.7),  # Ωm
        (-1.6, -0.4),  # w0
    ]
)


def chi_squared(params):
    f_cc = params[0]
    delta_sn = sn_mu_vals - mu_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = bao_data["value"] - bao_theory(params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, np.dot(inv_cov_cc * f_cc**2, delta_cc))
    return chi_sn + chi_bao + chi_cc


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 10 * ndim
    burn_in = 500
    nsteps = 20000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
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
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [f_cc_50, dM_50, h0_50, rd_50, Om_50, w0_50]

    deg_of_freedom = (
        z_sn_vals.size + bao_data["value"].size + z_cc_vals.size - len(best_fit)
    )

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=sn_mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $H_0$={h0_50:.2f}, $\Omega_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$f_{CCH}$", "ΔM", "$H_0$", "$r_d$", "Ωm", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
f_cc: 1.46 +0.19 -0.18
ΔM: -0.119 +0.114 -0.115 mag
H0: 68.7 +2.3 -2.3 km/s/Mpc
r_d: 147.0 +5.0 -4.7 Mpc
Ωm: 0.304 +0.008 -0.008
w0: -1
Chi squared: 70.40
Degrees of freedom: 62
Correlation matrix:
[[ 1.       0.00058  0.00247 -0.01492  0.02370]
 [ 0.00058  1.       0.64222 -0.63226 -0.13443]
 [ 0.00247  0.64222  1.      -0.97895 -0.23273]
 [-0.01492 -0.63226 -0.97895  1.       0.05164]
 [ 0.02370 -0.134426 -0.23273 0.05164  1.     ]]

==============================

Flat wCDM: w(z) = w0
f_cc: 1.45 +0.19 -0.18
ΔM: -0.156 +0.116 -0.117 mag
H0: 67.2 +2.4 -2.3 km/s/Mpc
r_d: 147.1 +5.0 -4.7 Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.872 +0.051 -0.051
Chi squared: 63.75
Degrees of freedom: 61
Correlation matrix:
[[ 1.       0.00673  0.00888 -0.01645  0.02135 -0.01429]
 [ 0.00673  1.       0.6506  -0.62745 -0.06394 -0.14188]
 [ 0.00888  0.6506   1.      -0.94849 -0.09146 -0.26603]
 [-0.01645 -0.62745 -0.94849  1.       0.03388  0.00958]
 [ 0.02135 -0.06394 -0.09146  0.03388  1.      -0.35729]
 [-0.01429 -0.14188 -0.26603  0.00958 -0.35729  1.     ]]

==============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 1.45 +0.19 -0.18
ΔM: -0.163 +0.116 -0.117 mag
H0: 66.7 +2.4 -2.4 km/s/Mpc
r_d: 147.1 +5.0 -4.7 Mpc
Ωm: 0.310 +0.008 -0.008
w0: -0.811 +0.065 -0.067 (2.82 - 2.91 sigma from -1)
Chi squared: 61.93
Degrees of freedom: 61
Correlation matrix:
[[ 1.       0.00197  0.00862 -0.01231  0.01508 -0.02921]
 [ 0.00197  1.       0.64869 -0.62696 -0.16224 -0.14468]
 [ 0.00862  0.64869  1.      -0.9407  -0.2728  -0.30101]
 [-0.01231 -0.62696 -0.9407   1.       0.05428  0.01311]
 [ 0.01508 -0.16224 -0.2728   0.05428  1.       0.24993]
 [-0.02921 -0.14468 -0.30101  0.01311  0.24993  1.     ]]

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
f_cc: 1.45 +0.18 -0.18
ΔM: -0.164 +0.114 -0.116 mag
H0: 66.4 +2.4 -2.4 km/s/Mpc
r_d: 146.9 +5.0 -4.7 Mpc
Ωm: 0.329 +0.016 -0.019
w0: -0.721 +0.113 -0.108 (2.47 - 2.58 sigma from -1)
wa: -0.910 +0.561 -0.548 (1.62 - 1.66 sigma from 0)
Chi squared: 60.50
Degrees of freedom: 60
Correlation matrix:
[[ 1.       0.0066   0.01034 -0.01067 -0.01846 -0.04633  0.04206]
 [ 0.0066   1.       0.64111 -0.62225 -0.09876 -0.10326  0.05165]
 [ 0.01034  0.64111  1.      -0.92479 -0.26684 -0.29577  0.20613]
 [-0.01067 -0.62225 -0.92479  1.       0.00734 -0.01326  0.01192]
 [-0.01846 -0.09876 -0.26684  0.00734  1.       0.76572 -0.87183]
 [-0.04633 -0.10326 -0.29577 -0.01326  0.76572  1.      -0.89101]
 [ 0.04206  0.05165  0.20613  0.01192 -0.87183 -0.89101  1.     ]]
"""
