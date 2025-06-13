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
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * evolving_de)


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
    print(
        np.array2string(
            np.corrcoef(samples, rowvar=False), precision=5, suppress_small=True
        )
    )

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
ΔM: -0.121 +0.115 -0.116 mag
H0: 68.6 +2.3 -2.3 km/s/Mpc
r_d: 147.2 +5.0 -4.7 Mpc
Ωm: 0.305 +0.008 -0.008
w0: -1
Chi squared: 70.13
Degrees of freedom: 62

==============================

Flat wCDM: w(z) = w0
f_cc: 1.45 +0.19 -0.18
ΔM: -0.162 +0.116 -0.117 mag
H0: 67.0 +2.4 -2.4 km/s/Mpc
r_d: 147.5 +5.1 -4.8 Mpc
Ωm: 0.299 +0.009 -0.009
w0: -0.871 +0.050 -0.051 (2.53 - 2.58 sigma)
Chi squared: 63.34
Degrees of freedom: 61

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
f_cc: 1.44 +0.19 -0.18
ΔM: -0.165 +0.115 -0.117 mag
H0: 66.7 +2.3 -2.3 km/s/Mpc
r_d: 147.4 +5.0 -4.7 Mpc
Ωm: 0.307 +0.009 -0.008
w0: -0.837 +0.059 -0.060 (2.72 - 2.76 sigma)
Chi squared: 62.20
Degrees of freedom: 61
Correlation matrix:
[ 1.       0.01128  0.0014  -0.0052   0.01718 -0.03021]
[ 0.01128  1.       0.64993 -0.62987 -0.13955 -0.14164]
[ 0.0014   0.64993  1.      -0.94412 -0.23606 -0.29018]
[-0.0052  -0.62987 -0.94412  1.       0.05335  0.01724]
[ 0.01718 -0.13955 -0.23606  0.05335  1.       0.10322]
[-0.03021 -0.14164 -0.29018  0.01724  0.10322  1.     ]
"""
