import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_bao = cho_factor(cov_matrix_bao)

c = 299792.458  # Speed of light in km/s


def Ez(z, Om, w0=-1, wa=0):
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(Om * one_plus_z**3 + (1 - Om) * evolving_de)


grid = np.linspace(0, np.max(z_cmb), num=2000)


def integral_Ez(params):
    x = grid
    y = 1 / Ez(grid, *params[3:])
    return np.interp(z_cmb, grid, cumulative_trapezoid(y=y, x=x, initial=0))


def theory_mu(params):
    dM, H0 = params[0], params[2]
    return dM + 25 + 5 * np.log10((1 + z_hel) * (c / H0) * integral_Ez(params))


def plot_bao_predictions(params):
    colors = {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"}
    errors = np.sqrt(np.diag(cov_matrix_bao))
    z_smooth = np.linspace(0, max(bao_data["z"]), 100)

    plt.figure(figsize=(8, 6))
    for q in set(bao_data["quantity"]):
        quantity_mask = bao_data["quantity"] == q
        plt.errorbar(
            x=bao_data["z"][quantity_mask],
            y=bao_data["value"][quantity_mask],
            yerr=errors[quantity_mask],
            fmt=".",
            color=colors[q],
            label=q,
            capsize=2,
            linestyle="None",
        )
        model_smooth = [bao_quantity_funcs[q](z, params) / params[1] for z in z_smooth]
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(bao_legend)
    plt.show()


def H_z(z, params):
    return params[2] * Ez(z, *params[3:])


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


bao_quantity_funcs = {
    "DV_over_rs": DV_z,
    "DM_over_rs": DM_z,
    "DH_over_rs": DH_z,
}


def bao_predictions(params):
    return np.array(
        [
            bao_quantity_funcs[qty](z, params) / params[1]
            for z, qty in zip(bao_data["z"], bao_data["quantity"])
        ]
    )


def chi_squared(params):
    delta_sn = mu_values - theory_mu(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = bao_data["value"] - bao_predictions(params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))
    return chi_sn + chi_bao


bounds = np.array(
    [
        (-0.5, 0.5),  # delta M
        (115, 160),  # r_d
        (50, 80),  # H0
        (0.1, 0.7),  # Ωm
        (-3, 0),  # w0
    ]
)


# Prior from Planck 2018 https://arxiv.org/abs/1807.06209
# Ωm x ​h^2 = 0.14237 ± 0.00135
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        Om_x_h2 = params[3] * (params[2] / 100) ** 2
        return -0.5 * ((0.14237 - Om_x_h2) / 0.00135) ** 2
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

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [dM_16, dM_50, dM_84],
        [rd_16, rd_50, rd_84],
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [dM_50, rd_50, H0_50, Om_50, w0_50]

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {bao_data['value'].size + z_cmb.size - len(best_fit)}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"Model: $\Omega_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$\Delta_M$", "$r_d$", "$H_0$", "$\Omega_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    if ndim == 1:
        axes = [axes]
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
Flat ΛCDM
ΔM: -0.076 +0.025 -0.026 mag
r_d: 148.48 +1.32 -1.29 Mpc
H0: 67.71 +0.93 -0.93 km/s/Mpc
Ωm: 0.310 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1658.97
Degrees of freedom: 1838

==============================

Flat wCDM
ΔM: -0.002 +0.038 -0.036 mag
r_d: 143.03 +2.23 -2.33 Mpc
H0: 69.13 +1.13 -1.08 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.871 +0.038 -0.039 (3.31 - 3.39 sigma)
wa: 0
Chi squared: 1648.09
Degrees of freedom: 1837

===============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.023 +0.030 -0.030 mag
r_d: 144.52 +1.72 -1.74 Mpc
H0: 68.31 +0.95 -0.95 km/s/Mpc
Ωm: 0.305 +0.008 -0.008
w0: -0.850 +0.041 -0.042 (3.57 - 3.66 sigma)
wa: 0
Chi squared: 1646.98
Degrees of freedom: 1837

===============================

Flat w0waCDM
ΔM: -0.071 +0.047 -0.038 mag
r_d: 148.08 +2.59 -3.25 Mpc
H0: 66.54 +1.67 -1.34 km/s/Mpc
Ωm: 0.322 +0.013 -0.015
w0: -0.783 +0.074 -0.068
wa: -0.729 +0.454 -0.458
Chi squared: 1645.45
Degrees of freedom: 1836
"""
