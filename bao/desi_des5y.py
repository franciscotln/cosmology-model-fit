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

sn_legend, z_cmb_vals, z_hel_vals, mu_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_bao = cho_factor(cov_matrix_bao)

c = 299792.458  # Speed of light in km/s
H0 = 70  # Hubble constant in km/s/Mpc as per DES5Y


def h_over_h0(z, params):
    O_m, w0 = params[2], params[3]
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * evolving_de)


z_grid_sn = np.linspace(0, np.max(z_cmb_vals), num=2000)


def integral_Ez(params):
    integral_values = cumulative_trapezoid(
        1 / h_over_h0(z_grid_sn, params), z_grid_sn, initial=0
    )
    return np.interp(z_cmb_vals, z_grid_sn, integral_values)


def distance_modulus(params):
    return (
        params[0] + 25 + 5 * np.log10((1 + z_hel_vals) * (c / H0) * integral_Ez(params))
    )


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
    return H0 * h_over_h0(z, params)


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


bounds = np.array(
    [
        (-0.5, 0.5),  # delta M
        (115, 160),  # r_d
        (0.1, 0.7),  # omega_m
        (-3, 0),  # w0
    ]
)


def chi_squared(params):
    delta_sn = mu_values - distance_modulus(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = bao_data["value"] - bao_predictions(params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))
    return chi_sn + chi_bao


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
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
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [dM_50, rd_50, Om_50, w0_50]

    print(f"ΔM: {dM_50:.4f} +{(dM_84 - dM_50):.4f} -{(dM_50 - dM_16):.4f}")
    print(f"r_d: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(
        f"Degrees of freedom: {bao_data['value'].size + z_cmb_vals.size - len(best_fit)}"
    )

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb_vals,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=distance_modulus(best_fit),
        label=f"Best fit: $\Omega_m$={Om_50:.4f}",
        x_scale="log",
    )

    labels = ["$\Delta_M$", "$r_d$", "$\Omega_m$", "$w_0$"]
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
ΔM: -0.0040 +0.0065 -0.0065
r_d: 143.6251 +0.9391 -0.9443 Mpc
Ωm: 0.3104 +0.0081 -0.0079
w0: -1
wa: 0
Chi squared: 1658.9659
Degrees of freedom: 1839

==============================

Flat wCDM
ΔM: 0.0256 +0.0108 -0.0110
r_d: 141.2230 +1.1841 -1.1620 Mpc
Ωm: 0.2979 +0.0090 -0.0090
w0: -0.8709 +0.0382 -0.0388 (3.33 - 3.38 sigma)
wa: 0
Chi squared: 1648.0923
Degrees of freedom: 1838

===============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: 0.0298 +0.0117 -0.0115
r_d: 141.0102 +1.1880 -1.1929 Mpc
Ωm: 0.3053 +0.0081 -0.0080
w0: -0.8503 +0.0418 -0.0426 (3.51 - 3.58 sigma)
wa: 0
Chi squared: 1646.9805
Degrees of freedom: 1838

==============================

Flat w0waCDM
ΔM: 0.0374 +0.0138 -0.0140
r_d: 140.8937 +1.2340 -1.1907 Mpc
Ωm: 0.3199 +0.0132 -0.0167
w0: -0.7942 +0.0733 -0.0685 (2.9 sigma)
wa: -0.6713 +0.4795 -0.4728 (1.41 sigma)
Chi squared: 1646.1246
Degrees of freedom: 1837
"""
