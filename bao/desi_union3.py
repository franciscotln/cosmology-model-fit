from pyexpat import model
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

c = 299792.458  # Speed of light in km/s
H0 = 70  # Hubble constant in km/s/Mpc


def h_over_h0(z, params):
    O_m, w0 = params[2], params[3]
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * evolving_de)


z_grid = np.linspace(0, np.max(z_sn_vals), num=2000)


def integral_of_e_z(params):
    integral_values = cumulative_trapezoid(
        1 / h_over_h0(z_grid, params), z_grid, initial=0
    )
    return np.interp(z_sn_vals, z_grid, integral_values)


def distance_modulus(params):
    delta_M = params[0]
    a0_over_ae = 1 + z_sn_vals
    comoving_distance = (c / H0) * integral_of_e_z(params)
    return delta_M + 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def H_z(z, params):
    return H0 * h_over_h0(z, params)


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": DV_z,
    "DM_over_rs": DM_z,
    "DH_over_rs": DH_z,
}


def bao_predictions(params):
    r_d = params[1]
    return np.array([(quantity_funcs[qty](z, params) / r_d) for z, _, qty in bao_data])


def plot_bao_predictions(params):
    colors = {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"}
    quantity_types = bao_data["quantity"]
    errors = np.sqrt(np.diag(bao_cov_matrix))
    r_d = params[1]
    z_smooth = np.linspace(0, max(bao_data["z"]), 100)

    plt.figure(figsize=(8, 6))
    for q in set(quantity_types):
        mask = quantity_types == q
        plt.errorbar(
            x=bao_data["z"][mask],
            y=bao_data["value"][mask],
            yerr=errors[mask],
            fmt=".",
            color=colors[q],
            label=q,
            capsize=2,
            linestyle="None",
        )
        model_smooth = []
        for z in z_smooth:
            model_smooth.append(quantity_funcs[q](z, params) / r_d)
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(bao_legend)
    plt.show()


bounds = np.array(
    [
        (-0.6, 0.6),  # delta M
        (115, 160),  # r_d
        (0.2, 0.7),  # omega_m
        (-3, 1),  # w0
    ]
)


def chi_squared(params):
    delta_sn = mu_vals - distance_modulus(params)
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
    nwalkers = 20 * ndim
    burn_in = 200
    nsteps = 10000 + burn_in
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
        [delta_M_16, delta_M_50, delta_M_84],
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [delta_M_50, rd_50, omega_50, w0_50]

    print(
        f"ΔM: {delta_M_50:.3f} +{(delta_M_84 - delta_M_50):.3f} -{(delta_M_50 - delta_M_16):.3f}"
    )
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(
        f"Ωm: {omega_50:.3f} +{(omega_84 - omega_50):.3f} -{(omega_50 - omega_16):.3f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(
        f"Degrees of freedom: {bao_data['value'].size + z_sn_vals.size - len(best_fit)}"
    )

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=distance_modulus(best_fit),
        label=f"Best fit: $\Omega_m$={omega_50:.3f}, $w_0$={w0_50:.3f}",
        x_scale="log",
    )

    labels = ["$\Delta_M$", "$r_d$", "$\Omega_m$", "$w_0$"]
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
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
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
Flat ΛCDM model
ΔM: -0.0755 +0.0880 -0.0895 mag
r_d: 144.3352 +1.0044 -1.0057 Mpc
Ωm: 0.3039 +0.0084 -0.0083
w0: -1
wa: 0
Chi squared: 38.8156
Degrees of freedom: 32

=============================

Flat wCDM
ΔM: -0.065 +0.088 -0.088 mag
r_d: 141.06 +1.57 -1.56 Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.866 +0.050 -0.051 (2.63 sigma)
wa: 0
Chi squared: 32.1547
Degrees of freedom: 31

==============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
ΔM: -0.060 +0.089 -0.089 mag
r_d: 140.44 +1.70 -1.66 Mpc
Ωm: 0.306 +0.009 -0.008
w0: -0.830 +0.059 -0.060 (2.83 - 2.88 sigma)
wa: 0
Chi squared: 30.9566
Degrees of freedom: 31

=============================

Flat w0waCDM
ΔM: -0.0455 +0.0895 -0.0899 mag
r_d: 139.0781 +1.8874 -1.8501 Mpc
Ωm: 0.3310 +0.0154 -0.0179
w0: -1.7117 +0.4692 -0.4614 (1.53 sigma)
wa: -1.0165 +0.5654 -0.5614 (1.80 sigma)
Chi squared: 28.7904
Degrees of freedom: 30
"""
