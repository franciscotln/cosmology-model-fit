import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data import get_data
from hubble.plotting import plot_predictions as plot_sn_predictions

legend, z_cmb, z_hel, mb_vals, cov_matrix_sn = get_data()
bao_legend, data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

c = 299792.458  # Speed of light in km/s


def Ez(z, O_m, w0=-1):
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * evolving_de)


grid = np.linspace(0, np.max(z_cmb), num=2000)


def integral_Ez(params):
    x = grid
    y = 1 / Ez(grid, *params[3:])
    return np.interp(z_cmb, x, cumulative_trapezoid(y=y, x=x, initial=0))


def apparent_mag(params):
    M, H0 = params[1], params[2]
    dL = (1 + z_hel) * (c / H0) * integral_Ez(params)
    return M + 25 + 5 * np.log10(dL)


def plot_bao_predictions(params):
    errors = np.sqrt(np.diag(bao_cov_matrix))
    colors = {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"}
    r_d = params[0]
    z_smooth = np.linspace(0, max(data["z"]), 100)

    plt.figure(figsize=(8, 6))
    for q in set(data["quantity"]):
        mask = data["quantity"] == q
        plt.errorbar(
            x=data["z"][mask],
            y=data["value"][mask],
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
                model_smooth.append((DH_z(z, params)) / r_d)
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


quantity_funcs = {
    "DV_over_rs": DV_z,
    "DM_over_rs": DM_z,
    "DH_over_rs": DH_z,
}


def bao_predictions(params):
    r_d = params[0]
    return np.array([(quantity_funcs[qty](z, params) / r_d) for z, _, qty in data])


bounds = np.array(
    [
        (120, 160),  # r_d
        (-20, -19),  # M
        (50, 80),  # H0
        (0.2, 0.7),  # Ωm
        (-2, 0),  # w0
    ]
)


def chi_squared(params):
    delta_sn = mb_vals - apparent_mag(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = data["value"] - bao_predictions(params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))
    return chi_sn + chi_bao


# Prior for r_d from Planck 2018 https://arxiv.org/abs/1807.06209
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -0.5 * ((147.09 - params[0]) / 0.26) ** 2
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
        [rd_16, rd_50, rd_84],
        [M_16, M_50, M_84],
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, M_50, H0_50, Om_50, w0_50]

    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"M0: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {data['z'].size + z_cmb.size - len(best_fit)}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_cmb,
        y=mb_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag(best_fit),
        label=f"Best fit: $\Omega_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$r_d$", "$M_0$", "$H_0$", "$\Omega_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM
r_d: 147.09 +0.26 -0.26
M0: -19.402 +0.013 -0.013
H0: 68.66 +0.48 -0.47
Ωm: 0.304 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1416.14
Degrees of freedom: 1599

====================

Flat wCDM
r_d: 147.09 +0.26 -0.26 Mpc
M0: -19.416 +0.015 -0.015 mag
H0: 67.83 +0.61 -0.60 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.914 +0.040 -0.040 (2.15 sigma)
wa: 0
Chi squared: 1411.54
Degrees of freedom: 1598

====================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
r_d: 147.09 +0.26 -0.26 Mpc
M0: -19.416 +0.015 -0.015 mag
H0: 67.79 +0.61 -0.61 km/s/Mpc
Ωm: 0.302 +0.008 -0.008
w0: -0.901 +0.044 -0.044 (2.25 sigma)
wa: 0
Chi squared: 1411.31
Degrees of freedom: 1598

====================

Flat w0waCDM
r_d: 147.08 +0.26 -0.26 Mpc
M0: -19.415 +0.015 -0.015 mag
H0: 67.80 +0.61 -0.61 km/s/Mpc
Ωm: 0.303 +0.016 -0.024
w0: -0.892 +0.062 -0.056 (1.74 - 1.93 sigma)
wa: 0.162 +0.495 - 0.454 (0.33 - 0.36 sigma)
Chi squared: 1411.36
Degrees of freedom: 1597
"""
