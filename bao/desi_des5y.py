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
from .plot_predictions import plot_bao_predictions

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_bao = cho_factor(cov_matrix_bao)

c = 299792.458  # Speed of light in km/s


def Ez(z, Om, w0):
    z_plus_1 = 1 + z
    rho_de = (2 * z_plus_1**3 / (1 + z_plus_1**3)) ** (2 * (1 + w0))
    return np.sqrt(Om * z_plus_1**3 + (1 - Om) * rho_de)


grid = np.linspace(0, np.max(z_cmb), num=2000)


def integral_Ez(params):
    x = grid
    y = 1 / Ez(grid, *params[2:])
    return np.interp(z_cmb, grid, cumulative_trapezoid(y=y, x=x, initial=0))


def theory_mu(params):
    dL = (1 + z_hel) * c * integral_Ez(params)
    return params[0] + 25 + 5 * np.log10(dL)


def H_z(z, params):
    return Ez(z, *params[2:])


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


bao_quantity_funcs = {
    "DV_over_rs": lambda z, params: DV_z(z, params) / (params[1] * 100),
    "DM_over_rs": lambda z, params: DM_z(z, params) / (params[1] * 100),
    "DH_over_rs": lambda z, params: DH_z(z, params) / (params[1] * 100),
}


def theory_predictions(z, qty, params):
    return np.array([(bao_quantity_funcs[qty](z, params)) for z, qty in zip(z, qty)])


def chi_squared(params):
    delta_sn = mu_values - theory_mu(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = bao_data["value"] - theory_predictions(
        bao_data["z"], bao_data["quantity"], params
    )
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))
    return chi_sn + chi_bao


bounds = np.array(
    [
        (-10, -8.5),  # ΔM
        (90, 110),  # r_d * h
        (0.1, 0.7),  # Ωm
        (-2, 0),  # w0
    ]
)


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0
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

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d * h: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {bao_data['value'].size + z_cmb.size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_predictions(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$Δ_M$", "$r_d x h$", "$Ω_M$", "$w_0$"]
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
ΔM: -9.230 +0.006 -0.006 mag
r_d * h: 100.55 +0.67 -0.66 Mpc
Ωm: 0.310 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1658.97
Degrees of freedom: 1839

==============================

Flat wCDM
ΔM: -9.200 +0.011 -0.011 mag
r_d * h: 98.85 +0.83 -0.81 Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.872 +0.039 -0.038 (3.28 - 3.37 sigma)
wa: 0
Chi squared: 1648.11 (Δ chi2 10.86)
Degrees of freedom: 1838

===============================

Flat w0 - (1 + w0) * (((1 + z)**3 - 1) / ((1 + z)**3 + 1))
ΔM: -9.193 +0.012 -0.012 mag
r_d * h: 98.64 +0.84 -0.83 Mpc
Ωm: 0.307 +0.008 -0.008
w0: -0.834 +0.045 -0.046 (3.61 - 3.69 sigma)
wa: 0
Chi squared: 1646.49 (Δ chi2 12.48)
Degrees of freedom: 1838

===============================

Flat w(z) = w0 + wa * z / (1 + z)
ΔM: -9.187 +0.014 -0.014 mag
r_d * h: 98.53 +0.84 -0.84 Mpc
Ωm: 0.321 +0.013 -0.016
w0: -0.785 +0.073 -0.068 (2.95 - 3.16 sigma)
wa: -0.708 +0.459 -0.464 (1.54 sigma)
Chi squared: 1645.45 (Δ chi2 13.52)
Degrees of freedom: 1837
"""
