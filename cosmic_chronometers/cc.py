import numpy as np
from scipy.linalg import cho_factor, cho_solve
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2005cc.data import get_data

# Speed of light in km/s
c = 299792.458

legend, z_values, H_values, cov_matrix = get_data()
cho = cho_factor(cov_matrix)

def H_z(z, params):
    h0, o_m, w0 = params[0], params[1], params[2]
    one_plus_z = 1 + z
    return h0 * np.sqrt(o_m * one_plus_z**3 + (1 - o_m) * ((2 * one_plus_z**2)/(1 + one_plus_z**2))**(3 * (1 + w0)))


bounds = np.array([
    (40, 110),   # H0
    (0.1, 0.6),  # Ωm
    (-3.5, 0.5), # w0
])


def chi_squared(params):
    delta = H_values - H_z(z_values, params)
    return np.dot(delta, cho_solve(cho, delta))


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


L, _ = cho
log_det = 2 * np.sum(np.log(np.diag(L)))
pi_const = len(H_values) * np.log(2 * np.pi)

def log_likelihood(params):
    return -0.5 * (chi_squared(params) + log_det + pi_const)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def plot_predictions(params):
    h0 = params[0]
    z_smooth = np.linspace(0, max(z_values), 100)
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=z_values,
        y=H_values,
        yerr=np.sqrt(np.diag(cov_matrix)),
        fmt='.',
        color='blue',
        alpha=0.4,
        label="CC data",
        capsize=2,
        linestyle="None",
    )
    plt.plot(z_smooth, H_z(z_smooth, params), color='red', alpha=0.5)
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z_values) + 0.2)
    plt.legend()
    plt.title(f"Cosmic chronometers: $H_0$={h0:.2f} km/s/Mpc")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=z_values,
        y=H_values - H_z(z_values, params),
        yerr=np.sqrt(np.diag(cov_matrix)),
        fmt='.',
        color='blue',
        alpha=0.4,
        label="Residuals",
        capsize=2,
        linestyle="None",
    )
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z) - H_{model}(z)$")
    plt.xlim(0, np.max(z_values) + 0.2)
    plt.title(f"Residuals")
    plt.legend()
    plt.show()


def main():
    ndim = len(bounds)
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 20000 + burn_in
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

    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [H0_16, H0_50, H0_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [H0_50, omega_50, w0_50]

    print(f"H0: {H0_50:.4f} +{(H0_84 - H0_50):.4f} -{(H0_50 - H0_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Degs of freedom: {z_values.size  - len(best_fit)}")

    plot_predictions(best_fit)

    labels = [f"$H_0$", f"$\Omega_m$", f"$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864), # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()


if __name__ == "__main__":
    main()


"""
******************************
Results for data from
https://arxiv.org/pdf/2307.09501
*****************************

Flat ΛCDM
H0: 66.7067 +5.4326 -5.4541 km/s/Mpc
Ωm: 0.3340 +0.0790 -0.0626
w0: -1
wa: 0
Chi squared: 14.5403
Degs of freedom: 30

===============================

Flat wCDM
H0: 70.7997 +10.4111 -8.1662 km/s/Mpc
Ωm: 0.3070 +0.0787 -0.0674
w0: -1.4889 +0.6806 -0.9017
wa: 0
Chi squared: 15.0385
Degs of freedom: 29

===============================

Flat w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 71.1489 +10.9308 -8.8713 km/s/Mpc
Ωm: 0.3060 +0.0802 -0.0639
w0: -1.5587 +0.7770 -0.9710
wa: 0
Chi squared: 14.6439
Degs of freedom: 29

=============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 71.5058 +11.2009 -9.1650 km/s/Mpc
Ωm: 0.3163 +0.1017 -0.0858
w0: -1.5619 +0.8692 -0.9932
wa: -0.6121 +2.8483 -2.9732
Chi squared: 14.9993
Degs of freedom: 28
"""