import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import emcee
import corner
from multiprocessing import Pool
from y2005cc.data import get_data


legend, z_values, H_values, cov_matrix = get_data()
dH_values = np.sqrt(np.diag(cov_matrix)) # simple diagonal covariance matrix

# Fitting models directly to the CC data (presents high correlation between H0 and w0)
best_fit = {
    "LCDM": [66.71, 0.334],
    "wCDM": [70.80, 0.307, -1.489],
    "wxCDM": [71.15, 0.306, -1.559]
}

def H_lcdm_z(z, params):
    h0, o_m = params
    return h0 * np.sqrt(o_m * (1 + z)**3 + (1 - o_m))


def H_wcdm_z(z, params):
    h0, o_m, w0 = params
    exponent = 3 * (1 + w0)
    sum = 1 + z
    return h0 * np.sqrt(o_m * sum**3 + (1 - o_m) * sum**exponent)


def H_wxcdm_z(z, params):
    h0, o_m, w0 = params
    exponent = 3 * (1 + w0)
    sum = 1 + z
    return h0 * np.sqrt(o_m * sum**3 + (1 - o_m) * ((2 * sum**2) / (1 + sum**2))**exponent)


bounds = np.array([
    (50, 70),     # H0
    (0.1, 0.6),   # Ωm
    (-1.5, -0.6), # w0
])


def chi_squared(params, H_z, zvals, Hvals, dHvals):
    delta = Hvals - H_z(zvals, params)
    return np.sum(delta**2 / dHvals**2)


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params, H_z, zvals, Hvals, dHvals):
    return -0.5 * chi_squared(params, H_z, zvals, Hvals, dHvals)


def log_probability(params, H_z, zvals, Hvals, dHvals):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params, H_z, zvals, Hvals, dHvals)


def main():
    z_grid = np.linspace(0, np.max(z_values), 1000)
    n_bootstrap = 20_000
    H_fits = np.zeros((n_bootstrap, len(z_grid)))

    # Bootstrap loop: resampling with replacement
    for i in range(n_bootstrap):
        indices = np.random.choice(len(z_values), size=len(z_values), replace=True)
        z_boot = z_values[indices]
        H_boot = H_values[indices]
        dH_boot = dH_values[indices]

        sort_idx = np.argsort(z_boot)
        z_boot = z_boot[sort_idx]
        H_boot = H_boot[sort_idx]
        dH_boot = dH_boot[sort_idx]

        spline_i = UnivariateSpline(z_boot, H_boot, w=1/dH_boot, k=2, s=len(z_boot))
        H_fits[i] = spline_i(z_grid)

    H_16, H_50, H_84 = np.percentile(H_fits, [15.9, 50, 84.1], axis=0)
    upper = H_84[0] - H_50[0]
    lower = H_50[0] - H_16[0]
    print(f"Spline H0: {H_50[0]:.1f} +{upper:.1f} - {lower:.1f} km/s/Mpc")
    # Spline H0: 67.1 +3.7 - 3.8 km/s/Mpc

    model_name = 'wxCDM'
    H_z = H_wxcdm_z
    std = np.std(H_fits, axis=0)
    args = (H_z, z_grid, H_50, std)
    ndim = len(bounds)
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 15000 + burn_in
    initial_pos = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(nwalkers, ndim))

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=log_probability,
            pool=pool,
            args=args,
        )
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

    best_fit_params = [H0_50, omega_50, w0_50]
    chi_squared_value = chi_squared(best_fit_params, H_z, z_grid, H_50, std)
    print(f"chi-squared: {chi_squared_value:.2f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")

    labels = [f"$H_0$", f"$\Omega_m$", f"$w_0$"]
    corner.corner(
        samples,
        bins=100,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        title_fmt=".4f",
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
        levels=(0.393, 0.864), # 1 and 2 sigmas in 2D
    )
    plt.savefig(f"{model_name}_cc_triangle.png", dpi=500)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.errorbar(z_values, H_values, yerr=dH_values, fmt='.', label=legend, capsize=2)
    plt.plot(z_grid, H_50, label=fr"Spline: $H_0 = {H_50[0]:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}$ km/s/Mpc", color='black')
    plt.fill_between(z_grid, H_16, H_84, color='gray', alpha=0.5, label=r'spline $1\sigma$')
    plt.plot(z_grid, H_z(z_grid, best_fit_params), label=fr"{model_name} $\chi^2 = {chi_squared_value:.2f}$", color='orange', linestyle='--', alpha=0.9)
    plt.title('Smoothing Spline 2deg with resampling uncertainties')
    plt.xlabel('z')
    plt.xlim(0, 2)
    plt.ylabel("H(z) - km/s/Mpc")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name}_cc_spline.png", dpi=500)
    plt.close()

if __name__ == "__main__":
    main()
