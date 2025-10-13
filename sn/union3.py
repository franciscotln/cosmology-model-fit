from numba import njit
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid
from y2023union3.data import get_data

legend, z_values, mu_vals, cov_matrix = get_data()

cho = cho_factor(cov_matrix)

C = 299792.458  # Speed of light (km/s)
H0 = 70  # Hubble constant (km/s/Mpc)


z = np.linspace(0, np.max(z_values), num=1000)
cubed = (1 + z) ** 3

# params indices
D_M = 0
OM = 1
W0 = 2

bounds = np.array([(-0.6, 0.6), (0.0, 1.0), (-2.0, 0.0)])  # ΔM, Ωm, w0


@njit
def Ez(params):
    Om, w0 = params[OM], params[W0]
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(params), z, initial=0)
    return np.interp(z_values, z, integral_values)


def mu_theory(params):
    a0_over_ae = 1 + z_values
    comoving_distance = (C / H0) * integral_Ez(params)
    return params[D_M] + 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def chi_squared(params):
    delta = mu_vals - mu_theory(params)
    return delta.dot(cho_solve(cho, delta, check_finite=False))


def log_likelihood(params):
    return -0.5 * chi_squared(params)


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import corner, emcee
    from scipy.stats import skew
    from multiprocessing import Pool
    import matplotlib.pyplot as plt
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from .plotting import plot_predictions, print_color, plot_residuals

    n_dim = len(bounds)
    n_walkers = 150
    burn_in = 200
    n_steps = burn_in + 2000
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, n_dim))

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.30),
                (emcee.moves.DEMove(), 0.56),
                (emcee.moves.DESnookerMove(), 0.14),
            ],
        )
        sampler.run_mcmc(initial_pos, n_steps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print(
            "effective samples", n_walkers * (n_steps - burn_in) * n_dim / np.max(tau)
        )
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chain_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    print_color("Gelman-Rubin R-hat:", gelman_rubin(chain_samples))

    one_sigma_ci = [15.9, 50, 84.1]
    dM_16, dM_50, dM_84 = np.percentile(samples[:, D_M], one_sigma_ci)
    Om_16, Om_50, Om_84 = np.percentile(samples[:, OM], one_sigma_ci)
    w0_16, w0_50, w0_84 = np.percentile(samples[:, W0], one_sigma_ci)

    best_fit_params = np.array([dM_50, Om_50, w0_50], dtype=np.float64)

    predicted_distances = mu_theory(best_fit_params)
    residuals = mu_vals - predicted_distances

    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mu_vals - np.mean(mu_vals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calculate root mean square deviation
    rmsd = np.sqrt(np.mean(residuals**2))

    dM_label = f"{dM_50:.4f} +{dM_84-dM_50:.4f}/-{dM_50-dM_16:.4f}"
    Om_label = f"{Om_50:.4f} +{Om_84-Om_50:.4f}/-{Om_50-Om_16:.4f}"
    w0_label = f"{w0_50:.4f} +{w0_84-w0_50:.4f}/-{w0_50-w0_16:.4f}"

    print_color("Dataset", legend)
    print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
    print_color("Sample size", len(z_values))
    print_color("ΔM", dM_label)
    print_color("Ωm", Om_label)
    print_color("w0", w0_label)
    print_color("R-squared (%)", f"{100 * r2:.2f}")
    print_color("RMSD (mag)", f"{rmsd:.3f}")
    print_color("Skewness of residuals", f"{skew(residuals):.3f}")
    print_color("Chi squared", f"{chi_squared(best_fit_params):.1f}")
    print_color(
        "Log Evidence", f"{log_evidence(samples, log_probs, log_probability):.1f}"
    )
    print_color("Degs of freedom", len(z_values) - len(best_fit_params))

    labels = ["$Δ_M$", "$Ω_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=2.0,
        smooth1d=2.0,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    plt.figure(figsize=(16, 1.5 * n_dim))
    for n in range(n_dim):
        plt.subplot2grid((n_dim, 1), (n, 0))
        plt.plot(chain_samples[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()

    sigma_mu = np.sqrt(np.diag(cov_matrix))

    plot_predictions(
        legend=legend,
        x=z_values,
        y=mu_vals,
        y_err=sigma_mu,
        y_model=predicted_distances,
        label=f"Best fit: $Ω_m$={Om_50:.4f}",
        x_scale="log",
    )

    plot_residuals(z_values=z_values, residuals=residuals, y_err=sigma_mu, bins=40)


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*******************************

Flat ΛCDM: w(z) = -1

ΔM: -0.0698 +0.0868/-0.0876
Ωm: 0.3572 +0.0273/-0.0261
w0: -1
wa: 0
R-squared (%): 99.95
RMSD (mag): 0.050
Skewness of residuals: 0.590
Chi squared: 24.0
Log Evidence: -16.2
Degs of freedom: 20

===============================

Flat wCDM: w(z) = w0

ΔM: -0.0579 +0.0868/-0.0874
Ωm: 0.2518 +0.0868/-0.1085
w0: -0.7463 +0.1521/-0.1835
wa: 0
R-squared (%): 99.94
RMSD (mag): 0.055
Skewness of residuals: -1.272
Chi squared: 22.1
Log Evidence: -15.9
degrees of freedom: 19

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)

ΔM: -0.0541 +0.0873/-0.0860
Ωm: 0.2955 +0.0526/-0.0540
w0: -0.7517 +0.1441/-0.1702
wa: -(1 + w0) = 0.2483 +0.1702/-0.1441
R-squared (%): 99.94
RMSD (mag): 0.053
Skewness of residuals: -1.072
Chi squared: 21.7
Log Evidence: -15.9
Degs of freedom: 19

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)

ΔM: -0.0319 +0.0904/-0.0899 mag
Ωm: 0.4406 +0.0555/-0.0923
w0: -0.5574 +0.2799/-0.2343
wa: -4.1187 +2.8976/-3.2694
R-squared (%): 99.96
RMSD (mag): 0.043
Skewness of residuals: 0.669
Chi squared: 20.6
Log Evidence: -13.4
Degs of freedom: 18
"""
