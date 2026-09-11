from numba import njit
import numpy as np
import time
from scipy.linalg import cho_factor
from scipy.constants import c as c0
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2022pantheonSHOES.data import get_data

legend, z_cmb, z_hel, mb_vals, cov_matrix = get_data()

c = c0 / 1000  # Speed of light (km/s)

cho = cho_factor(cov_matrix, lower=True)[0]
logdet = 2.0 * np.sum(np.log(np.diag(cho)))
N_sample = z_cmb.size

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=3000)
dz = z_grid[1] - z_grid[0]

zp1_cube = (1.0 + z_grid) ** 3

names=["H0", "M0", "Om", "v100"]
labels = ["H_0", "M_0", "Ω_m", "v_{100}"]
H0, M0, OM, V100 = range(len(names))
bounds = np.empty((len(names), 2))
bounds[H0] = (50, 90)
bounds[M0] = (-20, -19)
bounds[OM] = (0, 0.7)
bounds[V100] = (-3, 3)


@njit
def H_z(p):
    return p[H0] * np.sqrt(p[OM] * zp1_cube + (1.0 - p[OM]))


@njit
def DM_z(p, z):
    dh_grid = c / H_z(p)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def get_z_cosmo(p):
    # Heaviside step at z = 0.15
    v_km_s = 100 * p[V100] * np.where(z_cmb <= 0.15, 1, -1)
    z_pec = v_km_s / c
    return -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)


def mu_corr(params, DM_ref):
    # For plotting purposes
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(params, z_cosmo) / DM_ref)


@njit
def mu_theory(DM):
    return 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params):
    z_cosmo = get_z_cosmo(params)
    delta = mb_vals - params[M0] - mu_theory(DM_z(params, z_cosmo))
    y = solve_triangular(cho, delta)
    return np.dot(y, y)


@njit
def log_likelihood(params):
    normalization = N_sample * np.log(2.0 * np.pi) + logdet
    return -0.5 * (chi_squared(params) + normalization)


uniform_prior_norm = -np.sum(np.log(bounds[1:, 1] - bounds[1:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    # TRGB Freedman et al
    h0_mean, h0_sigma = 70.39, 1.80
    log_h0_prior = (
        -0.5 * ((params[H0] - h0_mean) / h0_sigma) ** 2
        - np.log(h0_sigma * np.sqrt(2.0 * np.pi))
    )

    return uniform_prior_norm + log_h0_prior


@njit
def log_probability_jit(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf, -np.inf
    ll = log_likelihood(params)
    return lp + ll, ll


def log_probability(params):
    start_time = time.perf_counter()
    log_prob, log_like = log_probability_jit(params)
    exec_time = time.perf_counter() - start_time
    return log_prob, np.array([log_like, exec_time])


def main():
    import emcee
    from multiprocessing import Pool
    from getdist import MCSamples, plots
    import matplotlib.pyplot as plt
    from log_evidence import log_evidence
    from sn.plotting import plot_predictions, plot_residuals

    burn_in = 500
    ndim = len(bounds)
    nwalkers = 80
    nsteps = burn_in + 3000
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, moves=moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"})

    samples = sampler.get_chain(discard=burn_in, flat=False)
    blobs = sampler.get_blobs(discard=burn_in, flat=False)
    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    flat_log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    log_evd = log_evidence(
        flat_samples, flat_log_probs, lambda params: log_probability(params)[0], bounds,
    )

    exec_times = 1000 * blobs[:, :, 1].ravel()
    print("Execution time:")
    print(f"mean: {np.mean(exec_times):.4f} ms")
    print(f"std: {np.std(exec_times):.4f} ms")
    print(f"max: {np.max(exec_times):.4f} ms")
    print(f"min: {np.min(exec_times):.4f} ms\n")

    # reshape for getdist
    chain_list = np.moveaxis(samples, 1, 0)
    loglike_list = np.moveaxis(blobs[:, :, 0], 1, 0)

    gd_samples = MCSamples(
        samples=chain_list,
        loglikes=-loglike_list,
        names=names,
        labels=labels,
        label='Pantheon+'
    )
    gd_samples.addDerived(100 * gd_samples["v100"], name="v_km_s", label="v_{km/s}")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    MAP_params = np.argmax(flat_log_probs)
    best_fit = flat_samples[MAP_params]

    DM = DM_z(best_fit, z_cmb)
    mB_pred = mu_theory(DM) + best_fit[M0]
    corrected_mags = mb_vals - mu_corr(best_fit, DM)
    residuals = corrected_mags - mB_pred

    print("DOF", len(z_cmb) - len(best_fit))
    print("Chi squared", f"{chi_squared(best_fit):.2f}")
    print("Log Evidence", f"{log_evd:.1f}")

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=names,
        title_limit=1,
        contour_colors=["C0"],
        filled=True,
    )
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=corrected_mags - best_fit[M0],
        y_err=np.sqrt(np.diag(cov_matrix)),
        y_model=mB_pred - best_fit[M0],
        label=f"$Ω_m$={best_fit[OM]:.3f}",
        x_scale="log",
    )
    plot_residuals(
        z_values=z_cmb,
        residuals=residuals,
        y_err=np.sqrt(np.diag(cov_matrix)),
        bins=40,
    )


if __name__ == "__main__":
    main()


# *********************************
# Dataset: Pantheon+ (2022)
# z range: 0.0102 - 2.2614
# Sample size: 1590
# *********************************


# ----------- Flat ΛCDM -----------
# M: -19.340 +- 0.056 mag
# H0: 70.4 +- 1.8 km/s/Mpc
# Ωm: 0.332 +- 0.018
# DOF: 1587
# Chi squared 1402.92
# Log Evidence 838.4
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.15 inflow z > 0.15 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)
#
# M: -19.352 +- 0.056
# H0: 70.4 +- 1.8 km/s/Mpc
# Ωm: 0.315 +- 0.020
# v: -68 +- 41 km/s (prior ~ U[-300, 300])
# DOF: 1586
# Chi squared: 1400.15
# Log Evidence: 838.0
# ---------------------------------


# ----------- Flat wCDM -----------
# M: -19.336 +- 0.057
# H0: 70.4 +- 1.8 km/s/Mpc
# Ωm: 0.285 +0.081/-0.058
# w0: -0.91 +0.17/-0.13 (prior ~ U[-1.5, -0.5])
# DOF: 1586
# Chi squared 1402.47
# Log Evidence 837.7
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < 0 enforced in the likelihood
#
# M0: 19.336 +0.058/-0.053 mag
# H0: 70.4 +- 1.8 km/s/Mpc
# Ωm: 0.316 +0.12 -0.049
# w0: -0.92 +0.17 -0.14 (prior ~ U[-3.0, 1.0])
# wa: -0.61 +1.4 -0.58 (prior ~ U[-3.0, 2.0])
# DOF: 1585
# Chi squared 1402.44
# Log Evidence 836.5
# ---------------------------------
