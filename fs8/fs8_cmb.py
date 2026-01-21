from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
from interpolator import interp_quad
import y2018fs8.data as fs8_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

data = fs8_data.data
inv_cov_mat = np.linalg.inv(fs8_data.cov_mat)

z_max = np.max(data["z"]) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0, wa):
    a3 = 1 / (1 + z) ** 3
    return 4 / ((1 + w0) * a3 + (1 - w0)) ** 2
    # return (1 + z) ** (3 * (1 + w0))


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0, wa)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def Hz(z, theta):
    H0, Obh2, Och2, w0 = theta[0:4]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


@njit
def dH_da(z, theta):
    a = 1 / (1 + z)
    dz = 1e-05
    dH_dz = (Hz(z + dz, theta) - Hz(z - dz, theta)) / (2 * dz)
    return -dH_dz / a**2


@njit
def DM(z, theta):
    dh_grid = c / Hz(z_grid, theta)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid))
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_quad(z, z_grid, cum_dm)


H0_fid = 67.6
Obh2_fid = 0.0222
params_fid = [H0_fid, Obh2_fid, 0.12, -1.0, 0.80, 1.0]
denominator_fiducial = np.empty(len(data["z"]), dtype=np.float64)

for i in range(len(data["z"])):
    z = data["z"][i]
    Om_fid = data["omega_fid"][i]
    params_fid[2] = Om_fid * (H0_fid / 100) ** 2 - Obh2_fid - Omnuh2
    denominator_fiducial[i] = Hz(z, params_fid) * DM(z, params_fid)


@njit
def growth_ode(a, y, *params):
    H0, Obh2, Och2 = params[0], params[1], params[2]
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2

    z = 1 / a - 1
    H_val = Hz(z, params)
    dH_da_val = dH_da(z, params)

    delta, d_delta_da = y

    source = (3 / 2) * (Obc / a**5) * delta * (H0 / H_val) ** 2
    friction = -(3 / a + dH_da_val / H_val) * d_delta_da
    d2_delta_da = friction + source

    return [d_delta_da, d2_delta_da]


max_z = 1100
a_vals = np.logspace(np.log10(1 / (1 + max_z)), 0, 11_000)


def fs8_theory(z, params):
    sol = solve_ivp(
        growth_ode,
        t_span=(a_vals[0], a_vals[-1]),
        y0=(a_vals[0], 1.0),
        t_eval=a_vals,
        rtol=1e-8,
        atol=1e-10,
        args=params,
    )
    delta, d_delta_da = sol.y
    sig8 = params[-2]

    delta0 = np.interp(1.0, a_vals, delta)
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    a = 1 / (1 + z)
    return a * np.interp(a, a_vals, d_delta_da) * sig8 / delta0


def chi_squared(theta):
    q = Hz(data["z"], theta) * DM(data["z"], theta) / denominator_fiducial
    delta = data["fs8"] - fs8_theory(data["z"], theta) / q
    chi2_fs8 = theta[-1] ** 2 * np.dot(delta, np.dot(inv_cov_mat, delta))

    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Hz, theta[1], theta[2], theta)
    chi2_cmb = np.dot(delta_cmb, np.dot(cmb.inv_cov_mat, delta_cmb))

    return chi2_fs8 + chi2_cmb


N = len(data["z"])
logdet = np.linalg.slogdet(fs8_data.cov_mat)[1]
norm_fact_1 = N * np.log(2 * np.pi) + logdet


def log_likelihood(theta):
    norm_fact = norm_fact_1 - 2 * N * np.log(theta[-1])
    return -0.5 * (chi_squared(theta) + norm_fact)


bounds = np.array(
    [
        (50.0, 80.0),  # H0
        (0.01, 0.035),  # Ob * h^2
        (0.1, 0.35),  # Oc * h^2
        (-1.0, 0.0),  # w0
        (0.2, 1.2),  # sigma8
        (0.5, 2.2),  # f_err: overstimation factor of the errors
    ],
    dtype=np.float64,
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


def log_probability(theta):
    lp = log_prior(theta)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def main():
    from multiprocessing import Pool
    import emcee
    import matplotlib.pyplot as plt
    from corner_plot import plot_corner_and_chains
    from log_evidence import log_evidence

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 400
    nsteps = 2600 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("mean acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (w0_16, w0_50, w0_84),
        (s8_16, s8_50, s8_84),
        (f_16, f_50, f_84),
    ] = pct

    Obch2_samples = samples[:, 1] + samples[:, 2]
    Omh2_samples = Obch2_samples + Omnuh2
    Om_samples = Omh2_samples / (samples[:, 0] / 100) ** 2
    S8_samples = 100 * samples[:, -2] * np.sqrt(Obch2_samples / 0.3) / samples[:, 0]

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])
    S8_16, S8_50, S8_84 = np.percentile(S8_samples, [15.9, 50, 84.1])

    best_fit = np.percentile(samples, 50, axis=0)

    print(f"H0 = {H0_50:.2f} +{H0_84-H0_50:.2f} -{H0_50-H0_16:.2f} km/s/Mpc")
    print(f"Ωbh2 = {Obh2_50:.5f} +{Obh2_84-Obh2_50:.5f} -{Obh2_50-Obh2_16:.5f}")
    print(f"Ωch2 = {Och2_50:.5f} +{Och2_84-Och2_50:.5f} -{Och2_50-Och2_16:.5f}")
    print(f"Ωmh2 = {Omh2_50:.4f} +{Omh2_84-Omh2_50:.4f} -{Omh2_50-Omh2_16:.4f}")
    print(f"Ωm = {Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}")
    print(f"σ8 = {s8_50:.3f} +{s8_84-s8_50:.3f} -{s8_50-s8_16:.3f}")
    print(f"S8 = {S8_50:.3f} +{S8_84-S8_50:.3f} -{S8_50-S8_16:.3f}")
    print(f"w0 = {w0_50:.3f} +{w0_84-w0_50:.3f} -{w0_50-w0_16:.3f}")
    print(f"f = {f_50:.2f} +{f_84-f_50:.2f} -{f_50-f_16:.2f}")
    print(f"chi2 = {chi_squared(best_fit):.2f}")
    print(f"log likelihood = {log_likelihood(best_fit):.1f}")
    print(f"log evidence = {log_evd:.1f}")
    print(f"degs of freedom = {N - len(best_fit)}")

    labels = [
        "$H_0$",
        "$Ωbh^2$",
        "$Ωch^2$",
        "$w_0$",
        "$\sigma_8$",
        "$f_{err}$",
    ]
    plot_corner_and_chains(labels, samples, chains_samples)

    z_plot = np.linspace(0, np.max(data["z"]), 200)
    fs8_plot = fs8_theory(z_plot, best_fit)

    q = Hz(data["z"], best_fit) * DM(data["z"], best_fit) / denominator_fiducial

    plt.errorbar(
        data["z"],
        data["fs8"] * q,
        yerr=data["fs8_err"] * q / f_50,
        fmt=".",
        label="data",
    )
    plt.plot(z_plot, fs8_plot, label="model", color="C1")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.legend()
    plt.show()

    residuals = q * data["fs8"] - fs8_theory(data["z"], best_fit)
    plt.errorbar(
        data["z"],
        residuals,
        yerr=data["fs8_err"] * q / f_50,
        fmt=".",
        label="residuals",
    )
    plt.axhline(0, color="k", ls="--")
    plt.xlabel("z")
    plt.ylabel("residuals")
    plt.legend()
    plt.show()

    plt.hist(residuals, bins=12, density=True)
    plt.xlabel("residuals")
    plt.ylabel("density")
    plt.show()


if __name__ == "__main__":
    main()


"""
flat ΛCDM
H0 = 67.89 +0.48 -0.48 km/s/Mpc
Ωbh2 = 0.02252 +0.00011 -0.00011
Ωch2 = 0.11871 +0.00115 -0.00116
Ωmh2 = 0.1419 +0.0011 -0.0011
Ωm = 0.308 +0.007 -0.007
σ8 = 0.775 +0.011 -0.011
S8 = 0.783 +0.012 -0.012 (consistent within 1 sigma with both CMB-only and fσ8-only)
f = 1.30 +0.12 -0.12
chi2 = 62.58
log likelihood = 100.0
log evidence = 78.7
degs of freedom = 58

===============================

flat wCDM
H0 = 69.06 +1.93 -1.84 km/s/Mpc
Ωbh2 = 0.02252 +0.00011 -0.00011
Ωch2 = 0.11891 +0.00119 -0.00119
Ωmh2 = 0.1421 +0.0012 -0.0011
Ωm = 0.298 +0.017 -0.016
σ8 = 0.770 +0.013 -0.013
S8 = 0.765 +0.030 -0.029
w0 = -1.042 +0.062 -0.065
f = 1.29 +0.12 -0.12
chi2 = 61.43
log likelihood = 100.1
log evidence = 74.2
degs of freedom = 57

===============================

flat wzCDM
H0 = 66.61 +0.97 -1.40 km/s/Mpc
Ωbh2 = 0.02253 +0.00011 -0.00011
Ωch2 = 0.11851 +0.00117 -0.00116
Ωmh2 = 0.1417 +0.0011 -0.0011
Ωm = 0.320 +0.014 -0.011
σ8 = 0.781 +0.012 -0.012
S8 = 0.804 +0.025 -0.019
w0 = -0.908 +0.102 -0.065
f = 1.29 +0.12 -0.11
chi2 = 62.59
log likelihood = 99.5
log evidence = 75.2
degs of freedom = 57
"""
