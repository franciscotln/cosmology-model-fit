from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_ivp import solve_ivp
from solve_triangular import solve_triangular
import y2018fs8.data as fs8_data
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

data = fs8_data.data
z_vals = fs8_data.data["z"]
fs8_vals = fs8_data.data["fs8"]
a_vals = 1 / (1.0 + z_vals)

N = len(data)
logdet = np.linalg.slogdet(fs8_data.cov_mat)[1]
norm_factor = N * np.log(2 * np.pi) + logdet

cho = cho_factor(fs8_data.cov_mat, lower=True)[0]

z_max = np.max(z_vals) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def w_de_z(z, w0):
    # Thawing quintessence wzCDM
    return -1.0 + 2 * (1.0 + w0) / (1.0 + w0 + (1.0 - w0) * (1.0 + z) ** 3)


@njit
def Ode_z(z, w0):
    # Thawing quintessence wzCDM
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2


@njit
def d_Ode_dz(z, w0):
    return Ode_z(z, w0) * 3 * (1.0 + w_de_z(z, w0)) / (1.0 + z)


@njit
def d_Omnu_dz(z):
    return cmb.Omnu_z(z) * 3 * (1.0 + cmb.w_nu_z(z)) / (1.0 + z)


@njit
def Ez(z, H0, Obh2, Och2, w0):
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2
    Onu = Omnuh2 / h**2
    Or = Orh2 / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def Hz(z, theta):
    H0 = theta[0]
    return H0 * Ez(z, H0, Obh2=theta[1], Och2=theta[2], w0=theta[3])


cmb.set_HZ(Hz)


@njit
def dH_da(z, H_val, theta):
    H0, Obh2, Och2, w0 = theta[0:4]
    h = H0 / 100

    Obc = (Obh2 + Och2) / h**2
    Or = Orh2 / h**2
    Onu = Omnuh2 / h**2
    Ode = 1.0 - Obc - Or - Onu

    matter = Obc * 3 * (1.0 + z) ** 2
    rad = Or * 4 * (1.0 + z) ** 3
    nu = Onu * d_Omnu_dz(z)
    de = Ode * d_Ode_dz(z, w0)

    numerator = matter + rad + nu + de
    denominator = 2 * H_val / (1.0 + z) ** 2
    return -numerator * H0**2 / denominator


@njit
def DM(z, theta):
    dh_grid = c / Hz(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid), dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def growth_ODE(a, y, params):
    H0, Obh2, Och2 = params[0], params[1], params[2]
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2

    z = 1 / a - 1.0
    H_val = Hz(z, params)
    dH_da_val = dH_da(z, H_val, params)

    delta, d_delta_da = y

    source = (3 / 2) * (Obc / a**5) * delta * (H0 / H_val) ** 2
    friction = -(3 / a + dH_da_val / H_val) * d_delta_da
    d2_delta_da = friction + source

    return np.array([d_delta_da, d2_delta_da])


max_z = 500
a_span = np.logspace(np.log10(1 / (1.0 + max_z)), 0, 5_000)


@njit
def fs8_theory(a, params):
    sol = solve_ivp(
        growth_ODE,
        t_span=(a_span[0], a_span[-1]),
        y0=(a_span[0], 1.0),
        t_eval=a_span,
        rtol=1e-6,
        atol=1e-8,
        args=(params,),
    )
    delta, d_delta_da = sol.y
    sigma8_0 = params[-2]
    delta_0 = delta[-1]
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    return a * interp_pchip(a, a_span, d_delta_da) * sigma8_0 / delta_0


Hz_DMz_fid = np.empty(N, dtype=np.float64)
for i in range(N):
    z = z_vals[i]
    Obh2_fid = 0.0222
    w0_fid = -1.0
    Om_fid = data["omega_fid"][i]
    H0_fid = data["H0_fid"][i]
    Och2_fid = Om_fid * (H0_fid / 100) ** 2 - Obh2_fid - Omnuh2
    sig8_fid = data["s8_fid"][i]
    params_fid = [H0_fid, Obh2_fid, Och2_fid, w0_fid, sig8_fid, 1.0]
    DM_i = DM(np.array([z]), params_fid)[0]
    Hz_DMz_fid[i] = Hz(z, params_fid) * DM_i


@njit
def chi2_fs8(theta):
    q = Hz(z_vals, theta) * DM(z_vals, theta) / Hz_DMz_fid
    delta = fs8_vals - fs8_theory(a_vals, theta) / q
    return theta[-1] ** 2 * solve_triangular(cho, delta)


@njit
def chi2_cmb(theta):
    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(theta[1], theta[2], theta)
    return delta_cmb @ cmb.inv_cov_mat @ delta_cmb


@njit
def chi_squared(theta):
    return chi2_fs8(theta) + chi2_cmb(theta)


@njit
def log_likelihood(theta):
    norm_fact = norm_factor - 2 * N * np.log(theta[-1])
    return -0.5 * (chi_squared(theta) + norm_fact)


bounds = np.array(
    [
        (50, 80),  # H0
        (0.01, 0.035),  # Ob * h^2
        (0.1, 0.35),  # Oc * h^2
        (-1.0, 0.0),  # w0
        (0.5, 1.0),  # sigma8
        (0.2, 3.2),  # f_err: overstimation factor of the errors
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    return normalization


@njit
def log_probability_jit(theta):
    lp = log_prior(theta)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def log_probability(theta):
    return log_probability_jit(theta)


def main():
    from multiprocessing import Pool
    import emcee
    from corner_plot import plot_corner_and_chains
    from fs8.plot_predictions import plot_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 100
    burn_in = 400
    nsteps = 2600 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

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
    MAP_samples = samples[np.argmax(log_probs)]

    print(f"H0 = {H0_50:.2f} +{H0_84-H0_50:.2f} -{H0_50-H0_16:.2f} km/s/Mpc")
    print(f"Ωbh2 = {Obh2_50:.5f} +{Obh2_84-Obh2_50:.5f} -{Obh2_50-Obh2_16:.5f}")
    print(f"Ωch2 = {Och2_50:.5f} +{Och2_84-Och2_50:.5f} -{Och2_50-Och2_16:.5f}")
    print(f"Ωmh2 = {Omh2_50:.4f} +{Omh2_84-Omh2_50:.4f} -{Omh2_50-Omh2_16:.4f}")
    print(f"Ωm = {Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}")
    print(f"σ8 = {s8_50:.3f} +{s8_84-s8_50:.3f} -{s8_50-s8_16:.3f}")
    print(f"S8 = {S8_50:.3f} +{S8_84-S8_50:.3f} -{S8_50-S8_16:.3f}")
    print(f"w0 = {w0_50:.3f} +{w0_84-w0_50:.3f} -{w0_50-w0_16:.3f}")
    print(f"f = {f_50:.2f} +{f_84-f_50:.2f} -{f_50-f_16:.2f}")
    print(f"chi2 = {chi_squared(MAP_samples):.2f}")
    print(f"log likelihood = {log_likelihood(MAP_samples):.1f}")
    print(f"degs of freedom = {N + len(cmb.DISTANCE_PRIORS) - len(best_fit)}")

    labels = ["$H_0$", "$Ωbh^2$", "$Ωch^2$", "$w_0$", "$\\sigma_8$", "$f_{err}$"]
    plot_corner_and_chains(labels, samples, chains_samples)
    plot_predictions(
        fs8_theory=lambda z: fs8_theory(1 / (1 + z), best_fit),
        data=data,
        q=Hz(z_vals, best_fit) * DM(z_vals, best_fit) / Hz_DMz_fid,
        f_err=f_50,
    )


if __name__ == "__main__":
    main()


# ----------- flat ΛCDM -----------
# H0 = 67.61 +0.47 -0.47 km/s/Mpc
# Ωbh2 = 0.02249 +0.00011 -0.00011
# Ωch2 = 0.11933 +0.00115 -0.00113
# Ωmh2 = 0.1425 +0.0011 -0.0011
# Ωm = 0.312 +0.007 -0.007
# σ8 = 0.789 +0.009 -0.009
# S8 = 0.802 +0.011 -0.011
# f = 1.80 +0.17 -0.17 (error overestimation factor)
# chi2 = 56.32
# log likelihood = 106.7
# degs of freedom = 54
# ---------------------------------


# ----------- flat wCDM -----------
# H0 = 65.79 +1.39 -1.35 km/s/Mpc
# Ωbh2 = 0.02252 +0.00011 -0.00011
# Ωch2 = 0.11886 +0.00120 -0.00118
# Ωmh2 = 0.1420 +0.0012 -0.0011
# Ωm = 0.328 +0.014 -0.014
# σ8 = 0.798 +0.011 -0.011
# S8 = 0.833 +0.025 -0.025
# w0 = -0.93 +0.05 -0.05 (prior U[-1.5, -0.5])
# f = 1.82 +0.18 -0.17 (error overestimation factor)
# chi2 = 56.03
# log likelihood = 107.8
# degs of freedom = 53
# ---------------------------------


# ----------- flat wzCDM ----------
# H0 = 64.59 +1.37 -1.33 km/s/Mpc
# Ωbh2 = 0.02253 +0.00011 -0.00011
# Ωch2 = 0.11866 +0.00118 -0.00115
# Ωmh2 = 0.1418 +0.0011 -0.0011
# Ωm = 0.340 +0.015 -0.014
# σ8 = 0.806 +0.012 -0.012
# S8 = 0.856 +0.027 -0.026
# w0 = -0.77 +0.10 -0.10 (prior U[-1, 0])
# f = 1.87 +0.18 -0.18 (error overestimation factor)
# chi2 = 57.39
# log likelihood = 109.1
# degs of freedom = 53
# ---------------------------------
