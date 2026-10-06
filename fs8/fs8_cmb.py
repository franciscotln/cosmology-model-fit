from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_ivp import solve_ivp
from solve_triangular import solve_triangular
import y2018fs8.data as fs8_data
import cmb.data_act_planck_compression as cmb

c = cmb.c_km_per_s
Orh2 = cmb.Or_h2
Omnuh2 = cmb.Omnu_h2

data = fs8_data.data
z_vals = fs8_data.data["z"]
fs8_vals = fs8_data.data["fs8"]
a_vals = 1 / (1.0 + z_vals)

N = len(data)
cho = cho_factor(fs8_data.cov_mat, lower=True)[0]
logdet = 2 * np.sum(np.log(np.diag(cho)))
norm_factor = N * np.log(2 * np.pi) + logdet

z_max = np.max(z_vals) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def w_de_z(z, w0, wa):
    zp1 = 1.0 + z
    return w0 + wa * (zp1**2 - 1) / (zp1**2 + 1)  # w1w2CDM
    # return w0 + wa * z / zp1  # w0waCDM


@njit
def Ode_z(z, w0, wa):
    zp1 = 1.0 + z
    return zp1**(3 * (1.0 + w0 + wa)) * (2 * zp1**2 / (zp1**2 + 1))**(-3 * wa) # w1w2CDM
    # return zp1**(3 * (1.0 + w0 + wa)) * np.exp(-3 * wa * z / zp1) # w0waCDM


@njit
def d_Ode_dz(z, w0, wa):
    return Ode_z(z, w0, wa) * 3 * (1.0 + w_de_z(z, w0, wa)) / (1.0 + z)


@njit
def d_Omnu_dz(z):
    return cmb.Omnu_z(z) * 3 * (1.0 + cmb.w_nu_z(z)) / (1.0 + z)


@njit
def Hz(z, theta):
    H0, Obh2, Och2, w0, wa = theta[0:5]
    h = H0 / 100
    zp1 = 1.0 + z

    radiation_term = Orh2 * zp1**4
    matter_term = (Obh2 + Och2) * zp1**3
    neutrino_term = Omnuh2 * cmb.Omnu_z(z)
    dark_energy_term = (h**2 - Orh2 - Obh2 - Och2 - Omnuh2) * Ode_z(z, w0, wa)
    return 100 * np.sqrt(radiation_term + matter_term + neutrino_term + dark_energy_term)


cmb.set_HZ(Hz)


@njit
def dH_da(z, H_val, theta):
    H0, Obh2, Och2, w0, wa = theta[0:5]
    h = H0 / 100
    Odeh2 = h**2 - Obh2 - Och2 - Orh2 - Omnuh2

    matter = (Obh2 + Och2) * 3 * (1.0 + z) ** 2
    rad = Orh2 * 4 * (1.0 + z) ** 3
    nu = Omnuh2 * d_Omnu_dz(z)
    de = Odeh2 * d_Ode_dz(z, w0, wa)

    numerator = matter + rad + nu + de
    denominator = 2 * H_val / (1.0 + z) ** 2
    return -100**2 *numerator / denominator


@njit
def DM(z, theta):
    dh_grid = c / Hz(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid), dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def growth_ODE(a, y, theta):
    Obc_h2 = theta[1] + theta[2]

    z = 1 / a - 1.0
    H_val = Hz(z, theta)
    dH_da_val = dH_da(z, H_val, theta)

    delta, d_delta_da = y

    source = (3 / 2) * (Obc_h2 / a**5) * delta * (100 / H_val) ** 2
    friction = -(3 / a + dH_da_val / H_val) * d_delta_da
    d2_delta_da = friction + source

    return np.array([d_delta_da, d2_delta_da])


max_z = 500
a_span = np.logspace(np.log10(1 / (1.0 + max_z)), 0, 5_000)


@njit
def fs8_theory(a, theta):
    sol = solve_ivp(
        growth_ODE,
        t_span=(a_span[0], a_span[-1]),
        y0=(a_span[0], 1.0),
        t_eval=a_span,
        rtol=1e-6,
        atol=1e-8,
        args=(theta,),
    )
    delta, d_delta_da = sol.y
    sigma8_0 = theta[-2]
    delta_0 = delta[-1]
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    return a * interp_pchip(a, a_span, d_delta_da) * sigma8_0 / delta_0


Hz_DMz_fid = np.empty(N, dtype=np.float64)
for i in range(N):
    z = z_vals[i]
    Obh2_fid = 0.0222
    w0_fid = -1.0
    wa_fid = 0.0
    log_f_err = 0.0
    Om_fid = data["omega_fid"][i]
    H0_fid = data["H0_fid"][i]
    Och2_fid = Om_fid * (H0_fid / 100) ** 2 - Obh2_fid - Omnuh2
    sig8_fid = data["s8_fid"][i]
    theta_fid = [H0_fid, Obh2_fid, Och2_fid, w0_fid, wa_fid, sig8_fid, log_f_err]
    DM_i = DM(np.array([z]), theta_fid)[0]
    Hz_DMz_fid[i] = Hz(z, theta_fid) * DM_i


@njit
def chi2_fs8(theta):
    q = Hz(z_vals, theta) * DM(z_vals, theta) / Hz_DMz_fid
    delta = fs8_vals - fs8_theory(a_vals, theta) / q
    y = solve_triangular(cho, delta)
    return np.exp(-2 * theta[-1]) * np.dot(y, y)


@njit
def chi_squared(theta):
    return chi2_fs8(theta) + cmb.chi2(theta[1], theta[2], theta)


@njit
def log_likelihood(theta):
    norm_fact = norm_factor + 2 * N * theta[-1]
    return -0.5 * (chi_squared(theta) + norm_fact + cmb.prob_norm)


labels = ["$H_0$", "$Ωbh^2$", "$Ωch^2$", "$w_0$", "$w_a$", "$\\sigma_8$", "$ln(f)$"]
bounds = np.array(
    [
        (50, 80),  # H0
        (0.01, 0.035),  # Ob * h^2
        (0.1, 0.35),  # Oc * h^2
        (-3.0, 1.0),  # w0
        (-3.0, 2.0),  # wa
        (0.5, 1.0),  # sigma8
        (-1.2, 0.0),  # ln(f_err): log of overstimation factor of the errors
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(theta):
    if not np.all((bounds[:, 0] < theta) & (theta < bounds[:, 1])):
        return -np.inf
    if theta[3] + theta[4] >= 0:
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
    burn_in = 2000
    nsteps = 3000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [(emcee.moves.KDEMove(), 0.20), (emcee.moves.DEMove(), 0.80)]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"})

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
        (wa_16, wa_50, wa_84),
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

    MAP_params = samples[np.argmax(log_probs)]

    print(f"H0 = {H0_50:.2f} +{H0_84-H0_50:.2f} -{H0_50-H0_16:.2f} km/s/Mpc")
    print(f"Ωbh2 = {Obh2_50:.5f} +{Obh2_84-Obh2_50:.5f} -{Obh2_50-Obh2_16:.5f}")
    print(f"Ωch2 = {Och2_50:.5f} +{Och2_84-Och2_50:.5f} -{Och2_50-Och2_16:.5f}")
    print(f"Ωmh2 = {Omh2_50:.4f} +{Omh2_84-Omh2_50:.4f} -{Omh2_50-Omh2_16:.4f}")
    print(f"Ωm = {Om_50:.3f} +{Om_84-Om_50:.3f} -{Om_50-Om_16:.3f}")
    print(f"σ8 = {s8_50:.3f} +{s8_84-s8_50:.3f} -{s8_50-s8_16:.3f}")
    print(f"S8 = {S8_50:.3f} +{S8_84-S8_50:.3f} -{S8_50-S8_16:.3f}")
    print(f"w0 = {w0_50:.3f} +{w0_84-w0_50:.3f} -{w0_50-w0_16:.3f}")
    print(f"wa = {wa_50:.3f} +{wa_84-wa_50:.3f} -{wa_50-wa_16:.3f}")
    print(f"ln(f) = {f_50:.2f} +{f_84-f_50:.2f} -{f_50-f_16:.2f}")
    print(f"chi2 = {chi_squared(MAP_params):.2f}")
    print(f"log likelihood = {log_likelihood(MAP_params):.1f}")
    print(f"degs of freedom = {N + len(cmb.DISTANCE_PRIORS) - len(MAP_params)}")

    plot_corner_and_chains(labels, samples, chains_samples)
    plot_predictions(
        fs8_theory=lambda z: fs8_theory(1 / (1 + z), MAP_params),
        data=data,
        q=Hz(z_vals, MAP_params) * DM(z_vals, MAP_params) / Hz_DMz_fid,
        f_err=1 / np.exp(MAP_params[-1]),
    )


if __name__ == "__main__":
    main()


# ----------- flat ΛCDM -----------
# H0 = 67.61 +0.47 -0.47 km/s/Mpc
# Ωbh2 = 0.02249 +0.00011 -0.00011
# Ωch2 = 0.11935 +0.00113 -0.00112
# Ωmh2 = 0.1425 +0.0011 -0.0011
# Ωm = 0.312 +0.007 -0.007
# σ8 = 0.789 +0.009 -0.009
# S8 = 0.802 +0.011 -0.011
# ln(f) = -0.58 +0.10 -0.09
# chi2 = 55.53
# log likelihood = 121.4
# degs of freedom = 54
# ---------------------------------


# ----------- flat wCDM -----------
# H0 = 65.75 +1.41 -1.36 km/s/Mpc
# Ωbh2 = 0.02252 +0.00011 -0.00011
# Ωch2 = 0.11886 +0.00120 -0.00121
# Ωmh2 = 0.1420 +0.0012 -0.0012
# Ωm = 0.328 +0.014 -0.014
# σ8 = 0.798 +0.011 -0.011
# S8 = 0.833 +0.025 -0.025
# w = -0.931 +0.048 -0.050 (prior U[-2, 0])
# ln(f) = -0.59 +0.10 -0.09
# chi2 = 56.80
# log likelihood = 122.4
# degs of freedom = 53
# ---------------------------------


# ---------- flat w0waCDM ---------
# w0 + wa < 0 enforced in the likelihood

# H0 = 64.69 +1.30 -1.22 km/s/Mpc
# Ωbh2 = 0.02250 +0.00011 -0.00011
# Ωch2 = 0.11926 +0.00120 -0.00120
# Ωmh2 = 0.1424 +0.0012 -0.0012
# Ωm = 0.340 +0.014 -0.014
# σ8 = 0.814 +0.013 -0.013
# S8 = 0.865 +0.027 -0.027
# w0 = -0.552 +0.145 -0.145 (prior U[-3.0, 1.0])
# wa = -1.331 +0.505 -0.512 (prior U[-3.0, 2.0])
# ln(f) = -0.64 +0.10 -0.09
# chi2 = 56.71
# log likelihood = 125.9
# degs of freedom = 52
# ---------------------------------


# ---------- flat w1w2CDM ---------
# w1 + w2 < 0 enforced in the likelihood
# w(z) = w1 + w2 * ((1 + z)^2 - 1) / ((1 + z)^2 + 1)
#
# H0 = 64.80 +1.30 -1.25 km/s/Mpc
# Ωbh2 = 0.02250 +0.00011 -0.00011
# Ωch2 = 0.11926 +0.00121 -0.00120
# Ωmh2 = 0.1424 +0.0012 -0.0012
# Ωm = 0.339 +0.014 -0.014
# σ8 = 0.814 +0.013 -0.013
# S8 = 0.864 +0.027 -0.027
# w1 = -0.581 +0.137 -0.138 (prior U[-3.0, 1.0])
# w2 = -1.069 +0.403 -0.420 (prior U[-3.0, 2.0])
# ln(f) = -0.64 +0.10 -0.10
# chi2 = 53.85
# log likelihood = 125.8
# degs of freedom = 52
# ---------------------------------