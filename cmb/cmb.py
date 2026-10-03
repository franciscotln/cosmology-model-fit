from numba import njit
import numpy as np
from solve_triangular import solve_triangular
import cmb.data_act_planck_compression as cmb

c = cmb.c_km_per_s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

N = len(cmb.DISTANCE_PRIORS)


@njit
def Hz(z, pars):
    h, Ob_h2, Oc_h2 = pars[0], pars[1], pars[2]

    radiation = Or_h2 * (1.0 + z)**4
    cd_matter =  (Ob_h2 + Oc_h2) * (1.0 + z)**3
    neutrino = Omnu_h2 * cmb.Omnu_z(z)
    dark_energy = h**2 - (Or_h2 + Omnu_h2 + Ob_h2 + Oc_h2)

    return 100 * np.sqrt(radiation + cd_matter + neutrino + dark_energy)


cmb.set_HZ(Hz)

bounds = np.array([
    (0.60, 0.75),  # h
    (0.020, 0.025),  # Ωb * h^2
    (0.05, 0.25),  # Ωcdm * h^2
])

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


@njit
def log_likelihood(params):
    Ob_h2, Oc_h2 = params[1], params[2]
    Om_h2 = Ob_h2 + Oc_h2 + Omnu_h2

    zstar = cmb.z_star(Ob_h2, Om_h2)
    rs_star = cmb.rs_z(zstar, Ob_h2, params)
    DM_star = cmb.DM_z(zstar, params)
    thetastar = rs_star / DM_star
    lA = np.pi / thetastar
    R = 100 * np.sqrt(Om_h2) * DM_star / c

    delta = cmb.DISTANCE_PRIORS - np.array([R, lA, Ob_h2])
    y = solve_triangular(cmb.L, delta)
    chi2 = np.dot(y, y)
    log_like = -0.5 * (chi2 + cmb.prob_norm)

    # blobs: (100 θ*, r*, DM* in Gpc, z*, R)
    blobs = np.array([100 * thetastar, rs_star, DM_star / 1000, zstar, R])
    return log_like, blobs


@njit
def log_probability_jit(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf, np.empty(5)
    ll, blobs = log_likelihood(params)
    return lp + ll, blobs


def log_probability(params):
    return log_probability_jit(params)


def main():
    import emcee
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    ndim = len(bounds)
    nwalkers = 200
    burn_in = 2000
    nsteps = 8000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [(emcee.moves.KDEMove(), 0.20), (emcee.moves.DEMove(), 0.80)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, moves=moves)
    sampler.run_mcmc(initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"})

    samples_list = sampler.get_chain(discard=burn_in, flat=False)
    blobs_list = sampler.get_blobs(discard=burn_in, flat=False)
    chain_list = np.moveaxis(np.concatenate([samples_list, blobs_list], axis=2), 1, 0)
    loglikes_list = np.moveaxis(sampler.get_log_prob(discard=burn_in, flat=False), 1, 0)

    names = ["h", "ombh2", "omch2", "thetastar", "rstar", "DAstar", "zstar", "R"]
    labels = ["h", "ω_b", "ω_c", "100θ_*", "r_*", r"D_{\rm{M_*}}/{\rm{Gpc}}", "z_*", "R"]
    samples = MCSamples(
        samples=chain_list,
        loglikes=-loglikes_list,
        names=names,
        labels=labels,
        label="CMB Compressed likelihood",
    )
    samples.addDerived(100 * samples["h"], name="H0", label="H_0")
    samples.addDerived(samples["ombh2"] + samples["omch2"] + Omnu_h2, name="omegamh2", label="ω_m")
    samples.addDerived(samples["omegamh2"] / samples["h"] ** 2, name="omegam", label="Ω_m")
    samples.addDerived(
        cmb.z_drag(samples["ombh2"], samples["omegamh2"]),
        name="zdrag",
        label=r"z_{drag}",
    )
    samples.addDerived(
        cmb.r_drag(samples["ombh2"], samples["omegamh2"]),
        name="rdrag",
        label=r"r_{drag}",
    )
    samples.addDerived(samples["rdrag"] * samples["h"], name="hrd", label="h r_d")
    samples.addDerived(
        -1 + (samples["ombh2"] + samples["omch2"]) / cmb.Omega_r_h2(),
        name="zeq",
        label=r"z_{eq}",
    )
    samples.updateBaseStatistics()

    for name in samples.getParamNames().names:
        print(samples.getInlineLatex(name, limit=1))

    g = plots.getSubplotPlotter()
    params = ["H0", "omegam", "thetastar", "rdrag"]
    g.triangle_plot(
        samples,
        params=params,
        filled=True,
        title_limit=1,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    MAP = flat_samples[np.argmax(log_probs)]
    log_like_map = log_likelihood(MAP)[0]
    print(f"χ2 (MAP): {-2 * log_like_map - cmb.logdet - N * np.log(2 * np.pi):.3f}")
    print(f"log likelihood (MAP): {log_like_map:.2f}")


if __name__ == "__main__":
    main()


# Model: flat ΛCDM


# -----------------------------
# SPT + Planck PR4 + ACT DR6 compression (2026)
# -----------------------------
# H0 = 67.19 ± 0.38 km/s/Mpc
# ωb = 0.022399 ± 0.000095
# ωc = 0.12027 ± 0.00094
# 100 θ* = 1.04161 ± 0.00023
# r_rec = 144.45 ± 0.23 Mpc
# DM_rec = 13.868 ± 0.022 Gpc
# z_rec = 1088.77 ± 0.13
# R = 1.7512 ± 0.0030
# ωm = 0.14332 ± 0.00091
# Ωm = 0.3175 ± 0.0056 (1.7 sigma tension with BAO)
# h * r_d = 98.77 ± 0.69 (2.5 sigma tension with BAO)
# z_drag = 1060.00 ± 0.21
# r_d = 147.00 ± 0.24 Mpc
# z_eq = 3410 ± 22
# χ2 (MAP): 0.000
# log likelihood (MAP): 21.92
# -----------------------------


# -----------------------------
# plikHM TT, TE, EE + lowl + lowE compression (Planck 2019 - PR3)
# -----------------------------
# H0: 67.27 ± 0.60 km/s/Mpc
# ωb: 0.02236 ± 0.00015
# ωc: 0.1202 ± 0.0014
# 100 θ*: 1.04109 ± 0.00030
# r*: 144.39 ± 0.30 Mpc
# DM*: 13.869 ± 0.028 Gpc
# z*: 1089.95 ± 0.27
# R: 1.7507 ± 0.0046
# ωm: 0.1432 ± 0.0013
# Ωm: 0.3166 ± 0.0084
# h * r_d = 98.9 ± 1.0
# z_drag: 1059.93 ± 0.30
# r_d: 147.06 ± 0.30 Mpc
# z_eq: 3407 ± 31
# Age: 13.801 ± 0.024 Gyr
# χ2 (MAP): 0.000
# log likelihood (MAP): 14.26
# -----------------------------


# -----------------------------
# plikHM TT, TE, EE + lowl + lowE + Lensing compression (Planck 2019 - PR3)
# -----------------------------
# H0: 67.36 ± 0.54 km/s/Mpc
# ωb: 0.02237 ± 0.00015
# ωc: 0.1200 ± 0.0012
# 100 θ*: 1.04110 ± 0.00031
# r*: 144.43 ± 0.26 Mpc
# DM*: 13.873 ± 0.025 Gpc
# z*: 1089.92 ± 0.25
# R: 1.7500 ± 0.0040
# ωm: 0.1430 ± 0.0011
# Ωm: 0.3153 ± 0.0073
# h * r_d = 99.08 ± 0.92
# z_drag: 1059.94 ± 0.30
# r_d: 147.10 ± 0.26 Mpc
# z_eq: 3402 ± 27
# Age: 13.798 ± 0.023 Gyr
# χ2 (MAP): 0.000
# log likelihood (MAP): 14.39
# -----------------------------


# -----------------------------
# Early ΛCDM (arXiv:2302.12911v2)
# -----------------------------
# H0 = 67.49 ± 0.58 km/s/Mpc
# ωb = 0.02223 ± 0.00015
# ωc = 0.1192 ± 0.0013
# 100 θ* = 1.04103 ± 0.00026
# r* = 144.75 ± 0.28 Mpc
# DM* = 13.904 ± 0.026 Gpc
# z* = 1090.02 ± 0.27
# R = 1.7481 ± 0.0044
# ωm = 0.1421 ± 0.0012
# Ωm = 0.3120 ± 0.0080 (0.9 sigma tension with BAO)
# h * r_d = 99.5 ± 1.0 (1.4 sigma tension with BAO)
# z_drag = 1059.56 ± 0.29
# r_d = 147.46 ± 0.28 Mpc
# z_eq = 3379 ± 29
# χ2 (MAP): 0.000
# log likelihood (MAP): 21.30
# -----------------------------


# -----------------------------
# ACT DR6 compression
# -----------------------------
# H0: 66.11 ± 0.79 km/s/Mpc
# ωb: 0.02259 ± 0.00017
# ωc: 0.1238 ± 0.0021
# 100 θ*: 1.04075 ± 0.00031
# r*: 143.31 ± 0.54 Mpc
# DM*: 13.770 ± 0.050 Gpc
# z*: 1089.96 ± 0.30
# R: 1.7612 ± 0.0065
# ωm: 0.1470 ± 0.0021
# Ωm: 0.337 ± 0.013 (2.3 sigma tension with BAO)
# h * r_d = 96.4 ± 1.5
# z_drag: 1060.72 ± 0.39
# r_d: 145.87 ± 0.56 Mpc
# z_eq: 3499 ± 51
# h * r_d = 96.4 ± 1.5 (2.9 sigma tension with BAO)
# Age: 13.790 ± 0.018 Gyr
# χ2 (MAP): 0.000
# log likelihood (MAP): 13.52
# -----------------------------


# -----------------------------
# ACT DR6 + Planck compression
# -----------------------------
# H0: 67.62 ± 0.50 km/s/Mpc
# ωb: 0.02250 ± 0.00011
# ωc: 0.1193 ± 0.0012
# 100 θ*: 1.04094 ± 0.00025
# r*: 144.52 ± 0.29 Mpc
# DM*: 13.884 ± 0.027 Gpc
# z*: 1089.68 ± 0.21
# R: 1.7480 ± 0.0039
# ωm: 0.1425 ± 0.0012
# Ωm: 0.3117 ± 0.0071
# h * r_d = 99.50 ± 0.91
# z_drag: 1060.17 ± 0.23
# r_d: 147.14 ± 0.29 Mpc
# z_eq: 3390 ± 28
# Age: 13.802 ± 0.023 Gyr
# χ2 (MAP): 0.000
# log likelihood (MAP): 14.69
# -----------------------------
