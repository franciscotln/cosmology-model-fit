from numba import njit
import numpy as np
import cmb.data_planck_act_compression as cmb

c = cmb.c  # km/s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2
Omnu_z = cmb.Omnu_z


@njit
def Hz(z, params):
    H0, Obh2, Och2 = params
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    radiation = Or * (1.0 + z) ** 4
    cd_matter = Obc * (1.0 + z) ** 3
    dark_energy = Ode
    neutrino = Onu * Omnu_z(z)

    return H0 * np.sqrt(radiation + cd_matter + neutrino + dark_energy)


cmb.set_HZ(Hz)

bounds = np.array(
    [
        (60.0, 75.0),  # H0
        (0.020, 0.025),  # Ωb * h^2
        (0.05, 0.25),  # Ωc * h^2
    ]
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


@njit
def log_likelihood(params):
    Obh2, Och2 = params[1], params[2]
    Omh2 = Obh2 + Och2 + Omnu_h2

    zstar = cmb.z_star(Obh2, Omh2)
    rs_star = cmb.rs_z(zstar, Obh2, params)
    DM_star = cmb.DM_z(zstar, params)
    thetastar = rs_star / DM_star
    lA = np.pi / thetastar
    R = 100 * np.sqrt(Omh2) * DM_star / c  # shift parameter

    delta = cmb.DISTANCE_PRIORS - np.array([R, lA, Obh2])
    log_like = -0.5 * (delta @ cmb.inv_cov_mat @ delta)
    # blobs: (100 θ*, r*, DM* in Gpc, z*)
    return log_like, np.array([100 * thetastar, rs_star, DM_star / 1000, zstar])


@njit
def log_probability_jit(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf, np.empty(4)
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
    burn_in = 1000
    nsteps = 5000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [(emcee.moves.KDEMove(), 0.20), (emcee.moves.DEMove(), 0.80)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, moves=moves)
    sampler.run_mcmc(
        initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
    )

    samples_list = sampler.get_chain(discard=burn_in, flat=False)
    blobs_list = sampler.get_blobs(discard=burn_in, flat=False)
    chain_list = np.moveaxis(np.concatenate([samples_list, blobs_list], axis=2), 1, 0)
    loglikes_list = np.moveaxis(sampler.get_log_prob(discard=burn_in, flat=False), 1, 0)

    names = ["H0", "ombh2", "omch2", "thetastar", "rstar", "DAstar", "zstar"]
    labels = ["H_0", "ω_b", "ω_c", "100θ_*", "r_*", r"D_{\rm{M_*}}/{\rm{Gpc}}", "z_*"]
    samples = MCSamples(
        samples=chain_list,
        loglikes=-loglikes_list,
        names=names,
        labels=labels,
        label="CMB Compressed likelihood",
    )
    samples.addDerived(
        samples["ombh2"] + samples["omch2"] + Omnu_h2, name="omegamh2", label="ω_m"
    )
    samples.addDerived(
        samples["omegamh2"] / (samples["H0"] / 100) ** 2, name="omegam", label="Ω_m"
    )
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
    samples.addDerived(
        -1 + (samples["ombh2"] + samples["omch2"]) / cmb.Omega_r_h2(),
        name="zeq",
        label=r"z_{eq}",
    )
    samples.updateBaseStatistics()

    for name in samples.getParamNames().names:
        print(samples.getInlineLatex(name, limit=1))

    g = plots.getSubplotPlotter()
    params = ["thetastar", "H0", "omegam", "DAstar", "rstar", "zstar", "zdrag", "rdrag"]
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
    print(f"Chi squared: {-2 * log_likelihood(MAP)[0]:.4f}")


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
# r* = 144.45 ± 0.23 Mpc
# DM* = 13.868 ± 0.022 Gpc
# z* = 1088.78 ± 0.14
# ωm = 0.14331 ± 0.00092
# Ωm = 0.3175 ± 0.0056
# z_drag = 1059.95 ± 0.21
# r_d = 147.00 ± 0.24 Mpc
# z_eq = 3410 ± 22
# Chi squared: 0.0008
# -----------------------------


# -----------------------------
# plikHM TT, TE, EE + lowl + lowE compression (Planck 2019 - PR3)
# -----------------------------
# H0: 67.26 ± 0.60 km/s/Mpc
# ωc: 0.1202 ± 0.0014
# ωb: 0.02236 ± 0.00015
# ωm: 0.1432 ± 0.0013
# Ωm: 0.3166 ± 0.0084
# z_eq: 3407 ± 31
# z*: 1089.95 ± 0.27
# r*: 144.39 ± 0.30 Mpc
# 100 θ*: 1.04109 ± 0.00030
# DM*: 13.869 ± 0.028 Gpc
# z_drag: 1059.93 ± 0.30
# r_d: 147.05 ± 0.30 Mpc
# Chi squared: 0.0003
# -----------------------------


# -----------------------------
# plikHM TT, TE, EE + lowl + lowE + Lensing compression (Planck 2019 - PR3)
# -----------------------------
# H0: 67.35 ± 0.54 km/s/Mpc
# ωc: 0.1200 ± 0.0012
# ωb: 0.02237 ± 0.00015
# ωm: 0.1430 ± 0.0011
# Ωm: 0.3153 ± 0.0073
# z_eq: 3402 ± 26
# z*: 1089.92 ± 0.25
# r*: 144.43 ± 0.26 Mpc
# 100 θ*: 1.04110 ± 0.00031
# DM*: 13.873 ± 0.025 Gpc
# z_drag: 1059.94 ± 0.30
# r_d: 147.09 ± 0.26 Mpc
# Chi squared: 0.0004
# -----------------------------


# -----------------------------
# Early ΛCDM (arXiv:2302.12911v2)
# -----------------------------
# H0: 67.49 ± 0.59 km/s/Mpc
# ωc: 0.1192 ± 0.0013
# ωb: 0.02223 ± 0.00015
# ωm: 0.1421 ± 0.0012
# Ωm: 0.3121 ± 0.0080
# z_eq: 3380 ± 29
# z*: 1090.12 ± 0.27
# r*: 144.75 ± 0.28 Mpc
# 100 θ*: 1.04103 ± 0.00026
# DM*: 13.905 ± 0.026 Gpc
# z_drag: 1059.65 ± 0.29
# r_d: 147.46 ± 0.28 Mpc
# Chi squared: 0.0002
# -----------------------------


# -----------------------------
# ACT DR6 compression
# -----------------------------
# H0: 66.10 ± 0.79 km/s/Mpc
# ωc: 0.1238 ± 0.0021
# ωb: 0.02259 ± 0.00017
# ωm: 0.1470 ± 0.0021
# Ωm: 0.337 ± 0.013
# z_eq: 3499 ± 51
# z*: 1089.96 ± 0.30
# r*: 143.31 ± 0.54 Mpc
# 100 θ*: 1.04075 ± 0.00031
# DM*: 13.770 ± 0.051 Gpc
# z_drag: 1060.72 ± 0.39
# r_d: 145.87 ± 0.56 Mpc
# Chi squared: 0.0006
# -----------------------------


# -----------------------------
# ACT DR6 + Planck compression
# -----------------------------
# H0: 67.62 ± 0.50 km/s/Mpc
# ωc: 0.1193 ± 0.0012
# ωb: 0.02250 ± 0.00011
# ωm: 0.1425 ± 0.0012
# Ωm: 0.3117 ± 0.0071
# z_eq: 3390 ± 28
# z*: 1089.68 ± 0.21
# r*: 144.53 ± 0.29 Mpc
# 100 θ*: 1.04094 ± 0.00025
# DM*: 13.884 ± 0.027 Gpc
# z_drag: 1060.17 ± 0.23
# r_d: 147.14 ± 0.29 Mpc
# Chi squared: 0.0002
# -----------------------------
