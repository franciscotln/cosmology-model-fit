import numpy as np
from getdist import loadMCSamples
from scipy.integrate import quad
from scipy.constants import c as c0
from numba import njit

c = c0 / 1000  # km/s


import numpy as np


def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""
    SCALING_FID = 1.001067940891529  # reproduces rdrag from mcsamples

    a1 = 0.00257366
    a2 = 0.05032
    a3 = 0.013
    a4 = 0.7720642
    a5 = 0.24346362
    a6 = 0.00641072
    a7 = 0.5350899
    a8 = 32.7525
    a9 = 0.315473

    term_A_denominator = (a1 * (wb**a2)) + (a3 * (wb**a4) * (wm**a5)) + (a6 * (wm**a7))
    term_A = 1.0 / term_A_denominator
    term_B = a8 / (wm**a9)
    return SCALING_FID * (term_A - term_B)


def z_drag(wb, wm):
    """arXiv:2106.00428v2 (eq A2)"""
    SCALING_FID = 1.000042094274071  # reproduces zdrag from mcsamples

    return (
        SCALING_FID
        * (1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615)
        * wm**-0.714129
    )


@njit
def z_star(wb, wm):
    """arXiv:astro-ph/9510117v2 (eq-1)"""
    SCALING_FID = 0.9981950308412795  # reproduces z* from mcsamples

    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    factor_1 = 1 + 0.00124 * wb**-0.738
    factor_2 = 1 + g1 * wm**g2
    return SCALING_FID * 1048 * factor_1 * factor_2


samples = loadMCSamples(
    "y2025cmb_actbase_lcdm_camb/raw/actbase_lcdm_camb",
    settings={"ignore_rows": 0.3},
)
samples.addDerived(samples["omegam"] * (samples["H0"] / 100) ** 2, name="omegamh2")
samples.addDerived(z_star(samples["ombh2"], samples["omegamh2"]), name="zstar_computed")

N_EFF = 3.044
O_GAMMA_H2 = 2.4729753e-05


@njit
def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.227107 * Neff)


def rs_z(Ez_func, z, params, H0, Ob_h2):
    Rb = 3 * Ob_h2 / (4 * O_GAMMA_H2)

    def integrand(a):
        denom = a**2 * Ez_func(1 / a - 1, params) * np.sqrt(3 * (1 + Rb * a))
        return 1 / denom

    return (c / H0) * quad(integrand, 1e-09, 1 / (1 + z))[0]


def DM_z(Ez_func, z, params, H0):
    I = quad(lambda zp: c / Ez_func(zp, params), 0, z)[0]
    return I / H0


@njit
def Ez(z, params):
    Om, Ol, Or = params
    return np.sqrt(Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + Ol)


n = len(samples.samples)


rstar_values = np.zeros(n, dtype=np.float64)
DMstar_values = np.zeros(n, dtype=np.float64)

Omnuh2 = 0.06 / 93.14

for i in range(n):
    H0 = samples["H0"][i]
    h = H0 / 100
    Ombh2 = samples["ombh2"][i]
    Omch2 = samples["omch2"][i]
    Ombc = (Ombh2 + Omch2) / h**2
    Obh2 = samples["ombh2"][i]
    Ol = samples["omega_de"][i]

    Or_rstar = Omega_r_h2(3.044) / h**2

    Or_DMstar = Omega_r_h2(2.044) / h**2
    massive_neutrino = Omnuh2 / h**2

    zst = samples["zstar"][i]

    rstar_values[i] = rs_z(Ez, zst, (Ombc, Ol, Or_rstar), H0, Obh2)
    DMstar_values[i] = DM_z(Ez, zst, (Ombc + massive_neutrino, Ol, Or_DMstar), H0)

    if i % 2000 == 0:
        print(f"Computed r* for sample {i}/{n}")

samples.addDerived(rstar_values, name="rstar_computed")
samples.addDerived(DMstar_values, name="DMstar_computed")
samples.addDerived(
    100 * samples["rstar_computed"] / samples["DMstar_computed"],
    name="thetastar_computed",
)
print(samples.mean(["thetastar", "thetastar_computed"]))
print(samples.std(["thetastar", "thetastar_computed"]))
print(samples.corr(["thetastar", "thetastar_computed"]))
