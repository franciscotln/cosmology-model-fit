"""
arXiv:2503.14452v2
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_lcdm_get.html
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_info.html#:~:text=Throughout%20CH25%20and%20L25%20we,for%20the%20cobaya%20MCMC%20sampler.
https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_prod_table.html
"""

import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# 100 x θ*, rdrag
DISTANCE_PRIORS = np.array([1.04094236, 147.13946054], dtype=np.float64)
covariance = 1e-05 * np.array(
    [
        [6.61822738e-03, 1.49110073],
        [1.49110073, 8.61796956e+03],
    ],
    dtype=np.float64,
)
inv_cov_mat = np.linalg.inv(covariance)

N_EFF = 3.044
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4729e-05


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


def z_star(wb, wm):
    """arXiv:astro-ph/9510117v2 (eq-1)"""
    SCALING_FID = 0.9969799

    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    factor_1 = 1 + 0.00124 * wb**-0.738
    factor_2 = 1 + g1 * wm**g2
    return SCALING_FID * 1048 * factor_1 * factor_2


def r_drag(wb, wm):
    """arXiv:2106.00428v2 (eq 8)"""
    SCALING_FID = 1.00110357

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
    SCALING_FID = 1.00003192

    return (
        SCALING_FID
        * (1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615)
        * wm**-0.714129
    )


def rs_z(Ez_func, z, params, H0, Ob_h2):
    Rb = 3 * Ob_h2 / (4 * O_GAMMA_H2)

    def integrand(a):
        denom = a**2 * Ez_func(1 / a - 1, params) * np.sqrt(3 * (1 + Rb * a))
        return c / denom

    return quad(integrand, 0, 1 / (1 + z))[0] / H0


def DA_z(Ez_func, z, params, H0):
    I = quad(lambda zp: c / Ez_func(zp, params), 0, z)[0]
    return (I / H0) / (1.0 + z)


def cmb_distances(Ez_func, params, H0, Om, Ob_h2):
    Om_h2 = Om * (H0 / 100) ** 2
    rs_drag = r_drag(wb=Ob_h2, wm=Om_h2)
    zstar = z_star(wb=Ob_h2, wm=Om_h2)
    rs_star = rs_z(Ez_func, zstar, params, H0, Ob_h2)
    DM_star = (1 + zstar) * DA_z(Ez_func, zstar, params, H0)
    thetastar = rs_star / DM_star
    return np.array([100 * thetastar, rs_drag], dtype=np.float64)


"""
correlation matrix using r_drag scaled properly:
             rdrag      rdrag_comp
rdrag      [[1.         0.99999535]
rdrag_comp  [0.99999535 1.        ]]

rdrag: 147.13946054 +/- 0.29356378
rdrag_comp: 147.13946034 +/- 0.29215137
"""

"""
correlation matrix using z_star formula:
                 thetastar  thetastar_comp
thetastar      [[1.         0.99973778]
thetastar_comp  [0.99973778 1.        ]]

thetastar, thetastar_comp: [1.04094236 1.04094857]
thetastar_std, thetastar_comp_std: [0.00025726 0.00025633]
"""
