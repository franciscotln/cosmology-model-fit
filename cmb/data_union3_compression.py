import numpy as np
from scipy.integrate import quad
from scipy.constants import c as c0

c = c0 / 1000  # km/s

# --- PLANCK DISTANCE PRIORS (Rubin+ arXiv:2311.12098v2) ---
# θ ≡ rs(z*) / DM(z*)
# R ≡ √(Ωm H0²) * DA(z*) * (1 + z*) / c
DISTANCE_PRIORS = np.array(
    [
        1.7492768568335353,  # R
        1.039233410719115,  # 100 θ
        0.02239245,  # ωb
    ]
)
inv_cov_mat = np.array(
    [
        [92701.58172970748, 348041.8137694254, 1613445.8550364415],
        [348041.8137694254, 13114681.644682042, -3019007.1687636944],
        [1613445.8550364415, -3019007.1687636944, 80842256.32398143],
    ]
)
N_EFF = 3.04
TCMB = 2.72548  # K
O_GAMMA_H2 = 2.4729e-5 * (TCMB / 2.72548) ** 4


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


def z_star(wb, wm):
    # arXiv:2106.00428v2 (A4)
    return wm**-0.731631 + (
        (391.672 * wm**-0.372296 + 937.422 * wb**-0.97966) * wm**0.0192951 * wb**0.93681
    )


def z_drag(wb, wm):
    # arXiv:2106.00428v2 (A2)
    return (
        1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615
    ) * wm**-0.714129


def rs_z(Ez_func, z, H0, Ob_h2):
    Rb = 3 * Ob_h2 / (4 * O_GAMMA_H2)

    def integrand(zp):
        denom = Ez_func(zp) * np.sqrt(3 * (1 + Rb / (1 + zp)))
        return 1 / denom

    z_lower = z
    z_upper = np.inf
    I = quad(integrand, z_lower, z_upper, limit=100)[0]
    return (c / H0) * I


def DA_z(Ez_func, z, H0):
    integral = quad(lambda zp: 1 / Ez_func(zp), 0, z)[0]
    return (c / H0) * integral / (1 + z)


def cmb_distances(Ez_func, H0, Om, Ob_h2):
    Om_h2 = Om * (H0 / 100) ** 2
    zstar = z_star(Ob_h2, Om_h2)
    rs_star = rs_z(Ez_func, zstar, H0, Ob_h2)
    DA_star = DA_z(Ez_func, zstar, H0)

    R = np.sqrt(Om) * H0 * (1 + zstar) * DA_star / c
    theta_100 = 100 * rs_star / ((1 + zstar) * DA_star)
    return np.array([R, theta_100, Ob_h2])


def r_drag(wb, wm):
    # arXiv:2106.00428v2 (eq 6)
    numerator = 45.5337 * np.log(7.20376 / wm)
    denominator = np.sqrt(1 + 9.98592 * (wb**0.801347))
    return numerator / denominator
