from numba import njit


@njit
def r_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="planck"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        489.9973821539056
        + 17.883242837917686 * wb_scaled**0.22689741 * wm_scaled**0.66428509
        - 55.09057359120734 * wb_scaled**0.36313719 * wm_scaled**-0.23994741
        - 1.0020862584137394 * wb_scaled**-1.0747657 * wm_scaled**-0.0049968377
        - 0.04627171509927949 * wb_scaled**0.93300333 * wm_scaled**2.9870209
        - 303.1122083800487 * wb_scaled**0.011925673 * wm_scaled**0.22281918
        + 3.5199665374142724 * wb_scaled**-0.014229026 * wm_scaled**1.2723541
    )


@njit
def z_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="planck"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        343.6616159649319
        + 20.650777283252273 * wb_scaled**-0.46977879 * wm_scaled**0.52159161
        + 668.4885275401721 * wb_scaled**0.072999021
        - 0.1241376871167733 * wb_scaled**-0.109656 * wm_scaled**1.712214
        + 1.9591331127635687 * wb_scaled**0.60312578 * wm_scaled**-0.30361202
        + 19.00667608027392 * wb_scaled**0.45739675 * wm_scaled**0.028175625
        - 0.06087735282805131 * wb_scaled**-2.8661401 * wm_scaled**1.6072032
    )


@njit
def z_star(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="planck"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        1058.0341748664727
        + 2.6074466226435663 * wb_scaled**-1.3127752 * wm_scaled**-0.48217291
        + 9.031763566798284 * wb_scaled**-1.0252711 * wm_scaled**0.70492967
        - 12.91669277154505 * wb_scaled**-1.3870357 * wm_scaled**0.68620668
        + 35.87413079556361 * wb_scaled**-1.0622814 * wm_scaled**0.46828184
        - 0.005504826299736309 * wb_scaled**2.1516052 * wm_scaled**-1.8222422
        - 0.1975000172167808 * wb_scaled**-2.9661 * wm_scaled**-0.30294611
    )
