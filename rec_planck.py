from numba import njit


@njit
def r_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="planck"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        137.103922151
        + 20.1125208175 * (
            17.5459587192
            + 0.889159693118 * wb_scaled**0.22689741 * wm_scaled**0.66428509
            - 2.73911828811 * wb_scaled**0.36313719 * wm_scaled**-0.23994741
            - 0.0498240010542 * wb_scaled**-1.0747657 * wm_scaled**-0.0049968377
            - 0.00230064224764 * wb_scaled**0.93300333 * wm_scaled**2.9870209
            - 15.070821362 * wb_scaled**0.011925673 * wm_scaled**0.22281918
            + 0.175013692682 * wb_scaled**-0.014229026 * wm_scaled**1.2723541
        )
    )


@njit
def z_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="planck"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return 1066.06318494 + 19.4030520065 * (
        -37.231337046
        + 1.06430561936 * wb_scaled**-0.46977879 * wm_scaled**0.52159161
        + 34.4527514185 * wb_scaled**0.072999021 * wm_scaled**0
        - 0.00639784334316 * wb_scaled**-0.109656 * wm_scaled**1.712214
        + 0.100970358277 * wb_scaled**0.60312578 * wm_scaled**-0.30361202
        + 0.979571465041 * wb_scaled**0.45739675 * wm_scaled**0.028175625
        - 0.00313751428423 * wb_scaled**-2.8661401 * wm_scaled**1.6072032
    )


@njit
def z_star(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="planck"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return 1094.43545185 + 15.5225273161 * (
        -2.3450612289
        + 0.167978227356 * wb_scaled**-1.3127752 * wm_scaled**-0.48217291
        + 0.58184877906 * wb_scaled**-1.0252711 * wm_scaled**0.70492967
        - 0.832125626743 * wb_scaled**-1.3870357 * wm_scaled**0.68620668
        + 2.31110115415 * wb_scaled**-1.0622814 * wm_scaled**0.46828184
        - 0.000354634666613 * wb_scaled**2.1516052 * wm_scaled**-1.8222422
        - 0.0127234446553 * wb_scaled**-2.9661 * wm_scaled**-0.30294611
    )
