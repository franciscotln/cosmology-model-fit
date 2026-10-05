from numba import njit


@njit
def r_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="cosmorec"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        490.06271510029
        + 17.886657056 * wb_scaled**0.22689741 * wm_scaled**0.66428509
        - 55.09469015609 * wb_scaled**0.36313719 * wm_scaled**-0.23994741
        - 1.003746676058 * wb_scaled**-1.0747657 * wm_scaled**-0.0049968377
        - 0.04632007827 * wb_scaled**0.93300333 * wm_scaled**2.9870209
        - 303.1660617269 * wb_scaled**0.011925673 * wm_scaled**0.22281918
        + 3.5221151199 * wb_scaled**-0.014229026 * wm_scaled**1.2723541
    )


@njit
def z_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="cosmorec"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        416.816295694468
        + 20.3953685410092 * wb_scaled**-0.46977879 * wm_scaled**0.52159161
        + 611.127525728776 * wb_scaled**0.089056562 * wm_scaled**0
        - 0.126892744697531 * wb_scaled**0.11346999 * wm_scaled**1.5887055
        + 1.68266733249701 * wb_scaled**0.92095218 * wm_scaled**-0.2708297
        + 3.72966079685775 * wb_scaled**0.70345396 * wm_scaled**0.1308599
        - 0.0461776668082887 * wb_scaled**-2.8915016 * wm_scaled**1.8061516
    )


@njit
def z_star(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="cosmorec"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        1058.0360488657
        + 2.63797402649136 * wb_scaled**-1.3239913 * wm_scaled**-0.48217291
        + 9.00822082254065 * wb_scaled**-1.0252711 * wm_scaled**0.70492967
        - 12.9142963566347 * wb_scaled**-1.3870357 * wm_scaled**0.68620668
        + 35.8792180013677 * wb_scaled**-1.0622814 * wm_scaled**0.46828184
        - 0.00551066641139285 * wb_scaled**2.1516052 * wm_scaled**-1.8222422
        - 0.240138042318547 * wb_scaled**-2.8229421 * wm_scaled**-0.31118155
    )
