from numba import njit


@njit
def r_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="cosmorec"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return 137.113583315 + 20.1151852785 * (
        17.5464022279
        + 0.889211648232 * wb_scaled**0.22689741 * wm_scaled**0.66428509
        - 2.73896011363 * wb_scaled**0.36313719 * wm_scaled**-0.23994741
        - 0.0498999468392 * wb_scaled**-1.0747657 * wm_scaled**-0.0049968377
        - 0.00230274181576 * wb_scaled**0.93300333 * wm_scaled**2.9870209
        - 15.0715023267 * wb_scaled**0.011925673 * wm_scaled**0.22281918
        + 0.175097324294 * wb_scaled**-0.014229026 * wm_scaled**1.2723541
    )


@njit
def z_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="cosmorec"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return 1066.05092094 + 19.3964459251 * (
        -33.4718343635
        + 1.05150029133 * wb_scaled**-0.46977879 * wm_scaled**0.52159161
        + 31.5071909611 * wb_scaled**0.089056562 * wm_scaled**0
        - 0.00654206163271 * wb_scaled**0.11346999 * wm_scaled**1.5887055
        + 0.0867513223296 * wb_scaled**0.92095218 * wm_scaled**-0.2708297
        + 0.192285783244 * wb_scaled**0.70345396 * wm_scaled**0.1308599
        - 0.00238072825231 * wb_scaled**-2.8915016 * wm_scaled**1.8061516
    )


@njit
def z_star(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="cosmorec"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return 1094.40610929 + 15.5055973416 * (
        -2.3456084679 \
        + 0.170130435376 * wb_scaled**-1.3239913 * wm_scaled**-0.48217291
        + 0.58096573928 * wb_scaled**-1.0252711 * wm_scaled**0.70492967
        - 0.832879641598 * wb_scaled**-1.3870357 * wm_scaled**0.68620668
        + 2.31395264632 * wb_scaled**-1.0622814 * wm_scaled**0.46828184
        - 0.000355398524158 * wb_scaled**2.1516052 * wm_scaled**-1.8222422
        - 0.0154871842102 * wb_scaled**-2.8229421 * wm_scaled**-0.31118155
    )
