from numba import njit


@njit
def r_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="hyrec"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        490.058910691858
        + 17.8849267017991 * wb_scaled**0.22689741 * wm_scaled**0.66428509
        - 55.094029247955 * wb_scaled**0.36313719 * wm_scaled**-0.23994741
        - 1.00470114733618 * wb_scaled**-1.0747657 * wm_scaled**-0.0049968377
        - 0.0463164950392738 * wb_scaled**0.93300333 * wm_scaled**2.9870209
        - 303.160820488599 * wb_scaled**0.011925673 * wm_scaled**0.22281918
        + 3.52235577333832 * wb_scaled**-0.014229026 * wm_scaled**1.2723541
    )


@njit
def z_drag(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="hyrec"
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        413.255527491952
        + 20.4945767460008 * wb_scaled**-0.46977879 * wm_scaled**0.52159161
        + 615.477265657272 * wb_scaled**0.089056562
        - 0.23335032904072 * wb_scaled**0.38904347 * wm_scaled**1.2261564
        + 1.01239259507866 * wb_scaled**0.94978138 * wm_scaled**-0.39184073
        + 3.6305634354568 * wb_scaled**0.80441388 * wm_scaled**0.1308599
        - 0.0545504030444375 * wb_scaled**-2.718635 * wm_scaled**1.8298705
    )


@njit
def z_star(wb, wm):
    # wb = (0.01, 0.04), wm = (0.05, 0.35) recfast_approx_model="hyrec"
    # SPT uses CLASS and the redshift this formula fits is where the visibility function peaks
    # z last scattering
    wb_scaled = wb / 0.02
    wm_scaled = wm / 0.132287565553
    return (
        1062.66319932272
        + 1.93856588145585 * wb_scaled**-0.97513302 * wm_scaled**-0.60026424
        + 19.6180435758368 * wb_scaled**-1.1576553 * wm_scaled**0.63280311
        + 0.01330971113583 * wb_scaled**0.47544644 * wm_scaled**2.187071
        + 20.278265726591 * wb_scaled**-1.0496969 * wm_scaled**0.3954512
        + 0.010696537621192 * wb_scaled**-3.0181396 * wm_scaled**1.4713501
        - 13.6250275002192 * wb_scaled**-1.4193326 * wm_scaled**0.6697137
    )
