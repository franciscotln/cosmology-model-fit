import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from y2022pantheonSHOES.data import get_data

legend, z_values, distance_modulus_values, sigma_distance_moduli = get_data()

z_vals = np.array([
    0.38,
    0.51,
    0.70,
    0.85,
    1.48,
    2.33,
    2.33,
], dtype= np.float64)

distances_DM = np.array([
    10.27,
    13.38,
    17.65,
    19.50,
    30.21,
    37.60,
    37.30,
], dtype= np.float64)

sigma_distances_DM = np.array([
    0.15,
    0.18,
    0.30,
    1.00,
    0.79,
    1.90,
    1.70,
], dtype= np.float64)

distances_DH = np.array([
    24.89,
    22.43,
    19.78,
    19.60,
    13.23,
    8.93,
    9.08,
], dtype= np.float64)

sigma_distances_DH = np.array([
    0.58,
    0.48,
    0.46,
    2.10,
    0.47,
    0.28,
    0.34,
], dtype= np.float64)

# z_vals = np.array([
#     0.15,
#     0.38,
#     0.51,
#     0.61,
#     0.85,
#     1.48,
#     2.33,
# ], dtype= np.float64)

# distances_DM = np.array([
#     4.466,
#     10.27,
#     13.38,
#     14.99,
#     18.91,
#     30.21,
#     37.50,
# ], dtype= np.float64)

# sigma_distances_DM = np.array([
#     0.168,
#     0.15,
#     0.13,
#     0.15,
#     0.35,
#     0.79,
#     1.20,
# ], dtype= np.float64)

# distances_DH = np.array([
#     23.392,
#     24.89,
#     22.43,
#     21.09,
#     19.27,
#     13.23,
#     8.99,
# ], dtype= np.float64)

# sigma_distances_DH = np.array([
#     0.758,
#     0.68,
#     0.57,
#     0.52,
#     0.52,
#     0.47,
#     0.28,
# ], dtype= np.float64)

# Speed of light (km/s)
C = 299792.458


def integral_of_e_z(zs, omega_m):
    i = 0
    res = np.empty((len(zs),), dtype=np.float64)
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([(1 / np.sqrt(omega_m * (1 + z) ** 3 + 1 - omega_m)) for z in z_axis], x=z_axis)
        res[i] = integ
        i = i + 1
    return res


def model_distance_modulus(z, h0, omega_m):
    a0_over_ae = 1 + z
    comoving_distance = (C / h0) * integral_of_e_z(zs = z, omega_m=omega_m)
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)


# Fit the curve to the data
[params_opt, params_cov] = curve_fit(
    f=model_distance_modulus,
    xdata=z_values,
    ydata=distance_modulus_values,
    sigma=sigma_distance_moduli,
    absolute_sigma=True,
    p0=[70, 0.3]
)

# Extract the optimal values for H0 = 71.53 ± 0.22 and p0 = Ω_m = 0.285 ± 0.013 for y2018pantheon
[h0, omega_m] = params_opt
[h0_std, omega_m_std] = np.sqrt(np.diag(params_cov))


def DM(z):
    comoving_distance = (C / h0) * integral_of_e_z(zs = z, omega_m=omega_m)
    return comoving_distance


def DH(z):
    a0_over_ae = 1 + z
    return C / (h0 *  np.sqrt(omega_m * a0_over_ae ** 3 + (1 - omega_m)))


# DV has a maximum at z = 1.441 where DV = 900.805 Mpc
def DV(z):
    return (z * DH(z) * DM(z) ** 2) ** (1/3)


z_range = np.linspace(0.01, 3, 100)

# r_d: fit sound horizon
def transverse(z, r_d):
    return DM(z=z) / r_d


def radial(z, r_d):
    return DH(z=z) / r_d


[transverse_params_opt, transverse_params_cov] = curve_fit(
    f=transverse, xdata=z_vals, ydata=distances_DM, sigma=sigma_distances_DM, absolute_sigma=True, p0=[141.8]
)

[r_d_trans] = transverse_params_opt
[r_d_trans_std] = np.sqrt(np.diag(transverse_params_cov))

[radial_params_opt, radial_params_cov] = curve_fit(
    f=radial, xdata=z_vals, ydata=distances_DH, sigma=sigma_distances_DH, absolute_sigma=True, p0=[141.8]
)

[r_d_rad] = radial_params_opt
[r_d_rad_std] = np.sqrt(np.diag(radial_params_cov))

print("r_d transverse:", f"{r_d_trans:.2f} ± {r_d_trans_std:.2f}")
print("r_d radial:", f"{r_d_rad:.2f} ± {r_d_rad_std:.2f}")


def transform_vals(v):
    return v

# Plotting
figure, (trans, rad) = plt.subplots(2, 1)

trans.plot(z_range, transverse(z=z_range, r_d=r_d_trans) / transform_vals(z_range), 'r-')
trans.scatter(x=z_vals, y=distances_DM / transform_vals(z_vals), label=legend, marker='.', alpha=0.6)
trans.errorbar(x=z_vals, y=distances_DM / transform_vals(z_vals), yerr=sigma_distances_DM, fmt='|', capsize=2)
trans.set_xscale('log')
trans.set_ylabel('DM/rd * √z')

rad.plot(z_range, radial(z=z_range, r_d=r_d_rad) * transform_vals(z_range), 'g-')
rad.scatter(x=z_vals, y=distances_DH * transform_vals(z_vals), label=legend, marker='.', alpha=0.6)
rad.errorbar(x=z_vals, y=distances_DH * transform_vals(z_vals), yerr=sigma_distances_DH, fmt='|', capsize=2)
rad.set_xscale('log')
rad.set_ylabel('√z * DH/rd')
rad.set_xlabel('z')

plt.show()
