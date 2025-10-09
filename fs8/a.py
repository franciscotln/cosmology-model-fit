import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


z_data = np.array(
    [
        0.35,
        0.77,
        0.17,
        0.02,
        0.02,
        0.25,
        0.37,
        0.25,
        0.37,
        0.44,
        0.60,
        0.73,
        0.067,
        0.30,
        0.40,
        0.50,
        0.60,
        0.80,
        0.35,
        0.18,
        0.38,
        0.32,
        0.32,
        0.57,
        0.15,
        0.10,
        1.40,
        0.59,
        0.38,
        0.51,
        0.61,
        0.38,
        0.51,
        0.61,
        0.76,
        1.05,
        0.32,
        0.57,
        0.727,
        0.02,
        0.6,
        0.86,
        0.60,
        0.86,
        0.1,
        0.001,
        0.85,
        0.31,
        0.36,
        0.40,
        0.44,
        0.48,
        0.52,
        0.56,
        0.59,
        0.64,
        0.1,
        1.52,
        1.52,
        0.978,
        1.23,
        1.526,
        1.944,
    ]
)
fs8_data = np.array(
    [
        0.440,
        0.490,
        0.510,
        0.314,
        0.398,
        0.3512,
        0.4602,
        0.3665,
        0.4031,
        0.413,
        0.390,
        0.437,
        0.423,
        0.407,
        0.419,
        0.427,
        0.433,
        0.470,
        0.429,
        0.360,
        0.440,
        0.384,
        0.48,
        0.417,
        0.490,
        0.370,
        0.482,
        0.488,
        0.497,
        0.458,
        0.436,
        0.477,
        0.453,
        0.410,
        0.440,
        0.280,
        0.427,
        0.426,
        0.296,
        0.428,
        0.48,
        0.48,
        0.550,
        0.400,
        0.48,
        0.505,
        0.45,
        0.469,
        0.474,
        0.473,
        0.481,
        0.482,
        0.488,
        0.482,
        0.481,
        0.486,
        0.376,
        0.420,
        0.396,
        0.379,
        0.385,
        0.342,
        0.364,
    ]
)
err_data = np.array(
    [
        0.050,
        0.18,
        0.060,
        0.048,
        0.065,
        0.0583,
        0.0378,
        0.0601,
        0.0586,
        0.080,
        0.063,
        0.072,
        0.055,
        0.055,
        0.041,
        0.043,
        0.067,
        0.080,
        0.089,
        0.090,
        0.060,
        0.095,
        0.10,
        0.045,
        0.145,
        0.130,
        0.116,
        0.060,
        0.045,
        0.038,
        0.034,
        0.051,
        0.050,
        0.044,
        0.040,
        0.080,
        0.056,
        0.029,
        0.0765,
        0.0465,
        0.12,
        0.10,
        0.120,
        0.110,
        0.16,
        0.085,
        0.11,
        0.098,
        0.097,
        0.086,
        0.076,
        0.067,
        0.065,
        0.067,
        0.066,
        0.070,
        0.038,
        0.076,
        0.079,
        0.176,
        0.099,
        0.070,
        0.106,
    ]
)
Om_fid = np.array(
    [
        0.25,
        0.25,
        0.3,
        0.266,
        0.3,
        0.276,
        0.276,
        0.276,
        0.276,
        0.27,
        0.27,
        0.27,
        0.27,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.27,
        0.27,
        0.274,
        0.274,
        0.274,
        0.31,
        0.3,
        0.27,
        0.307115,
        0.31,
        0.31,
        0.31,
        0.31,
        0.31,
        0.31,
        0.308,
        0.308,
        0.31,
        0.31,
        0.31,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.25,
        0.3121,
        0.3,
        0.307,
        0.307,
        0.307,
        0.307,
        0.307,
        0.307,
        0.307,
        0.307,
        0.307,
        0.282,
        0.26479,
        0.31,
        0.31,
        0.31,
        0.31,
        0.31,
    ]
)

# Planck15 parameters
Om_planck = 0.3156
s8_planck = 0.831

# WMAP7 parameters (for comparison)
Om_wmap = 0.266
s8_wmap = 0.801

default_w0 = -1


def E(z, Om, w0):
    add_z = 1 + z
    rho_de = (2 * add_z**3 / (1 + add_z**3)) ** (2 * (1 + w0))
    return np.sqrt(Om * add_z**3 + (1 - Om) * rho_de)


def int_Einv(z, Om, w0):
    integrand = lambda zp: 1 / E(zp, Om, w0=w0)
    return quad(integrand, 0, z)[0]


def compute_q(z, Om, Omf, w0):
    if Om == Omf:
        return 1.0
    return E(z, Om, w0) * int_Einv(z, Omf, w0) / (E(z, Omf, w0) * int_Einv(z, Om, w0))


# ODE for growth
def growth_deriv(y, a, Om, w0):
    if a == 0:
        return [0, 0]
    z = 1 / a - 1
    H = E(z, Om, w0)
    HH = H**2
    dHHda = -3 * Om / a**4
    Hprime = (1 / 2) * dHHda / H
    ga = 0
    Geff = 1 + ga * (1 - a) ** 2 - ga * (1 - a) ** 4
    ddelta = y[1]
    ddeltada = -(3 / a + Hprime / H) * y[1] + (3 / 2) * (Om / a**5) / HH * Geff * y[0]
    return np.array([ddelta, ddeltada])


def compute_fs8(zs, Om, sigma8, w0, a_vals=np.logspace(-3, 0, 1000)):

    sol = solve_ivp(
        fun=lambda a, y: growth_deriv(y, a, Om, w0),
        t_span=(a_vals[0], a_vals[-1]),
        y0=[a_vals[0], 1.0],
        t_eval=a_vals,
        rtol=1e-6,
        atol=1e-8,
    )
    delta = sol.y[0]
    ddelta = sol.y[1]

    delta_func = interp1d(a_vals, delta)
    ddelta_func = interp1d(a_vals, ddelta)
    fs8 = np.empty(zs.size, dtype=np.float64)
    for i, z in enumerate(zs):
        a_z = 1 / (1 + z)
        fs8[i] = sigma8 * a_z * ddelta_func(a_z) / delta_func(1.0)
    return fs8


def chi2_lcdm(params, z_d, fs8_d, err_d, Omf_d):
    Om, sigma8 = params
    w0 = -1
    fs8_th = compute_fs8(z_d, Om, sigma8, w0)
    q = np.array([compute_q(zi, Om, Omfi, w0) for zi, Omfi in zip(z_d, Omf_d)])
    fs8_corr = fs8_d * q
    err_corr = err_d * q
    return np.sum(((fs8_th - fs8_corr) / err_corr) ** 2)


# Parametrization fit function
def fs8_param(z, lamb, gamma, beta, Om, sigma8, w0):
    Omz = Om * (1 + z) ** 3 / E(z, Om, w0) ** 2
    return lamb * sigma8 * Omz**gamma / (1 + z) ** beta


def fit_parametrization(Om, sigma8, w0, z_fit=np.linspace(0, 2, 100)):
    popt, _ = curve_fit(
        lambda z, l, g, b: fs8_param(z, l, g, b, Om, sigma8, w0),
        z_fit,
        compute_fs8(z_fit, Om, sigma8, w0),
        p0=[1.5, 0.57, 1.0],
    )
    return popt


# Main reproduction
res_lcdm_full = minimize(
    chi2_lcdm,
    [0.3, 0.8],
    args=(z_data, fs8_data, err_data, Om_fid),
    bounds=((0.1, 0.5), (0.5, 1.0)),
)
Om_bf, s8_bf = res_lcdm_full.x
print(f"Best fit LCDM (full): Om = {Om_bf:.3f}, sigma8 = {s8_bf:.3f}")


# Early 20
early_slice = slice(0, 20)
res_lcdm_early = minimize(
    chi2_lcdm,
    [0.3, 0.8],
    args=(
        z_data[early_slice],
        fs8_data[early_slice],
        err_data[early_slice],
        Om_fid[early_slice],
    ),
    bounds=((0.1, 0.5), (0.5, 1.0)),
)
print(
    f"Best fit LCDM (early 20): Om = {res_lcdm_early.x[0]:.3f}, sigma8 = {res_lcdm_early.x[1]:.3f}"
)


# Late 20
late_slice = slice(-20, None)
res_lcdm_late = minimize(
    chi2_lcdm,
    [0.3, 0.8],
    args=(
        z_data[late_slice],
        fs8_data[late_slice],
        err_data[late_slice],
        Om_fid[late_slice],
    ),
    bounds=((0.1, 0.5), (0.5, 1.0)),
)
print(
    f"Best fit LCDM (late 20): Om = {res_lcdm_late.x[0]:.3f}, sigma8 = {res_lcdm_late.x[1]:.3f}"
)


popt = fit_parametrization(0.27, 0.8, -1)
print(
    f"Parametrization (Om=0.30, s8=0.77, ga=0): lambda={popt[0]:.3f}, gamma={popt[1]:.3f}, beta={popt[2]:.3f}"
)

z_plot = np.linspace(0, 2, 100)

fs8_numeric = compute_fs8(z_plot, 0.27, 0.8, -1)
fs8_parametric = fs8_param(z_plot, *popt, 0.27, 0.8, -1)

plt.plot(z_plot, fs8_numeric, "b.", label="Numerical")
plt.plot(z_plot, fs8_parametric, "r-", label="Param")
plt.xlabel("z")
plt.ylabel("$f\\sigma_8(z)$")
plt.legend()
plt.show()

fs8_planck = compute_fs8(z_plot, Om_planck, s8_planck, -1)
fs8_wmap = compute_fs8(z_plot, Om_wmap, s8_wmap, -1)
fs8_bf = compute_fs8(z_plot, Om_bf, s8_bf, -1)

plt.errorbar(z_data, fs8_data, err_data, fmt="k.", label="Full data")
plt.errorbar(z_data[:20], fs8_data[:20], err_data[:20], fmt="r.", label="Early 20")
plt.errorbar(z_data[-20:], fs8_data[-20:], err_data[-20:], fmt=".", label="Late 20")
plt.plot(z_plot, fs8_planck, "r--", label="Planck15/LCDM")
plt.plot(z_plot, fs8_wmap, "g--", label="WMAP7/LCDM")
plt.plot(z_plot, fs8_bf, "b-", label="LCDM Best Fit")
plt.xlabel("z")
plt.ylabel("f sigma8(z)")
plt.legend()
plt.show()
