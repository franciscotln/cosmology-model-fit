import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


# values from BAO + CCH + DES5Y
G = 1  # 6.6743e-11 # m^3/kg/s^2 normalised
c = 1  # 299792458 # m/s normalised
Rho_de_0 = 1  # normalised
H0 = 67.1  # Hubble constant in km/s/Mpc
Om = 0.308
Or = 4.1835e-05 / (H0 / 100) ** 2  # Radiation density
w0 = -0.833  # Equation of state parameter from fit

a_min = 1e-8
a_max = 4
N_a = 5000
a_vals = np.linspace(a_min, a_max, N_a)


w_de = lambda a: -1 + 2 * (1 + w0) * a**3 / (1 + a**3)

Rho_de = lambda a: Rho_de_0 * (2 / (1 + a**3)) ** (2 * (1 + w0))

H = lambda a: np.sqrt(Om * a**-3 + Or * a**-4 + (1 - Om - Or) * Rho_de(a) / Rho_de_0)

V_phi = lambda a: (1 - w_de(a)) * Rho_de(a) / 2

d_phi_da = lambda a: np.sqrt(Rho_de(a) * (1 + w_de(a))) / (a * H(a))

phi_vals = cumulative_trapezoid(d_phi_da(a_vals), a_vals, initial=0)

a_of_phi = interp1d(phi_vals, a_vals, bounds_error=False, fill_value="extrapolate")

V_of_phi = lambda phi: V_phi(a_of_phi(phi))

phi_plot = np.linspace(min(phi_vals), max(phi_vals), 2000)

d_phi_dt_val = d_phi_da(a_vals) * H(a_vals) * a_vals

# Scalar field
plt.figure(figsize=(8, 5))
plt.plot(a_vals, phi_vals, label=r"$\phi(a)$")
plt.axvline(x=1, color="r", linestyle="--", label="Present time")
plt.xlabel(r"$a$")
plt.ylabel(r"$\phi(a)$")
plt.xlim(0, None)
plt.ylim(0, max(phi_vals))
plt.title(r"Scalar Field $\phi(a)$")
plt.legend()
plt.grid(True)
plt.show()

# Potential
plt.figure(figsize=(8, 5))
plt.plot(phi_plot, V_of_phi(phi_plot), label=r"$V(\phi)$")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$V(\phi)$")
plt.xlim(0, None)
plt.title(r"Scalar Field Potential $V(\phi)$")
plt.legend()
plt.grid(True)
plt.show()

# Plot V(a)
plt.figure(figsize=(8, 5))
plt.plot(a_vals, V_phi(a_vals), label=r"$V(a)$")
plt.xlabel(r"$a$")
plt.ylabel(r"$V(a)$")
plt.title(r"Scalar Field Potential $V(a)$")
plt.legend()
plt.grid(True)
plt.show()


dt_da = lambda a: 1 / (a * H(a))
t_vals = cumulative_trapezoid(dt_da(a_vals), a_vals, initial=0)

Hubble_time_Gyr = 9.77813 / (H0 / 100)  # Hubble time in Gyr
t_vals_Gyr = t_vals * Hubble_time_Gyr
a_of_t = interp1d(t_vals_Gyr, a_vals, bounds_error=False, fill_value="extrapolate")
phi_of_t = lambda t: np.interp(t, t_vals_Gyr, phi_vals)
t_grid_Gyr = np.linspace(min(t_vals_Gyr), max(t_vals_Gyr), 1000)
t_today_Gyr = np.interp(1, a_vals, t_vals_Gyr)

# Convert time from 1/H0 units to Gyr
# H0 in km/s/Mpc, 1/H0 in seconds: 1 Mpc = 3.085677581e19 km
t_vals_Gyr = t_vals * 9.77813 / (H0 / 100)
t_grid_Gyr = np.linspace(min(t_vals_Gyr), max(t_vals_Gyr), 1000)
t_today_Gyr = np.interp(1, a_vals, t_vals_Gyr)

t_at_a1 = np.interp(1, a_vals, t_vals)
print(f"Hubble time (Gyr): {Hubble_time_Gyr}")
print(f"t_today (Gyr): {t_today_Gyr}")


# Kinetic term (in Gyr)
plt.figure(figsize=(8, 5))
plt.plot(t_vals_Gyr, 0.5 * d_phi_dt_val**2, label=r"Kinetic term")
plt.xlabel("t [Gyr]")
plt.ylabel(r"0.5 * $\left(\frac{d\phi}{dt}\right)^2$")
plt.title(r"Scalar Field Kinetic Term $\left(\frac{d\phi}{dt}\right)^2$")
plt.axvline(
    x=t_vals_Gyr[np.argmax(0.5 * d_phi_dt_val**2)],
    color="g",
    linestyle="--",
    label="Max speed",
    alpha=0.5,
)
plt.axvline(
    x=t_today_Gyr,
    color="r",
    linestyle="--",
    label="Present time",
    alpha=0.5,
)
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(t_grid_Gyr, phi_of_t(t_grid_Gyr), label=r"$\phi(t)$")
plt.xlabel(r"$t$ [Gyr]")
plt.ylabel(r"$\phi(t)$")
plt.title(r"Scalar Field $\phi$ vs Time $t$ [Gyr]")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(t_grid_Gyr, a_of_t(t_grid_Gyr), label=r"$a(t)$")
plt.xlabel(r"$t$ [Gyr]")
plt.ylabel(r"$a(t)$")
plt.title(r"Scale Factor $a$ vs Time $t$ [Gyr]")
plt.axvline(x=t_today_Gyr, color="r", linestyle="--", label="Present time", alpha=0.5)
plt.legend()
plt.grid(True)
plt.show()
