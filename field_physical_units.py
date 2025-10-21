import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, c, hbar
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# values from BAO + CCH + DES5Y
Rho_de_0 = 1  # normalised
H0 = 67.1  # Hubble constant in km/s/Mpc
Om = 0.308
Or = 4.1835e-05 / (H0 / 100) ** 2  # Radiation density
w0 = -0.833  # Equation of state parameter from fit

a_min = 1e-8
a_max = 4
a_vals = np.linspace(a_min, a_max, 5000)

# ============================================================================
# PHYSICAL UNIT CONVERSIONS
# ============================================================================

# Convert H0 to SI units (1/s)
km_to_m = 1e3
Mpc_to_m = 3.085677581e22  # meters
H0_SI = H0 * km_to_m / Mpc_to_m  # in 1/s
print(f"H0 in SI units: {H0_SI:.3e} s^-1")

# Critical density in SI units (kg/m^3)
rho_crit_SI = 3 * H0_SI**2 / (8 * np.pi * G)
print(f"Critical density: {rho_crit_SI:.3e} kg/m^3")

# Critical density in energy units (J/m^3)
rho_crit_energy_SI = rho_crit_SI * c**2
print(f"Critical energy density: {rho_crit_energy_SI:.3e} J/m^3")

# Reduced Planck mass in SI units (kg)
M_pl_SI = np.sqrt(hbar * c / G)  # kg
print(f"Reduced Planck mass: {M_pl_SI:.3e} kg")

# Reduced Planck mass in GeV/c^2
M_pl_GeV = M_pl_SI * c**2 / (1.602176634e-10)  # Convert J to GeV
print(f"Reduced Planck mass: {M_pl_GeV:.3e} GeV/c^2")

# Energy scale for the potential: (3 H0^2 M_pl^2 c^2) / (8π) in J/m^3
# This is the natural scale for scalar field potentials
V_scale_SI = (3 * H0_SI**2 * M_pl_SI**2 * c**2) / (8 * np.pi)
print(f"Potential energy scale: {V_scale_SI:.3e} J/m^3")

# For scalar field in SI: sqrt(J/m^3) * m^(3/2) = kg^(1/2) * m^(1/2)
phi_scale_SI = M_pl_SI  # In natural units, phi is in units of M_pl
print(f"Scalar field scale: {phi_scale_SI:.3e} kg (or {M_pl_GeV:.3e} GeV/c^2)")

print("\n" + "=" * 70 + "\n")

# ============================================================================
# COSMOLOGICAL EQUATIONS (in natural/reduced units)
# ============================================================================

w_de = lambda a: -1 + 2 * (1 + w0) * a**3 / (1 + a**3)

Rho_de = lambda a: Rho_de_0 * (2 / (1 + a**3)) ** (2 * (1 + w0))

H = (
    lambda a: H0
    * (Om * a**-3 + Or * a**-4 + (1 - Om - Or) * Rho_de(a) / Rho_de_0) ** 0.5
)

V_phi = lambda a: (1 - w_de(a)) * Rho_de(a) / 2

# Dimensionless Hubble parameter
h = lambda a: H(a) / H0

# d_phi/da in reduced Planck units (dimensionless)
d_phi_da = lambda a: np.sqrt(Rho_de(a) * (1 + w_de(a))) / (a * h(a))

phi_vals = cumulative_trapezoid(d_phi_da(a_vals), a_vals, initial=0)

a_of_phi = interp1d(phi_vals, a_vals, bounds_error=False, fill_value="extrapolate")

V_of_phi = lambda phi: V_phi(a_of_phi(phi))

phi_plot = np.linspace(min(phi_vals), max(phi_vals), 2000)

# d_phi/dt in units of H0 (to be converted to physical units later)
d_phi_dt_val = d_phi_da(a_vals) * h(a_vals) * a_vals

# dt/da in units of 1/H0 (dimensionless time)
dt_da = lambda a: 1 / (a * h(a))
t_vals = cumulative_trapezoid(dt_da(a_vals), a_vals, initial=0)

# Convert time from 1/H0 units to Gyr
Hubble_time_Gyr = 9.77813 / (H0 / 100)  # Hubble time in Gyr
t_vals_Gyr = t_vals * Hubble_time_Gyr
t_today_Gyr = np.interp(1, a_vals, t_vals_Gyr)

phi_today = np.interp(1, a_vals, phi_vals)

# ============================================================================
# CONVERT TO PHYSICAL UNITS
# ============================================================================

# Scalar field in physical units
phi_vals_GeV = phi_vals * M_pl_GeV  # in GeV/c^2
phi_today_GeV = phi_today * M_pl_GeV

# Potential in physical units (J/m^3)
V_vals_SI = V_phi(a_vals) * rho_crit_energy_SI  # in J/m^3
V_vals_GeV4 = V_vals_SI / (1.602176634e-10) ** 4 * (hbar * c) ** 3 / c**3  # in GeV^4

# Time derivative in physical units
d_phi_dt_vals_SI = d_phi_dt_val * H0_SI * phi_scale_SI  # in kg/s (or GeV/(c^2 * s))
kinetic_term_SI = 0.5 * d_phi_dt_vals_SI**2  # in kg^2/s^2

# Kinetic term as energy density (J/m^3)
# The kinetic term is (1/2)(dφ/dt)^2 which needs to be divided by appropriate volume
# In field theory: kinetic energy density = (1/2)(∂φ/∂t)^2
# Our kinetic_term is already in the right form but needs energy density units
kinetic_energy_density_SI = 0.5 * d_phi_dt_val**2 * rho_crit_energy_SI

# ============================================================================
# PLOTTING WITH PHYSICAL UNITS
# ============================================================================

# Scalar field in GeV
plt.figure(figsize=(8, 5))
plt.plot(a_vals, phi_vals_GeV, label=r"$\phi(a)$")
plt.axvline(x=1, color="r", linestyle="--", label="Present time")
plt.xlabel(r"$a$")
plt.ylabel(r"$\phi(a)$ [GeV/c$^2$]")
plt.xlim(0, None)
plt.ylim(0, max(phi_vals_GeV))
plt.title(r"Scalar Field $\phi(a)$ in Physical Units")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Potential in J/m^3
plt.figure(figsize=(8, 5))
plt.plot(a_vals, V_vals_SI, label=r"$V(a)$")
plt.axvline(x=1, color="r", linestyle="--", label="Present time")
plt.xlabel(r"$a$")
plt.ylabel(r"$V(a)$ [J/m$^3$]")
plt.title(r"Scalar Field Potential $V(a)$ in Physical Units")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Potential vs phi in physical units
phi_plot_GeV = phi_plot * M_pl_GeV
V_plot_SI = V_of_phi(phi_plot) * rho_crit_energy_SI

plt.figure(figsize=(8, 5))
plt.plot(phi_plot_GeV, V_plot_SI, label=r"$V(\phi)$")
plt.axvline(
    x=phi_today_GeV,
    color="r",
    linestyle="--",
    label=f"Present ($\\phi_0$ = {phi_today_GeV:.2e} GeV/c$^2$)",
)
plt.xlabel(r"$\phi$ [GeV/c$^2$]")
plt.ylabel(r"$V(\phi)$ [J/m$^3$]")
plt.title(r"Scalar Field Potential in Physical Units")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Kinetic energy density vs time
plt.figure(figsize=(8, 5))
plt.plot(t_vals_Gyr, kinetic_energy_density_SI, label=r"Kinetic energy density")
plt.xlabel("t [Gyr]")
plt.ylabel(r"$\frac{1}{2}\left(\frac{d\phi}{dt}\right)^2$ [J/m$^3$]")
plt.title(r"Scalar Field Kinetic Energy Density")
plt.yscale("log")
plt.axvline(
    x=t_today_Gyr,
    color="r",
    linestyle="--",
    label="Present time",
    alpha=0.5,
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================================
# COMPARISON PLOT: Natural vs Physical Units
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Natural units - φ(a)
axes[0, 0].plot(a_vals, phi_vals, "b-", linewidth=2)
axes[0, 0].axvline(x=1, color="r", linestyle="--", alpha=0.5)
axes[0, 0].set_xlabel(r"$a$")
axes[0, 0].set_ylabel(r"$\phi(a)$ [reduced Planck units]")
axes[0, 0].set_title(r"Natural Units: $\phi(a)$")
axes[0, 0].grid(True)
axes[0, 0].set_xlim(0, None)

# Physical units - φ(a)
axes[0, 1].plot(a_vals, phi_vals_GeV, "b-", linewidth=2)
axes[0, 1].axvline(x=1, color="r", linestyle="--", alpha=0.5)
axes[0, 1].set_xlabel(r"$a$")
axes[0, 1].set_ylabel(r"$\phi(a)$ [GeV/c$^2$]")
axes[0, 1].set_title(r"Physical Units: $\phi(a)$")
axes[0, 1].grid(True)
axes[0, 1].set_xlim(0, None)

# Natural units - V(φ)
axes[1, 0].plot(phi_plot, V_of_phi(phi_plot), "g-", linewidth=2)
axes[1, 0].axvline(x=phi_today, color="r", linestyle="--", alpha=0.5)
axes[1, 0].set_xlabel(r"$\phi$ [reduced Planck units]")
axes[1, 0].set_ylabel(r"$V(\phi)$ [normalized]")
axes[1, 0].set_title(r"Natural Units: $V(\phi)$")
axes[1, 0].grid(True)

# Physical units - V(φ)
axes[1, 1].plot(phi_plot_GeV, V_plot_SI, "g-", linewidth=2)
axes[1, 1].axvline(x=phi_today_GeV, color="r", linestyle="--", alpha=0.5)
axes[1, 1].set_xlabel(r"$\phi$ [GeV/c$^2$]")
axes[1, 1].set_ylabel(r"$V(\phi)$ [J/m$^3$]")
axes[1, 1].set_title(r"Physical Units: $V(\phi)$")
axes[1, 1].set_yscale("log")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# SUMMARY PRINTOUT
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY OF KEY VALUES")
print("=" * 70)
print(f"\nToday (a=1):")
print(f"  φ₀ (natural units):  {phi_today:.6f}")
print(f"  φ₀ (physical):       {phi_today_GeV:.3e} GeV/c²")
print(f"  V(φ₀) (normalized):  {V_phi(1):.6f}")
print(f"  V(φ₀) (physical):    {V_phi(1) * rho_crit_energy_SI:.3e} J/m³")
print(f"  w_DE(today):         {w_de(1):.6f}")
print(f"  Age of universe:     {t_today_Gyr:.3f} Gyr")

print(f"\nUnit conversion factors:")
print(f"  φ [natural] × {M_pl_GeV:.3e} = φ [GeV/c²]")
print(f"  V [natural] × {rho_crit_energy_SI:.3e} = V [J/m³]")
print(f"  t [1/H₀] × {Hubble_time_Gyr:.3f} = t [Gyr]")
print("=" * 70)
