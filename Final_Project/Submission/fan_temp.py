#!/usr/bin/env python3
"""
Generate a graph of steady-state maximum temperature vs fan speed
for fixed heat generation parameters (a, b, c).

This script investigates potential cooling plateaus with increasing fan speed.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from heat_eq_2D_opt import HeatEquation2D

# Physical Dimensions
cpu_x = 0.04  # m
cpu_y = 0.04  # m
cpu_z = 0.04  # m
N = 25

# Temporal Parameters
CFL = 0.5
# Silicon Constants
k_si = 149
rho_si = 2323
c_si = 19.789 / 28.085 * 1000  # J/(kgK)

# Fixed heat generation parameters
# Using optimal values from optimization results
a_fixed = -60.0   # W/m⁴
b_fixed = -60.0   # W/m⁴
c_fixed = 153500.0  # W/m³ (approximately gives 10W total)

# Fan speed range
v_min = 10.0   # m/s
v_max = 40.0  # m/s
n_points = 15  # Number of fan speeds to test
fan_speeds = np.linspace(v_min, v_max, n_points)

# Initial condition function
def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Initial temperature distribution"""
    u = 70 * np.sin(x * np.pi / cpu_x) * np.sin(y * np.pi / cpu_y) + 293
    return u

# Heat generation function
def heat_generation_function(x: np.ndarray, y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x + b * y + c

# Setup heat equation solver
heq = HeatEquation2D(cpu_x, cpu_y, cpu_z, N, N,
                     k=k_si, rho=rho_si, cp=c_si,
                     init_condition=initial_condition)

# Set fixed heat generation
heq.set_heat_generation(heat_generation_function, a_fixed, b_fixed, c_fixed)

# Solver settings
heq.max_iter = 5e4
global_tolerance = 1e-3
heq.verbose = False

# Storage for results
max_temperatures = []
mean_temperatures = []
min_temperatures = []

print("="*60)
print("Computing steady-state temperatures for varying fan speeds")
print("="*60)
print(f"Fixed parameters: a={a_fixed}, b={b_fixed}, c={c_fixed:.0f}")
print(f"Fan speed range: {v_min} to {v_max} m/s ({n_points} points)")
print("-"*60)

# Loop over fan speeds
for i, v in enumerate(fan_speeds):
    # Update fan velocity
    heq.set_fan_velocity(v)
    
    # Reset to initial condition
    heq.reset()
    
    # Solve to steady state
    heq.solve_until_steady_state(tol=global_tolerance)
    
    # Extract temperature statistics
    max_T = np.max(heq.u)
    mean_T = np.mean(heq.u)
    min_T = np.min(heq.u)
    
    max_temperatures.append(max_T)
    mean_temperatures.append(mean_T)
    min_temperatures.append(min_T)
    
    # Progress indicator
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  v={v:5.2f} m/s: T_max={max_T-273:.2f}°C, T_mean={mean_T-273:.2f}°C")

print("="*60)
print("Computations complete. Generating plot...")

# Convert to Celsius and numpy arrays
max_temps_C = np.array(max_temperatures) - 273.15
mean_temps_C = np.array(mean_temperatures) - 273.15
min_temps_C = np.array(min_temperatures) - 273.15

# Create the plot
plt.figure(figsize=(10, 7))

# Plot maximum temperature (most important for cooling analysis)
plt.plot(fan_speeds, max_temps_C, 'r-', linewidth=2.5, label='Maximum Temperature', marker='o', markersize=4)

# Plot mean temperature for additional insight
plt.plot(fan_speeds, mean_temps_C, 'b--', linewidth=1.5, label='Mean Temperature', alpha=0.7)

# Optional: plot minimum temperature (less relevant for cooling)
# plt.plot(fan_speeds, min_temps_C, 'g--', linewidth=1.5, label='Minimum Temperature', alpha=0.5)

# Formatting
plt.xlabel('Fan Speed $v$ [m/s]', fontsize=13, fontweight='bold')
plt.ylabel('Temperature [°C]', fontsize=13, fontweight='bold')
# plt.title('Steady-State Temperature vs Fan Speed\n(Fixed Heat Generation Parameters)', 
#           fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11, loc='best')

# Add annotations for potential plateau region
# Find where the rate of temperature decrease slows down significantly
if len(max_temps_C) > 5:
    # Calculate rate of change (derivative approximation)
    dT_dv = np.diff(max_temps_C) / np.diff(fan_speeds)
    
    # Find where cooling rate drops below a threshold (indicating plateau)
    # Use a threshold of 10% of maximum cooling rate
    threshold = 0.1 * np.max(np.abs(dT_dv))
    plateau_region = np.where(np.abs(dT_dv) < threshold)[0]
    
    if len(plateau_region) > 0:
        v_plateau_start = fan_speeds[plateau_region[0]]
        v_plateau_end = fan_speeds[plateau_region[-1] + 1]
        T_at_plateau = max_temps_C[plateau_region[0]]
        
        plt.axvspan(v_plateau_start, v_plateau_end, alpha=0.2, color='yellow', 
                   label='Potential Plateau Region')
        plt.text(v_plateau_start, T_at_plateau + 2, 
                f'Plateau starts\n~{v_plateau_start:.1f} m/s',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Add vertical line at optimal velocity (if known)
optimal_v = 20.0  # From optimization results
if optimal_v >= v_min and optimal_v <= v_max:
    optimal_idx = np.argmin(np.abs(fan_speeds - optimal_v))
    optimal_T = max_temps_C[optimal_idx]
    plt.axvline(x=optimal_v, color='green', linestyle=':', linewidth=2, 
               label=f'Optimal v={optimal_v} m/s')
    plt.plot(optimal_v, optimal_T, 'go', markersize=10, zorder=5)

plt.tight_layout()

# Save the plot
output_file = 'fan_speed_vs_temperature.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved as '{output_file}'")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Temperature range: {max_temps_C[-1]:.2f}°C (v={fan_speeds[-1]:.1f} m/s) to {max_temps_C[0]:.2f}°C (v={fan_speeds[0]:.1f} m/s)")
print(f"Total temperature reduction: {max_temps_C[0] - max_temps_C[-1]:.2f}°C")
print(f"Temperature at optimal v={optimal_v} m/s: {optimal_T:.2f}°C")

# Calculate cooling efficiency (temperature drop per unit velocity increase)
initial_slope = (max_temps_C[0] - max_temps_C[5]) / (fan_speeds[5] - fan_speeds[0])
final_slope = (max_temps_C[-6] - max_temps_C[-1]) / (fan_speeds[-1] - fan_speeds[-6])
print(f"\nInitial cooling rate: {initial_slope:.3f}°C/(m/s)")
print(f"Final cooling rate: {final_slope:.3f}°C/(m/s)")
print(f"Cooling efficiency drop: {((initial_slope - final_slope) / initial_slope * 100):.1f}%")

# Show the plot
plt.show()
