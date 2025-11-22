import scipy
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import time
import brequet_range_equation_jax as brequet_range_equation

#X = [V, h]

constraint1 = {
    'type' : 'ineq', 
    'fun' : lambda x: -x[1]+2e4
    }

constraint2 = {
    'type' : 'ineq',
    'fun' : lambda x: -x[0]+540/3.6 #m/s
    }
constraint3 = {
    'type' : 'ineq',
    'fun' : lambda x: x[0] 
}
constraint4 = {
    'type' : 'ineq',
    'fun' : lambda x: x[1]
}


constraints=[constraint1, constraint2, constraint3, constraint4]
x0 = np.array([200.0, 1500.0], dtype=np.float64)

#grad range
def range_wrapper(x,fuel_percentage=0.75):
    S=100#m^2
    fuel_percentage = fuel_percentage
    return -brequet_range_equation.range(x[0],x[1],fuel_percentage,S)

jax_grad = jax.grad(range_wrapper)
def jac(x):
    # Ensure input is float64 before passing to JAX
    x_float = np.array(x, dtype=np.float64)
    return np.array(jax_grad(jnp.array(x_float, dtype=jnp.float64)), dtype=np.float64)

iterates = []
range_values = []
constraint_values = []
sqp_iteration_times = []

def callback(xk):
    # Track time when callback is called (this marks the end of an iteration)
    sqp_iteration_times.append(time.time())
    # Ensure we store as float64 array
    iterates.append(np.array(xk, dtype=np.float64))
    # Track range value (negative because we're minimizing -range)
    range_val = -range_wrapper(xk)
    range_values.append(range_val)
    # Track constraint values
    constraint_vals = []
    for constraint in constraints:
        constraint_vals.append(constraint['fun'](xk))
    constraint_values.append(constraint_vals)

# Add initial condition to iterates before optimization starts
iterates.append(np.array(x0))
range_values.append(-range_wrapper(x0))
initial_constraints = []
for constraint in constraints:
    initial_constraints.append(constraint['fun'](x0))
constraint_values.append(initial_constraints)

# Start timing before optimization
sqp_start_time = time.time()
sqp_iteration_times.append(sqp_start_time)  # Mark start time

result = scipy.optimize.minimize(range_wrapper, x0=x0, jac=jac, method = 'SLSQP', callback=callback, constraints = constraints)
sqp_total_time = time.time() - sqp_start_time

# Calculate time per iteration from the times between consecutive callbacks
sqp_time_per_iteration = []
for i in range(1, len(sqp_iteration_times)):
    iter_time = sqp_iteration_times[i] - sqp_iteration_times[i-1]
    sqp_time_per_iteration.append(iter_time)

print(result)
print("SQP Total optimization time: {:.4f} s".format(sqp_total_time))
print("SQP Number of iterations: {}".format(len(sqp_time_per_iteration)))
print("SQP Average time per iteration: {:.4f} s ({:.2f} ms)".format(
    np.mean(sqp_time_per_iteration) if len(sqp_time_per_iteration) > 0 else 0,
    np.mean(sqp_time_per_iteration)*1000 if len(sqp_time_per_iteration) > 0 else 0))

# Save SQP timing data
sqp_iterations = np.arange(len(range_values))
np.savez('plots/sqp_timing_data.npz', 
         iterations=sqp_iterations, 
         objective=range_values, 
         time_per_iter=sqp_time_per_iteration,
         total_time=sqp_total_time)
grad=[]
for x in iterates:
    grad.append(np.linalg.norm(jac(x)))

plt.figure()
plt.semilogy(grad)
plt.xlabel("Iteration")
plt.ylabel("Magnitude of Gradient")
plt.title("Range Equation Gradient Convergence")
plt.savefig("plots/range_equation_optimization.png")
plt.show()


def plot_brequet_contour_with_path(func, path_history, title="Range Optimization Path", x_range=(10, 300), y_range=(0, 25000), show_plot=False):
    """Plot contour of Brequet range with optimization path overlaid"""
    print("Generating contour plot (this may take a moment)...")
    
    # Create grid (using fewer points for speed)
    x = np.linspace(float(x_range[0]), float(x_range[1]), 100)
    y = np.linspace(float(y_range[0]), float(y_range[1]), 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Evaluate function on grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    # Convert to actual range (remove negative sign)
    Z_range = -Z
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot filled contours
    contourf = ax.contourf(X, Y, Z_range, levels=30, cmap='viridis', alpha=0.8)
    cbar = plt.colorbar(contourf, ax=ax, label='Range (m)')
    
    # Plot contour lines
    contour = ax.contour(X, Y, Z_range, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.0f')
    
    # Plot optimization path
    path_array = np.array(path_history)
    ax.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2.5, 
            label='Optimization Path', alpha=0.9, zorder=5)
    ax.plot(path_array[:, 0], path_array[:, 1], 'wo', markersize=3, 
            alpha=0.6, zorder=6)
    
    # Mark start and end points
    ax.plot(path_array[0, 0], path_array[0, 1], 'go', markersize=12, 
            label=f'Start: V={path_array[0, 0]:.1f} m/s, h={path_array[0, 1]:.0f} m',
            markeredgecolor='white', markeredgewidth=2, zorder=7)
    ax.plot(path_array[-1, 0], path_array[-1, 1], 'r*', markersize=18, 
            label=f'End: V={path_array[-1, 0]:.1f} m/s, h={path_array[-1, 1]:.0f} m',
            markeredgecolor='white', markeredgewidth=1.5, zorder=7)
    
    # Labels and formatting
    ax.set_xlabel('Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n({len(path_history)-1} iterations)', 
                  fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'plots/{title}.png')
    if show_plot == True:
        plt.show()

# Plot range and constraint convergence
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot range convergence
iterations = np.arange(len(range_values))
ax1.plot(iterations, range_values, 'b-o', linewidth=2, markersize=6, label='Range')
ax1.set_xlabel('Design Iteration', fontsize=12, fontweight='bold')
ax1.set_ylabel('Range (km)', fontsize=12, fontweight='bold')
ax1.set_title('Range Convergence', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot constraint convergence (only constraints 1 and 2)
constraint_array = np.array(constraint_values)
ax2.plot(iterations, constraint_array[:, 0], 'r-', linewidth=2, label='Constraint 1: h ≤ 20000 m', marker='o', markersize=6)
ax2.plot(iterations, constraint_array[:, 1], 'g-', linewidth=2, label='Constraint 2: v ≤ 150 m/s', marker='s', markersize=6)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Constraint Boundary')
ax2.set_xlabel('Design Iteration', fontsize=12, fontweight='bold')
ax2.set_ylabel('Constraint Value', fontsize=12, fontweight='bold')
ax2.set_title('Constraint Convergence (Inequality: g(x) ≥ 0)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig('plots/range_constraint_convergence.png', dpi=300, bbox_inches='tight')
print("Saved convergence plot to plots/range_constraint_convergence.png")
plt.close()

#plot_contour_with_path(range_wrapper,iterates,"Range Optimization Path", x_range=(0,300),y_range=(0,25000))
plot_brequet_contour_with_path(range_wrapper, iterates, title="Range Optimization Path", show_plot=True)