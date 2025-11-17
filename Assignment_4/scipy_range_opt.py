import scipy
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import brequet_range_equation

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

def callback(xk):
    # Ensure we store as float64 array
    iterates.append(np.array(xk, dtype=np.float64))

# Add initial condition to iterates before optimization starts
iterates.append(np.array(x0))

result = scipy.optimize.minimize(range_wrapper, x0=x0, jac=jac, method = 'SLSQP', callback=callback, constraints = constraints)
print(result)
grad=[]
for x in iterates:
    grad.append(np.linalg.norm(jac(x)))

plt.figure()
plt.semilogy(grad)
plt.xlabel("Iteration")
plt.ylabel("Magnitude of Gradient")
plt.title("Range Equation Gradient Convergence")
plt.savefig("range_equation_optimization.png")
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
    plt.savefig(f'{title}.png')
    if show_plot == True:
        plt.show()

#plot_contour_with_path(range_wrapper,iterates,"Range Optimization Path", x_range=(0,300),y_range=(0,25000))
plot_brequet_contour_with_path(range_wrapper, iterates, title="Range Optimization Path", show_plot=True)