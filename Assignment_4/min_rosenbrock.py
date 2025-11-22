import numpy as np
import scipy
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

jnp_grad = jax.grad(rosenbrock)

iterates = []

def callback(xk):
    iterates.append(xk)

def jac(x):
    return np.array(jnp_grad(jnp.array(x)), dtype=np.float64)

x0 = np.array([0.1,0.1])
constraint_1 = {
    'type': 'ineq',
    'fun': lambda x: 1-x[0]**2-x[1]**2
}

# Add initial condition to iterates before optimization starts
iterates.append(x0.copy())

opt_result = scipy.optimize.minimize(rosenbrock,x0=x0, jac = jac,method='SLSQP', callback=callback,constraints=constraint_1)
n_iterations = opt_result.nit

grad_list =[]
for i in iterates:
    grad_list.append(np.linalg.norm(jac(i)))
print(grad_list)
print("ITerates\n\n",iterates)
#plot grad vs iteration
plt.semilogy(grad_list)
plt.ylabel("Magnitue of Gradient of Rosenbrock")
plt.xlabel("Iteration")
plt.title("Gradient Convergence of Rosenbrock Optimization")
plt.show()



def plot_contour_with_path(func, path_history, title, x_range=None, y_range=None):
    """Plot contour of function with optimization path overlaid"""
    # Create grid
    if x_range is None:
        x_range = [np.min([pt[0] for pt in path_history]), np.max([pt[0] for pt in path_history])]
    if y_range is None:
        y_range = [np.min([pt[1] for pt in path_history]), np.max([pt[1] for pt in path_history])]
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Evaluate function on grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contours (log scale for Rosenbrock function)
    levels = np.logspace(-1, 5, 35)
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot optimization path
    path_array = np.array(path_history)
    ax.plot(path_array[:, 0], path_array[:, 1], 'ro-', linewidth=2, 
            markersize=4, label='Optimization Path', alpha=0.8)
    ax.plot(path_array[0, 0], path_array[0, 1], 'go', markersize=10, 
            label=f'Start: ({path_array[0, 0]:.2f}, {path_array[0, 1]:.2f})')
    ax.plot(path_array[-1, 0], path_array[-1, 1], 'r*', markersize=15, 
            label=f'End: ({path_array[-1, 0]:.3f}, {path_array[-1, 1]:.3f})')
    
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'{title}\nIterations: {len(path_history)-1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
plot_contour_with_path(rosenbrock,iterates,"Rosenbrock Optimization Path",x_range=(-10,10),y_range=(-15,10))