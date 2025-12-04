import numpy as np
import scipy.optimize as opt
from scipy.optimize import Bounds, NonlinearConstraint

#CONSTANTS
gtol = 1e-4
OPTIMIZATION_METHOD = 'SLSQP'
ONE_HILL = False
MAX_X = 10
MAX_Y = 10
MIN_X = 0
MIN_Y = 0
MAX_VEL = 1.5
T = 15
N = 45 ## waypoints
dt = T/N #time step seconds
print("Optimization method: ", OPTIMIZATION_METHOD)
print("One hill: ", ONE_HILL)
#x_array = [x1, y1, x2, y2, ..., xN, yN]
#length of x_array is 2*N
#last index of x_array is 2*N-2

#hill_cost = lambda x: 1/((x[0]-5)**2 + (x[1]-5)**2 + 1)
if ONE_HILL:
    hill_cost = lambda x: 1/((x[0]-5)**2 + (x[1]-5)**2 + 1)
else:
    hill_cost = lambda x: np.cos(x[0])**2 * np.cos(x[1])**2


x_path = []
lagrangian_grad_norms = []
grad_f_norms = []
constraint_values_history = []  # Store constraint values at each iteration

# Define named constraint functions for tracking
def eq_initial_vel(xy_array):
    return (xy_array[2]-xy_array[0])**2 + (xy_array[3]-xy_array[1])**2

def eq_initial_x(xy_array):
    return xy_array[0]

def eq_initial_y(xy_array):
    return xy_array[1]

def eq_final_x(xy_array):
    return xy_array[2*N-2] - MAX_X

def eq_final_y(xy_array):
    return xy_array[2*N-1] - MAX_Y

# Select a few representative velocity constraints to track at different points
def ineq_vel_quarter(xy_array):
    """Velocity constraint at 1/4 of path"""
    idx = N // 4
    vx = (xy_array[2*idx] - xy_array[2*idx-2]) / dt
    vy = (xy_array[2*idx+1] - xy_array[2*idx-1]) / dt
    return MAX_VEL**2 - (vx**2 + vy**2)

def ineq_vel_mid(xy_array):
    """Velocity constraint at midpoint (1/2 of path)"""
    idx = N // 2
    vx = (xy_array[2*idx] - xy_array[2*idx-2]) / dt
    vy = (xy_array[2*idx+1] - xy_array[2*idx-1]) / dt
    return MAX_VEL**2 - (vx**2 + vy**2)

def ineq_vel_three_quarter(xy_array):
    """Velocity constraint at 3/4 of path"""
    idx = 3 * N // 4
    vx = (xy_array[2*idx] - xy_array[2*idx-2]) / dt
    vy = (xy_array[2*idx+1] - xy_array[2*idx-1]) / dt
    return MAX_VEL**2 - (vx**2 + vy**2)

# Constraints to track (name, function, type)
tracked_constraints = [
    ('Initial vel = 0', eq_initial_vel, 'eq'),
    ('Initial x = 0', eq_initial_x, 'eq'),
    ('Initial y = 0', eq_initial_y, 'eq'),
    ('Final x = MAX_X', eq_final_x, 'eq'),
    ('Final y = MAX_Y', eq_final_y, 'eq'),
    ('vel @ 1/4 path', ineq_vel_quarter, 'ineq'),
    ('vel @ 1/2 path', ineq_vel_mid, 'ineq'),
    ('vel @ 3/4 path', ineq_vel_three_quarter, 'ineq'),
]

def evaluate_tracked_constraints(xk):
    """Evaluate all tracked constraints at current point"""
    return {name: func(xk) for name, func, _ in tracked_constraints}

if OPTIMIZATION_METHOD == 'trust-constr':
    def callback(intermediate_result):
        x_path.append(intermediate_result.x.copy())
        lagrangian_grad_norms.append(np.linalg.norm(intermediate_result.lagrangian_grad))
        grad_f_norms.append(np.linalg.norm(intermediate_result.grad))
        constraint_values_history.append(evaluate_tracked_constraints(intermediate_result.x))

elif OPTIMIZATION_METHOD == 'SLSQP':
    def callback(xk):
        x_path.append(xk.copy())
        constraint_values_history.append(evaluate_tracked_constraints(xk))

def cost(xy_array):
    curr_cost=0
    for i in range(N):
        curr_cost += hill_cost((xy_array[2*i], xy_array[2*i+1]))
        if i != 0:
            curr_cost += (xy_array[2*i] - xy_array[2*i-2])**2 + (xy_array[2*i+1] - xy_array[2*i-1])**2
    return curr_cost



# Use Bounds for box constraints (much more efficient for trust-constr)
lower_bounds = np.zeros(2*N)
upper_bounds = np.zeros(2*N)
for i in range(N):
    lower_bounds[2*i] = MIN_X      # x lower bound
    lower_bounds[2*i+1] = MIN_Y    # y lower bound
    upper_bounds[2*i] = MAX_X      # x upper bound
    upper_bounds[2*i+1] = MAX_Y    # y upper bound
bounds = Bounds(lower_bounds, upper_bounds)

# Velocity constraint function (all segments at once)
def velocity_constraint(xy_array):
    """Returns velocity^2 for each segment (should be <= MAX_VEL^2)"""
    vel_sq = np.zeros(N-1)
    for i in range(1, N):
        vx = (xy_array[2*i] - xy_array[2*i-2]) / dt
        vy = (xy_array[2*i+1] - xy_array[2*i-1]) / dt
        vel_sq[i-1] = vx**2 + vy**2
    return vel_sq

# Equality constraints function (all at once)
def equality_constraints(xy_array):
    """Returns array of equality constraint values (should all be 0)"""
    return np.array([
        (xy_array[2]-xy_array[0])**2 + (xy_array[3]-xy_array[1])**2,  # initial vel = 0
        xy_array[0],                    # initial x = 0
        xy_array[1],                    # initial y = 0
        xy_array[2*N-2] - MAX_X,        # final x = MAX_X
        xy_array[2*N-1] - MAX_Y         # final y = MAX_Y
    ])

# Set up constraints based on method
if OPTIMIZATION_METHOD == 'trust-constr':
    # Use NonlinearConstraint objects (more efficient)
    constraints = [
        NonlinearConstraint(velocity_constraint, -np.inf, MAX_VEL**2),  # vel^2 <= MAX_VEL^2
        NonlinearConstraint(equality_constraints, 0, 0)  # equality constraints = 0
    ]
else:
    # SLSQP uses dict format
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: xy_array[2*i] - MIN_X})
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: xy_array[2*i+1] - MIN_Y})
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: MAX_X - xy_array[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: MAX_Y - xy_array[2*i+1]})
        if i != 0:
            constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: MAX_VEL**2 - (((xy_array[2*i] - xy_array[2*i-2])/dt)**2 + ((xy_array[2*i+1] - xy_array[2*i-1])/dt)**2) })
    constraints.append({'type': 'eq', 'fun': lambda xy_array: (xy_array[2]-xy_array[0])**2 + (xy_array[3]-xy_array[1])**2})
    constraints.append({'type': 'eq', 'fun': lambda xy_array: (xy_array[0])})
    constraints.append({'type': 'eq', 'fun': lambda xy_array: (xy_array[1])})
    constraints.append({'type': 'eq', 'fun': lambda xy_array: (xy_array[2*N-2]-MAX_X)})
    constraints.append({'type': 'eq', 'fun': lambda xy_array: (xy_array[2*N-1]-MAX_Y)})
    bounds = None  # SLSQP will use the inequality constraints for bounds


x_points = np.linspace(MIN_X, MAX_X, N)
y_points = np.linspace(MIN_Y, MAX_Y, N)
initial_path = np.empty(2 * N)
initial_path[0::2] = x_points
initial_path[1::2] = y_points
x0 = initial_path

# Print initial constraint values to understand starting point
print("\nINITIAL CONSTRAINT VALUES (before optimization):")
print("-" * 50)
for name, func, ctype in tracked_constraints:
    val = func(x0)
    print(f"  {name:<20}: {val:.6e}")
print("-" * 50)

# Optimization options
if OPTIMIZATION_METHOD == 'trust-constr':
    options = {
        'maxiter': 5000,      # Increase max iterations
        'gtol': 1e-4,         # Gradient tolerance
        'xtol': 1e-8,         # Variable tolerance
        'verbose': 1          # Show progress
    }
    result = opt.minimize(cost, x0, tol=gtol, method=OPTIMIZATION_METHOD, jac='2-point',
                          bounds=bounds, constraints=constraints, 
                          callback=callback, options=options)
else:
    result = opt.minimize(cost, x0, tol=gtol, method=OPTIMIZATION_METHOD, 
                          constraints=constraints, callback=callback)

print("Final Cost: ",cost(result.x))

import matplotlib.pyplot as plt

# Recreate meshgrid for hill cost visualization
x_grid = np.linspace(MIN_X, MAX_X, 100)
y_grid = np.linspace(MIN_Y, MAX_Y, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = hill_cost((X, Y))

plt.figure(figsize=(8, 6))
# Plot the hill cost as a contour map
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
plt.colorbar(contour, label='Hill Cost')

# Plot the initial guess path
x0_path = x0[0::2]
y0_path = x0[1::2]
plt.plot(x0_path, y0_path, 'b--o', label='Initial Guess')

# Plot the optimized path
xy = result.x
x_opt_path = xy[0::2]
y_opt_path = xy[1::2]
plt.plot(x_opt_path, y_opt_path, 'ro-', label='Optimized Path')

plt.xlim(MIN_X, MAX_X)
plt.ylim(MIN_Y, MAX_Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Final Path and Hill Cost')
plt.legend()
plt.grid(True)
if ONE_HILL:
    plt.savefig('plots/1_hill_path_opt.png')
else:
    plt.savefig('plots/many_hills_path_opt.png')
plt.show()


# Plot constraint values vs iterations
if len(constraint_values_history) > 0:
    iterations = list(range(1, len(constraint_values_history) + 1))
    
    # Extract constraint values into arrays
    constraint_names = list(constraint_values_history[0].keys())
    
    # Separate equality and inequality constraints for plotting
    eq_constraints = [(name, func, t) for name, func, t in tracked_constraints if t == 'eq']
    ineq_constraints = [(name, func, t) for name, func, t in tracked_constraints if t == 'ineq']
    
    # Plot equality constraints (should converge to 0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Equality constraints subplot (linear scale to see values at 0)
    ax1 = axes[0]
    for name, _, _ in eq_constraints:
        values = [cv[name] for cv in constraint_values_history]
        ax1.plot(iterations, np.abs(values), '-o', label=name, markersize=3)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Target (0)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('|Constraint Value|')
    ax1.set_title('Equality Constraints (should → 0)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)  # Extend axis to 0
    
    # Inequality constraints subplot
    ax2 = axes[1]
    for name, _, _ in ineq_constraints:
        values = [cv[name] for cv in constraint_values_history]
        ax2.plot(iterations, values, '-o', label=name, markersize=3)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Constraint boundary')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Constraint Value')
    ax2.set_title('Inequality Constraints (should be ≥ 0)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if ONE_HILL:
        plt.savefig('plots/1_hill_constraint_convergence.png', dpi=150)
    else:
        plt.savefig('plots/many_hills_constraint_convergence.png', dpi=150)
    plt.show()

# Plot Lagrangian gradient convergence (trust-constr only)
if OPTIMIZATION_METHOD == 'trust-constr' and len(lagrangian_grad_norms) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations_lag = list(range(1, len(lagrangian_grad_norms) + 1))
    
    # Lagrangian gradient norm (log scale)
    ax1 = axes[0]
    ax1.semilogy(iterations_lag, lagrangian_grad_norms, 'b-o', markersize=3, label='||∇L||')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('||∇L|| (log scale)')
    ax1.set_title('Convergence of Gradient of Lagrangian')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()
    
    # Objective gradient norm
    ax2 = axes[1]
    ax2.semilogy(iterations_lag, grad_f_norms, 'r-o', markersize=3, label='||∇f||')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('||∇f|| (log scale)')
    ax2.set_title('Convergence of Objective Gradient')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    if ONE_HILL:
        plt.savefig('plots/1_hill_lagrangian_gradient_convergence.png', dpi=150)
    else:
        plt.savefig('plots/many_hills_lagrangian_gradient_convergence.png', dpi=150)
    plt.show()
    
    print(f"\nFinal ||∇L||: {lagrangian_grad_norms[-1]:.6e}")
    print(f"Final ||∇f||: {grad_f_norms[-1]:.6e}")

# Print table of final constraint values
print("\n" + "="*60)
print("FINAL CONSTRAINT VALUES TABLE")
print("="*60)
print(f"{'Constraint':<25} {'Type':<8} {'Value':<15} {'Status'}")
print("-"*60)

final_x = result.x
for name, func, ctype in tracked_constraints:
    value = func(final_x)
    if ctype == 'eq':
        status = '✓ Satisfied' if abs(value) < 1e-6 else '✗ Violated'
    else:  # ineq
        status = '✓ Satisfied' if value >= -1e-6 else '✗ Violated'
    print(f"{name:<25} {ctype:<8} {value:<15.6e} {status}")

print("="*60)
print(f"\nOptimization terminated with status: {result.message}")
print(f"Number of iterations: {result.nit}")
