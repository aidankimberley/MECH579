import numpy as np
import scipy.optimize as opt
from scipy.optimize import Bounds, NonlinearConstraint

#CONSTANTS
gtol = 1e-4
OPTIMIZATION_METHOD = 'trust-constr'
distance_tolerance = 0.1
ONE_HILL = False
MAX_X = 10
MAX_Y = 10
MIN_X = 0
MIN_Y = 0
MAX_VEL = 1.5
T = 15
N = 45 ## waypoints
dt = T/N #time step seconds

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

# Define named constraint functions for tracking (for 2 drones)
# Drone 1 constraints
def eq_d1_initial_vel(xy_array):
    return (xy_array[2]-xy_array[0])**2 + (xy_array[3]-xy_array[1])**2

def eq_d1_initial_pos(xy_array):
    return xy_array[0]**2 + xy_array[1]**2  # should be 0 (starts at origin)

def eq_d1_final_pos(xy_array):
    return (xy_array[2*N-2] - MAX_X)**2 + (xy_array[2*N-1] - MAX_Y)**2  # should be 0

# Drone 2 constraints
def eq_d2_initial_vel(xy_array):
    return (xy_array[2*N+2]-xy_array[2*N])**2 + (xy_array[2*N+3]-xy_array[2*N+1])**2

def eq_d2_initial_pos(xy_array):
    return (xy_array[2*N] - MAX_X)**2 + (xy_array[2*N+1] - MAX_Y)**2  # should be 0 (starts at 10,10)

def eq_d2_final_pos(xy_array):
    return xy_array[4*N-2]**2 + xy_array[4*N-1]**2  # should be 0 (ends at origin)

# Collision constraint at midpoint
def ineq_collision_mid(xy_array):
    """Distance^2 between drones at midpoint (should be >= tolerance^2)"""
    idx = N // 2
    dx = xy_array[2*idx] - xy_array[2*N + 2*idx]
    dy = xy_array[2*idx + 1] - xy_array[2*N + 2*idx + 1]
    return dx**2 + dy**2 - distance_tolerance**2

# Velocity constraints
def ineq_d1_vel_mid(xy_array):
    """Drone 1 velocity at midpoint"""
    idx = N // 2
    vx = (xy_array[2*idx] - xy_array[2*idx-2]) / dt
    vy = (xy_array[2*idx+1] - xy_array[2*idx-1]) / dt
    return MAX_VEL**2 - (vx**2 + vy**2)

def ineq_d2_vel_mid(xy_array):
    """Drone 2 velocity at midpoint"""
    idx = N // 2
    vx = (xy_array[2*N + 2*idx] - xy_array[2*N + 2*idx - 2]) / dt
    vy = (xy_array[2*N + 2*idx + 1] - xy_array[2*N + 2*idx - 1]) / dt
    return MAX_VEL**2 - (vx**2 + vy**2)

# Constraints to track (name, function, type)
tracked_constraints = [
    ('D1 Initial vel=0', eq_d1_initial_vel, 'eq'),
    ('D1 Start @ (0,0)', eq_d1_initial_pos, 'eq'),
    ('D1 End @ (10,10)', eq_d1_final_pos, 'eq'),
    ('D2 Initial vel=0', eq_d2_initial_vel, 'eq'),
    ('D2 Start @ (10,10)', eq_d2_initial_pos, 'eq'),
    ('D2 End @ (0,0)', eq_d2_final_pos, 'eq'),
    ('Collision @ mid', ineq_collision_mid, 'ineq'),
    ('D1 vel @ mid', ineq_d1_vel_mid, 'ineq'),
    ('D2 vel @ mid', ineq_d2_vel_mid, 'ineq'),
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
        if (i != 0):
            curr_cost += (xy_array[2*i] - xy_array[2*i-2])**2 + (xy_array[2*i+1] - xy_array[2*i-1])**2

    return curr_cost

def cost_both_drones(xy_array):
    """Total cost for both drones"""
    drone1_path = xy_array[0:2*N]
    drone2_path = xy_array[2*N:4*N]
    return cost(drone1_path) + cost(drone2_path)

# Use Bounds for box constraints (much more efficient for trust-constr)
# For 2 drones: array is [drone1_x1, drone1_y1, ..., drone1_xN, drone1_yN, drone2_x1, drone2_y1, ..., drone2_xN, drone2_yN]
# Total length: 4*N
lower_bounds = np.full(4*N, MIN_X)  # All coordinates have same bounds
upper_bounds = np.full(4*N, MAX_X)
for i in range(2*N):  # For both drones
    lower_bounds[2*i] = MIN_X      # x lower bound
    lower_bounds[2*i+1] = MIN_Y    # y lower bound
    upper_bounds[2*i] = MAX_X      # x upper bound
    upper_bounds[2*i+1] = MAX_Y    # y upper bound
bounds = Bounds(lower_bounds, upper_bounds)

# Collision avoidance constraint: distance^2 >= distance_tolerance^2 at each time step
def collision_constraint(xy_array):
    """Returns distance^2 between drones at each time step (should be >= distance_tolerance^2)"""
    dist_sq = np.zeros(N)
    for i in range(N):
        # Drone 1 position at time i: xy_array[2*i], xy_array[2*i+1]
        # Drone 2 position at time i: xy_array[2*N + 2*i], xy_array[2*N + 2*i + 1]
        dx = xy_array[2*i] - xy_array[2*N + 2*i]
        dy = xy_array[2*i + 1] - xy_array[2*N + 2*i + 1]
        dist_sq[i] = dx**2 + dy**2
    return dist_sq

def velocity_constraint_single(xy_array):
    """Returns velocity^2 for each segment of a single drone path (should be <= MAX_VEL^2)"""
    vel_sq = np.zeros(N-1)
    for i in range(1, N):
        vx = (xy_array[2*i] - xy_array[2*i-2]) / dt
        vy = (xy_array[2*i+1] - xy_array[2*i-1]) / dt
        vel_sq[i-1] = vx**2 + vy**2
    return vel_sq

def velocity_constraint_both(xy_array):
    """Returns velocity^2 for both drones"""
    drone1_path = xy_array[0:2*N]
    drone2_path = xy_array[2*N:4*N]
    vel1 = velocity_constraint_single(drone1_path)
    vel2 = velocity_constraint_single(drone2_path)
    return np.concatenate([vel1, vel2])

# Equality constraints for both drones
def equality_constraints_both(xy_array):
    """Returns array of equality constraint values for both drones (should all be 0)"""
    # Drone 1: (0,0) -> (10,10)
    # Drone 2: (10,10) -> (0,0)
    return np.array([
        # Drone 1 constraints
        (xy_array[2]-xy_array[0])**2 + (xy_array[3]-xy_array[1])**2,  # initial vel = 0
        xy_array[0] - MIN_X,                    # initial x = 0
        xy_array[1] - MIN_Y,                    # initial y = 0
        xy_array[2*N-2] - MAX_X,                # final x = MAX_X
        xy_array[2*N-1] - MAX_Y,                # final y = MAX_Y
        # Drone 2 constraints
        (xy_array[2*N+2]-xy_array[2*N])**2 + (xy_array[2*N+3]-xy_array[2*N+1])**2,  # initial vel = 0
        xy_array[2*N] - MAX_X,                  # initial x = MAX_X (starts at 10)
        xy_array[2*N+1] - MAX_Y,                # initial y = MAX_Y (starts at 10)
        xy_array[4*N-2] - MIN_X,                # final x = 0
        xy_array[4*N-1] - MIN_Y,                # final y = 0
    ])

# Set up constraints based on method
if OPTIMIZATION_METHOD == 'trust-constr':
    # Use NonlinearConstraint objects (more efficient)
    constraints = [
        NonlinearConstraint(velocity_constraint_both, -np.inf, MAX_VEL**2),  # vel^2 <= MAX_VEL^2
        NonlinearConstraint(equality_constraints_both, 0, 0),  # equality constraints = 0
        NonlinearConstraint(collision_constraint, distance_tolerance**2, np.inf)  # distance^2 >= tolerance^2
    ]

else:
    # SLSQP uses dict format
    constraints = []
    for i in range(N):
        # Collision avoidance: distance^2 >= distance_tolerance^2
        # Drone 1 at time i: (xy[2*i], xy[2*i+1])
        # Drone 2 at time i: (xy[2*N + 2*i], xy[2*N + 2*i + 1])
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: 
            (xy_array[2*i] - xy_array[2*N + 2*i])**2 + (xy_array[2*i+1] - xy_array[2*N + 2*i + 1])**2 - distance_tolerance**2})
        
        # Drone 1 box constraints
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: xy_array[2*i] - MIN_X})
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: xy_array[2*i+1] - MIN_Y})
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: MAX_X - xy_array[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: MAX_Y - xy_array[2*i+1]})

        # Drone 2 box constraints
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: xy_array[2*N + 2*i] - MIN_X})
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: xy_array[2*N + 2*i + 1] - MIN_Y})
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: MAX_X - xy_array[2*N + 2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: MAX_Y - xy_array[2*N + 2*i + 1]})

        # Velocity constraints (skip first point)
        if i != 0:
            # Drone 1 velocity
            constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: 
                MAX_VEL**2 - (((xy_array[2*i] - xy_array[2*i-2])/dt)**2 + ((xy_array[2*i+1] - xy_array[2*i-1])/dt)**2)})
            # Drone 2 velocity
            constraints.append({'type': 'ineq', 'fun': lambda xy_array, i=i: 
                MAX_VEL**2 - (((xy_array[2*N + 2*i] - xy_array[2*N + 2*i - 2])/dt)**2 + ((xy_array[2*N + 2*i + 1] - xy_array[2*N + 2*i - 1])/dt)**2)})
    
    # Drone 1 equality constraints: (0,0) -> (10,10)
    constraints.append({'type': 'eq', 'fun': lambda xy_array: (xy_array[2]-xy_array[0])**2 + (xy_array[3]-xy_array[1])**2})  # initial vel = 0
    constraints.append({'type': 'eq', 'fun': lambda xy_array: xy_array[0] - MIN_X})  # start x = 0
    constraints.append({'type': 'eq', 'fun': lambda xy_array: xy_array[1] - MIN_Y})  # start y = 0
    constraints.append({'type': 'eq', 'fun': lambda xy_array: xy_array[2*N-2] - MAX_X})  # end x = 10
    constraints.append({'type': 'eq', 'fun': lambda xy_array: xy_array[2*N-1] - MAX_Y})  # end y = 10

    # Drone 2 equality constraints: (10,10) -> (0,0)
    constraints.append({'type': 'eq', 'fun': lambda xy_array: (xy_array[2*N+2]-xy_array[2*N])**2 + (xy_array[2*N+3]-xy_array[2*N+1])**2})  # initial vel = 0
    constraints.append({'type': 'eq', 'fun': lambda xy_array: xy_array[2*N] - MAX_X})  # start x = 10
    constraints.append({'type': 'eq', 'fun': lambda xy_array: xy_array[2*N + 1] - MAX_Y})  # start y = 10
    constraints.append({'type': 'eq', 'fun': lambda xy_array: xy_array[4*N - 2] - MIN_X})  # end x = 0
    constraints.append({'type': 'eq', 'fun': lambda xy_array: xy_array[4*N - 1] - MIN_Y})  # end y = 0
    
    bounds = None  # SLSQP will use the inequality constraints for bounds


#2 DRONES:
#need 3rd dimension for drone id:
#array: [x11, y11, x12, y12, ..., x1N, y1N, x21, y21, x22, y22, ..., x2N, y2N]
#length is 4N
#first half (0 to 2N-1) is drone 1's path
#second half (2N to 4N-1) is drone 2's path

# Drone 1: (0,0) -> (10,10)
x_points_d1 = np.linspace(MIN_X, MAX_X, N)
y_points_d1 = np.linspace(MIN_Y, MAX_Y, N)
initial_path_d1 = np.empty(2 * N)
initial_path_d1[0::2] = x_points_d1
initial_path_d1[1::2] = y_points_d1

# Drone 2: (10,10) -> (0,0) - opposite direction!
x_points_d2 = np.linspace(MAX_X, MIN_X, N)  # 10 -> 0
y_points_d2 = np.linspace(MAX_Y, MIN_Y, N)  # 10 -> 0
initial_path_d2 = np.empty(2 * N)
initial_path_d2[0::2] = x_points_d2
initial_path_d2[1::2] = y_points_d2

x0 = np.concatenate((initial_path_d1, initial_path_d2))


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
    result = opt.minimize(cost_both_drones, x0, tol=gtol, method=OPTIMIZATION_METHOD, jac='2-point',
                          bounds=bounds, constraints=constraints, 
                          callback=callback, options=options)
else:
    result = opt.minimize(cost_both_drones, x0, tol=gtol, method=OPTIMIZATION_METHOD, 
                          constraints=constraints, callback=callback)

print("Final Cost: ", cost_both_drones(result.x))

import matplotlib.pyplot as plt

# Recreate meshgrid for hill cost visualization
x_grid = np.linspace(MIN_X, MAX_X, 100)
y_grid = np.linspace(MIN_Y, MAX_Y, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = hill_cost((X, Y))

plt.figure(figsize=(10, 8))
# Plot the hill cost as a contour map
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
plt.colorbar(contour, label='Hill Cost')

# Extract paths
xy = result.x
# Drone 1 paths
drone1_x0 = initial_path_d1[0::2]
drone1_y0 = initial_path_d1[1::2]
drone1_x_opt = xy[0:2*N:2]
drone1_y_opt = xy[1:2*N:2]

# Drone 2 paths
drone2_x0 = initial_path_d2[0::2]
drone2_y0 = initial_path_d2[1::2]
drone2_x_opt = xy[2*N::2]
drone2_y_opt = xy[2*N+1::2]

# Plot initial guesses (dashed)
plt.plot(drone1_x0, drone1_y0, 'b--', alpha=0.5, linewidth=1, label='Drone 1 Initial')
plt.plot(drone2_x0, drone2_y0, 'r--', alpha=0.5, linewidth=1, label='Drone 2 Initial')

# Plot optimized paths (solid with markers)
plt.plot(drone1_x_opt, drone1_y_opt, 'b-o', markersize=4, linewidth=2, label='Drone 1 Optimized')
plt.plot(drone2_x_opt, drone2_y_opt, 'r-s', markersize=4, linewidth=2, label='Drone 2 Optimized')

# Mark start and end points
plt.scatter([drone1_x_opt[0]], [drone1_y_opt[0]], c='blue', s=150, marker='^', zorder=5, edgecolors='black', label='Drone 1 Start')
plt.scatter([drone1_x_opt[-1]], [drone1_y_opt[-1]], c='blue', s=150, marker='v', zorder=5, edgecolors='black', label='Drone 1 End')
plt.scatter([drone2_x_opt[0]], [drone2_y_opt[0]], c='red', s=150, marker='^', zorder=5, edgecolors='black', label='Drone 2 Start')
plt.scatter([drone2_x_opt[-1]], [drone2_y_opt[-1]], c='red', s=150, marker='v', zorder=5, edgecolors='black', label='Drone 2 End')

plt.xlim(MIN_X - 0.5, MAX_X + 0.5)
plt.ylim(MIN_Y - 0.5, MAX_Y + 0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Two Drone Path Optimization (Collision Tolerance: {distance_tolerance})')
plt.legend(loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
if ONE_HILL:
    plt.savefig('plots/2_drones_1_hill_path_opt.png', dpi=150)
else:
    plt.savefig('plots/2_drones_many_hills_path_opt.png', dpi=150)
plt.show()

# Print minimum distance between drones
min_dist = np.inf
min_dist_idx = 0
for i in range(N):
    dx = drone1_x_opt[i] - drone2_x_opt[i]
    dy = drone1_y_opt[i] - drone2_y_opt[i]
    dist = np.sqrt(dx**2 + dy**2)
    if dist < min_dist:
        min_dist = dist
        min_dist_idx = i
print(f"\nMinimum distance between drones: {min_dist:.4f} at time step {min_dist_idx}")
print(f"Distance tolerance: {distance_tolerance}")


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
        plt.savefig('plots/2_drones_1_hill_constraint_convergence.png', dpi=150)
    else:
        plt.savefig('plots/2_drones_many_hills_constraint_convergence.png', dpi=150)
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
        plt.savefig('plots/2_drones_1_hill_lagrangian_gradient_convergence.png', dpi=150)
    else:
        plt.savefig('plots/2_drones_many_hills_lagrangian_gradient_convergence.png', dpi=150)
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
