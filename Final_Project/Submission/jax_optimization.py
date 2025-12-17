# %%
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import partial
# Import FD solver for final solution verification
from heat_eq_2D_opt import HeatEquation2D

# Enable float64 for better numerical precision
jax.config.update("jax_enable_x64", True)

# ============================================================================
# Physical and Material Constants
# ============================================================================
# Domain
CPU_X = 0.04  # m
CPU_Y = 0.04  # m
CPU_Z = 0.04  # m (height/thickness)
N = 25  # Grid points per direction

# Silicon properties
K_SI = 149.0  # W/(m·K)
RHO_SI = 2323.0  # kg/m³
C_SI = 19.789 / 28.085 * 1000  # J/(kg·K)
THERMAL_ALPHA = K_SI / (RHO_SI * C_SI)

# Air properties
EXT_K = 0.02772  # W/(m·K)
EXT_PR = 0.7215  # Prandtl number
EXT_NU = 1.506e-5  # m²/s kinematic viscosity
EXT_T = 293.0  # K (20°C)

# Numerical parameters
CFL = 0.25  # Match working code
MAX_ITER = 500  # Match working code (was 500, too few iterations)  

# Derived mesh quantities
dx = CPU_X / (N - 1)
dy = CPU_Y / (N - 1)
dt = CFL * (dx * dy) / THERMAL_ALPHA
tau = THERMAL_ALPHA * dt / (dx * dy)

# Create mesh grids (static)
x_axis = jnp.linspace(0, CPU_X, N)
y_axis = jnp.linspace(0, CPU_Y, N)
X, Y = jnp.meshgrid(x_axis, y_axis, indexing='ij')


# ============================================================================
# Pure Functions for Physics
# ============================================================================

def fan_efficiency(v):
    """Fan efficiency as function of velocity"""
    return -0.002 * v**2 + 0.08 * v


def heat_generation(X, Y, a, b, c):
    """Volumetric heat generation: q = a*x + b*y + c"""
    return a * X + b * Y + c


def compute_total_heat(a, b, c):
    """Compute total heat generation using trapezoidal rule.
    
    """
    q = heat_generation(X, Y, a, b, c) * dx * dy * CPU_Z
    i0, iN, j0, jN = 0, N - 1, 0, N - 1
    
    # Match 's "buggy" implementation for consistency
    # Faces - note j0 and jN index ROWS, not columns (matching 's code)
    q = q.at[i0, :].divide(2.0)
    q = q.at[iN, :].divide(2.0)
    q = q.at[j0, :].divide(2.0)  # This divides row 0 again!
    q = q.at[jN, :].divide(2.0)  # This divides row N-1 again!
    
    # Corners
    q = q.at[i0, j0].divide(2.0)
    q = q.at[iN, jN].divide(2.0)
    q = q.at[iN, j0].divide(2.0)
    q = q.at[i0, jN].divide(2.0)
    
    return jnp.sum(q)


def h_boundary(u):
    """Natural convection heat transfer coefficient (Churchill-Chu)"""
    # Use film temperature for properties
    beta = 1.0 / ((u + EXT_T) / 2.0)
    
    # Temperature difference - DO NOT use abs() as it breaks automatic differentiation!
    rayleigh = 9.81 * beta * (u - EXT_T) * dx**3 / (EXT_NU**2) * EXT_PR
    rayleigh = jnp.maximum(rayleigh, 0.0)  # Minimum Rayleigh (0.0, not 1.0)
    
    nusselt = (0.825 + (0.387 * rayleigh**(1/6)) /
               (1 + (0.492/EXT_PR)**(9/16))**(8/27))**2
    return nusselt * EXT_K / dx


def h_top(v):
    """Forced convection from fan (flat plate correlation)"""
    Rex = v * X / EXT_NU
    Rex = jnp.maximum(Rex, 1e-10)  # Avoid division issues
    
    # Piecewise Nusselt: laminar (Rex < 5e5) vs turbulent
    Nux = jnp.where(
        Rex < 5e5,
        0.332 * Rex**0.5 * EXT_PR**(1/3),
        0.0296 * Rex**0.8 * EXT_PR**(1/3)
    )
    return Nux * EXT_K / (X + 1e-8)


def initial_condition():
    """Initial temperature distribution (Cosine profile to match FD version)"""
    return 70 * jnp.sin(X * jnp.pi / CPU_X) * jnp.sin(Y * jnp.pi / CPU_Y) + EXT_T


# ============================================================================
# Time-stepping (Pure Functional)
# ============================================================================

def step_forward(u, X_grid, Y_grid, params):
    """Single time step update - pure function, returns new u.

    - u: current temperature field
    - X_grid, Y_grid: mesh grids (passed explicitly, not as closures)
    - params: dictionary with 'v', 'a', 'b', 'c' keys
    
    This structure with explicit arguments (not closures) is CRITICAL
    for JAX to properly trace and differentiate through lax.scan.
    """
    v = params['v']
    a = params['a']
    b = params['b']
    c = params['c']
    
    i0, j0, iN, jN = 0, 0, N - 1, N - 1
    
    # Compute heat transfer coefficients
    h_b = h_boundary(u)
    h_t = h_top(v)
    e_dot = heat_generation(X_grid, Y_grid, a, b, c)
    
    old_u = u
    new_u = u
    
    # Interior update (central difference)
    interior = (
        old_u[1:-1, 1:-1] +
        tau * (
            dy * (old_u[2:, 1:-1] - 2*old_u[1:-1, 1:-1] + old_u[:-2, 1:-1]) / dx +
            dx * (old_u[1:-1, 2:] - 2*old_u[1:-1, 1:-1] + old_u[1:-1, :-2]) / dy
        ) +
        tau * h_t[1:-1, 1:-1] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[1:-1, 1:-1]) +
        tau * dx * dy / K_SI * e_dot[1:-1, 1:-1]
    )
    new_u = new_u.at[1:-1, 1:-1].set(interior)
    
    # Left boundary (i=0)
    left = (
        old_u[i0, 1:-1] +
        2 * tau * h_b[i0, 1:-1] / K_SI * dy * (EXT_T - old_u[i0, 1:-1]) +
        tau * dx * (old_u[i0, 2:] - old_u[i0, 1:-1]) / dy +
        tau * dx * (old_u[i0, 1:-1] - old_u[i0, 2:]) / dy +
        2 * tau * dy * (old_u[i0+1, 1:-1] - old_u[i0, 1:-1]) / dx +
        tau * h_t[i0, 1:-1] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[i0, 1:-1]) +
        tau * e_dot[i0, 1:-1] / K_SI * dx * dy
    )
    new_u = new_u.at[i0, 1:-1].set(left)
    
    # Right boundary (i=N-1)
    right = (
        old_u[iN, 1:-1] +
        2 * tau * h_b[iN, 1:-1] / K_SI * dy * (EXT_T - old_u[iN, 1:-1]) +
        tau * dx * (old_u[iN, 2:] - old_u[iN, 1:-1]) / dy +
        tau * dx * (old_u[iN, 1:-1] - old_u[iN, 2:]) / dy +
        2 * tau * dy * (old_u[iN-1, 1:-1] - old_u[iN, 1:-1]) / dx +
        tau * h_t[iN, 1:-1] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[iN, 1:-1]) +
        tau * e_dot[iN, 1:-1] / K_SI * dx * dy
    )
    new_u = new_u.at[iN, 1:-1].set(right)
    
    # Bottom boundary (j=0)
    bottom = (
        old_u[1:-1, j0] +
        2 * tau * h_b[1:-1, j0] / K_SI * dx * (EXT_T - old_u[1:-1, j0]) +
        tau * dy * (old_u[2:, j0] - old_u[1:-1, j0]) / dx +
        tau * dy * (old_u[1:-1, j0] - old_u[2:, j0]) / dx +
        2 * tau * dx * (old_u[1:-1, j0+1] - old_u[1:-1, j0]) / dy +
        tau * h_t[1:-1, j0] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[1:-1, j0]) +
        tau * e_dot[1:-1, j0] / K_SI * dx * dy
    )
    new_u = new_u.at[1:-1, j0].set(bottom)
    
    # Top boundary (j=N-1)
    top = (
        old_u[1:-1, jN] +
        2 * tau * h_b[1:-1, jN] / K_SI * dx * (EXT_T - old_u[1:-1, jN]) +
        tau * dy * (old_u[2:, jN] - old_u[1:-1, jN]) / dx +
        tau * dy * (old_u[1:-1, jN] - old_u[2:, jN]) / dx +
        2 * tau * dx * (old_u[1:-1, jN-1] - old_u[1:-1, jN]) / dy +
        tau * h_t[1:-1, jN] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[1:-1, jN]) +
        tau * e_dot[1:-1, jN] / K_SI * dx * dy
    )
    new_u = new_u.at[1:-1, jN].set(top)
    
    # Corners
    # Bottom-left
    bl = (
        old_u[i0, j0] +
        2 * tau * h_b[i0, j0] * (dy + dx) / K_SI * (EXT_T - old_u[i0, j0]) +
        2 * tau * dx * (old_u[i0, j0+1] - old_u[i0, j0]) / dy +
        2 * tau * dy * (old_u[i0+1, j0] - old_u[i0, j0]) / dx +
        tau * h_t[i0, j0] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[i0, j0]) +
        tau * e_dot[i0, j0] / K_SI * dx * dy
    )
    new_u = new_u.at[i0, j0].set(bl)
    
    # Bottom-right
    br = (
        old_u[iN, j0] +
        2 * tau * h_b[iN, j0] * (dy + dx) / K_SI * (EXT_T - old_u[iN, j0]) +
        2 * tau * dx * (old_u[iN, j0+1] - old_u[iN, j0]) / dy +
        2 * tau * dy * (old_u[iN-1, j0] - old_u[iN, j0]) / dx +
        tau * h_t[iN, j0] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[iN, j0]) +
        tau * e_dot[iN, j0] / K_SI * dx * dy
    )
    new_u = new_u.at[iN, j0].set(br)
    
    # Top-left
    tl = (
        old_u[i0, jN] +
        2 * tau * h_b[i0, jN] * (dy + dx) / K_SI * (EXT_T - old_u[i0, jN]) +
        2 * tau * dx * (old_u[i0, jN-1] - old_u[i0, jN]) / dy +
        2 * tau * dy * (old_u[i0+1, jN] - old_u[i0, jN]) / dx +
        tau * h_t[i0, jN] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[i0, jN]) +
        tau * e_dot[i0, jN] / K_SI * dx * dy
    )
    new_u = new_u.at[i0, jN].set(tl)
    
    # Top-right
    tr = (
        old_u[iN, jN] +
        2 * tau * h_b[iN, jN] * (dy + dx) / K_SI * (EXT_T - old_u[iN, jN]) +
        2 * tau * dx * (old_u[iN, jN-1] - old_u[iN, jN]) / dy +
        2 * tau * dy * (old_u[iN-1, jN] - old_u[iN, jN]) / dx +
        tau * h_t[iN, jN] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[iN, jN]) +
        tau * e_dot[iN, jN] / K_SI * dx * dy
    )
    new_u = new_u.at[iN, jN].set(tr)
    
    return new_u


@partial(jax.jit, static_argnums=(4,))
def run_solver(u0, X_grid, Y_grid, params, n_steps):
    """
    JIT-compiled solver that runs n_steps of the heat equation.
    
    - params is a TRACED argument (not static), so JAX can differentiate through it
    - n_steps is STATIC (position 4), so JAX compiles for each different value
    - body_fun closes over params, but since params is traced by this JIT'd function,
      JAX can properly compute gradients through the closure
    
    This is the KEY pattern that makes gradients work correctly!
    """
    def body_fun(u, _):
        u_next = step_forward(u, X_grid, Y_grid, params)
        return u_next, None
    
    u_final, _ = lax.scan(body_fun, u0, xs=None, length=n_steps)
    return u_final


def solve_steady_state(v, a, b, c, num_iter=MAX_ITER):
    """
    Solve to steady state using the JIT-compiled run_solver.
    
    Returns the steady-state temperature field.
    """
    u0 = initial_condition()
    
    # Pack parameters into a dict - this is passed to the JIT'd solver
    params = {'v': v, 'a': a, 'b': b, 'c': c}
    
    # Call the JIT'd solver - params flows through as a traced pytree
    final_u = run_solver(u0, X, Y, params, num_iter)
    
    return final_u


def solve_steady_state_with_error(v, a, b, c, num_iter=MAX_ITER):
    """
    Solve to steady state and return both the field and the convergence error.
    
    Returns (u_final, error) where error = ||u_new - u_old||_inf
    """
    u0 = initial_condition()
    
    # Pack parameters into a dict
    params = {'v': v, 'a': a, 'b': b, 'c': c}
    
    # Use the JIT'd solver
    final_u = run_solver(u0, X, Y, params, num_iter)
    
    # Compute error by doing one more step
    u_one_more = step_forward(final_u, X, Y, params)
    error = jnp.linalg.norm(u_one_more - final_u, jnp.inf)
    
    return final_u, error


# ============================================================================
# Objective Function (JAX-differentiable)
# ============================================================================

# Weights for multi-objective
W1 = 0.2  # weight on max temperature
W2 = 0.8  # weight on fan efficiency

def objective_function(x):
    """
    Objective: w1 * max(T)/273 - w2 * eta(v)
    
    x = [v, a, b, c]
    
    """
    v, a, b, c = x 
    
    # Solve for steady-state temperature
    u = solve_steady_state(v, a, b, c)
    
    # Compute objective components
    max_T = jnp.max(u)
    eta = fan_efficiency(v)
    
    return W1 * max_T / 273.0 - W2 * eta


def constraint_total_heat(x):
    """Equality constraint: total heat = 10W"""
    v, a, b, c = x  # Tuple unpacking

    return compute_total_heat(a, b, c) - 10.0


# ============================================================================
# Gradient Computation and Comparison
# ============================================================================

objective_grad_jax = jax.grad(objective_function)

def AD_objective_grad(x):
    return np.array(objective_grad_jax(x))

objective_hess_jax = jax.hessian(objective_function)

def AD_objective_hess(x):
    return np.array(objective_hess_jax(x))


constraint_grad_jax = jax.grad(constraint_total_heat)

def AD_constraint_grad(x):
    return np.array(constraint_grad_jax(x))

# %% 
# Optimization with Scipy

x_path = []
norm_grad_l = []
norm_grad_f = []
objective_history = []
max_T_history = []
eta_history = []
constraint_history = []
grad_obj_history = []


def callback2(xk):
    """Callback for SLSQP - records state after each iteration.
    
    Note: heq should already be updated from the last objective_function call,
    so we just read current values without re-solving.
    """
    x_jax = jnp.array(xk)
    
    obj_val = float(objective_function(x_jax))
    u = solve_steady_state(xk[0], xk[1], xk[2], xk[3])
    max_T = float(jnp.max(u))
    eta = float(fan_efficiency(xk[0]))
    constraint_val = float(compute_total_heat(xk[1], xk[2], xk[3])) - 10.0
    grad_obj = objective_grad_jax(xk)
    
    # Store history
    x_path.append(xk.copy())
    objective_history.append(obj_val)
    max_T_history.append(max_T)
    eta_history.append(eta)
    constraint_history.append(constraint_val)
    grad_obj_history.append(np.linalg.norm(grad_obj))
    
    # Print progress
    print(f"  Iter {len(x_path):3d}: obj={obj_val:.4f}, maxT={max_T-273:.2f}°C, η={eta:.4f}, v={xk[0]:.2f}")


def callback(intermediate_result):
    """Callback to track optimization progress (trust-constr signature)"""
    xk = intermediate_result.x
    x_jax = jnp.array(xk)

    # Extract Lagrangian gradient norm
    grad_L = intermediate_result.lagrangian_grad
    grad_L_norm = np.linalg.norm(grad_L) if grad_L is not None else 0.0
    norm_grad_l.append(grad_L_norm)
    
    # Compute current values
    obj_val = float(objective_function(x_jax))
    u = solve_steady_state(xk[0], xk[1], xk[2], xk[3])
    max_T = float(jnp.max(u))
    eta = float(fan_efficiency(xk[0]))
    constraint_val = float(compute_total_heat(xk[1], xk[2], xk[3])) - 10.0
    grad_obj = objective_grad_jax(xk)
    
    # Store history
    x_path.append(xk.copy())
    objective_history.append(obj_val)
    max_T_history.append(max_T)
    eta_history.append(eta)
    constraint_history.append(constraint_val)
    grad_obj_history.append(np.linalg.norm(grad_obj))
    
    # Note: We can track grad_L history too if we add a list for it
    # But here we just print it
    
    # Print progress (len(x_path) includes initial guess, so this is iteration number)
    print(f"  Iter {len(x_path)-1:3d}: obj={obj_val:.6f}, maxT={max_T-273:.2f}°C, "
            f"η={eta:.4f}, |∇f|={np.linalg.norm(grad_obj):.2e}, |∇L|={grad_L_norm:.2e}, "
            f"vars: v={xk[0]:.2f}, a={xk[1]:.2f}, b={xk[2]:.2f}, c={xk[3]:.2f}")
    if len(x_path) <= 3:  # Print gradient breakdown for first few iterations (including initial)
        print(f"      Gradients: ∇f_v={grad_obj[0]:.6e}, ∇f_a={grad_obj[1]:.6e}, "
                f"∇f_b={grad_obj[2]:.6e}, ∇f_c={grad_obj[3]:.6e}")
    
    return False  # Don't stop optimization

## Bounds for inputs
# v: fan velocity [0, 30] m/s (physical limits)
# a, b: heat distribution coefficients (unconstrained)
# c: base heat generation (can be negative if a,b contribute enough)
bounds = [
    (0.1, 30),        # v: fan velocity (avoid 0 for numerical stability)
    (-1e8, 1e8),      # a
    (-1e8, 1e8),      # b
    (0, 1e8),      # c
]

## Setting the constraints
constraints = [
    {'type': 'eq', 'fun': constraint_total_heat, 'jac': AD_constraint_grad},
]
## Creating the initial guess
# For 10W total power in 0.04³ m³ volume with a=b=0: c = 10 / (0.04 * 0.04 * 0.04) ≈ 156250 W/m³
v0 = 19.0  # m/s - start at reasonable fan velocity
a0 = 10.0   # Start with uniform heat generation
b0 = 10.0
c0 = 156250.0  # This gives approximately 10W total
x0 = [v0, a0, b0, c0]


# ============================================================
# SLSQP Initialize
x_jax = jnp.array(x0)

obj_val = float(objective_function(x_jax))
u = solve_steady_state(x0[0], x0[1], x0[2], x0[3])
max_T = float(jnp.max(u))
eta = float(fan_efficiency(x0[0]))
constraint_val = float(compute_total_heat(x0[1], x0[2], x0[3])) - 10.0
grad_obj = objective_grad_jax(x0)

# Store history
x_path.append(x0.copy())
objective_history.append(obj_val)
max_T_history.append(max_T)
eta_history.append(eta)
constraint_history.append(constraint_val)
grad_obj_history.append(np.linalg.norm(grad_obj))
# =================================================================


def zero_hess(x):
    n = len(x)
    return np.zeros((n, n))

print(f"Initial guess: v={v0} m/s, a={a0}, b={b0}, c={c0}")
print(f"Starting optimization with w1={W1}, w2={W2}...")
print("-" * 50)
## Optimize
optimization_result = optimize.minimize(
    objective_function,
    method='SLSQP',
    x0=x0,
    bounds=bounds,
    jac=AD_objective_grad,
    # hess='2-point',
    constraints=constraints,
    callback=callback2,
    options={'maxiter': 50, 'verbose': 3}
)

optimal_x = optimization_result.x



optimal_u = solve_steady_state(optimal_x[0], optimal_x[1], optimal_x[2], optimal_x[3])
optimal_max_T = np.max(optimal_u)
optimal_eta = float(fan_efficiency(optimal_x[0]))
optimal_obj = objective_function(optimal_x)
total_heat = compute_total_heat(optimal_x[1], optimal_x[2], optimal_x[3])

print("\n" + "=" * 50)
print("OPTIMIZATION RESULTS")
print("=" * 50)
print(f"Converged: {optimization_result.success}")
print(f"Message: {optimization_result.message}")
print(f"Number of iterations: {optimization_result.nit}")
print("-" * 50)
print("Optimal Design Variables:")
print(f"  v (fan velocity)   = {optimization_result.x[0]:.4f} m/s")
print(f"  a (heat coeff)     = {optimization_result.x[1]:.4f} W/m⁴")
print(f"  b (heat coeff)     = {optimization_result.x[2]:.4f} W/m⁴")
print(f"  c (heat coeff)     = {optimization_result.x[3]:.4f} W/m³")
print("-" * 50)
print("Objective Function Components:")
print(f"  Max Temperature    = {optimal_max_T:.2f} K ({optimal_max_T - 273:.2f} °C)")
print(f"  Fan Efficiency η   = {optimal_eta:.4f}")
print(f"  Objective Value    = {optimal_obj:.6f}")
print("-" * 50)
print("Constraint Satisfaction:")
print(f"  Total Heat Gen.    = {total_heat:.4f} W (target: 10 W)")
print(f"  Constraint Error   = {total_heat - 10:.6f} W")
print("=" * 50)

# Convert tracking lists to numpy arrays
x_path = np.array(x_path)
iterations = np.arange(len(x_path))

v_path = x_path[:, 0]
a_path = x_path[:, 1]
b_path = x_path[:, 2]
c_path = x_path[:, 3]

norm_grad_l = np.array(norm_grad_l)
norm_grad_f = np.array(norm_grad_f)
objective_history = np.array(objective_history)
max_T_history = np.array(max_T_history)
eta_history = np.array(eta_history)
constraint_history = np.array(constraint_history)

# ========== Part (b) Required Plots ==========

# 1. Gradient of Lagrangian Convergence (skip if no gradient data available)
if not np.all(np.isnan(norm_grad_l)):
    plt.figure(figsize=(8, 5))
    plt.semilogy(iterations, norm_grad_l, 'b-', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|\nabla \mathcal{L}\|$')
    plt.title('Convergence of Gradient of Lagrangian')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_lagrangian.png', dpi=150)
    plt.show()
else:
    print("Note: Gradient of Lagrangian not available for SLSQP method")

# 2. Objective Function Convergence
plt.figure(figsize=(8, 5))
plt.plot(iterations, objective_history, 'r-', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel(r'$\omega_1 \max(T) - \omega_2 \eta$')
plt.title('Objective Function vs Iteration')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('objective_function.png', dpi=150)
plt.show()

# 3. Maximum Temperature
plt.figure(figsize=(8, 5))
plt.plot(iterations, max_T_history - 273, 'g-', linewidth=1.5)  # Convert to Celsius
plt.xlabel('Iteration')
plt.ylabel(r'$\max(T)$ [°C]')
plt.title('Maximum Temperature vs Iteration')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('max_temperature.png', dpi=150)
plt.show()

# 4. Fan Efficiency
plt.figure(figsize=(8, 5))
plt.plot(iterations, eta_history, 'm-', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel(r'$\eta$')
plt.title('Fan Efficiency vs Iteration')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fan_efficiency.png', dpi=150)
plt.show()

# 5. Total Power Generation Constraint
plt.figure(figsize=(8, 5))
plt.plot(iterations, constraint_history, 'c-', linewidth=1.5)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8, label='Constraint = 0')
plt.xlabel('Iteration')
plt.ylabel(r'$\int\!\!\int\!\!\int f(x,y)\,dV - 10$ [W]')
plt.title('Total Power Generation Constraint vs Iteration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('power_constraint.png', dpi=150)
plt.show()

# 6. Design Parameters: Fan Velocity v
plt.figure(figsize=(8, 5))
plt.plot(iterations, v_path, 'b-', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel(r'$v$ [m/s]')
plt.title('Fan Velocity vs Iteration')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('velocity_path.png', dpi=150)
plt.show()

# 7. Design Parameters: a, b, c
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axes[0].plot(iterations, a_path, 'r-', linewidth=1.5)
axes[0].set_ylabel(r'$a$ [W/m$^4$]')
axes[0].set_title('Heat Generation Parameters vs Iteration')
axes[0].grid(True, alpha=0.3)

axes[1].plot(iterations, b_path, 'g-', linewidth=1.5)
axes[1].set_ylabel(r'$b$ [W/m$^4$]')
axes[1].grid(True, alpha=0.3)

axes[2].plot(iterations, c_path, 'b-', linewidth=1.5)
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel(r'$c$ [W/m$^3$]')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('abc_parameters.png', dpi=150)
plt.show()

# 8. Optimal Temperature Distribution
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X * 1000, Y * 1000, optimal_u - 273, levels=20, cmap='hot')
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Temperature [°C]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title('Optimal Steady-State Temperature Distribution')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('optimal_temperature_distribution.png', dpi=150)
plt.show()

# 9. Gradient of Objective Function (skip if no gradient data available)
if not np.all(np.isnan(norm_grad_f)):
    plt.figure(figsize=(8, 5))
    plt.semilogy(iterations, norm_grad_f, 'k-', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|\nabla f\|$')
    plt.title('Convergence of Gradient of Objective Function')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_objective.png', dpi=150)
    plt.show()


# %%
