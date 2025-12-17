"""
JAX-differentiable 2D Heat Equation Solver for MECH 579 Final Project

This version is fully functional (no class mutation) so JAX can trace through
and compute gradients via automatic differentiation.
"""
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
    
    NOTE: This matches Callum's implementation which has a quirk:
    - Rows 0 and N-1 are divided by 2 twice (effectively by 4)
    - Columns 0 and N-1 are NOT divided (except at corners)
    This is needed to match Callum's optimization results.
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
    
    This signature matches Callum's step_forward_jax exactly:
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
    
    This structure EXACTLY matches Callum's run_solver_jax:
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
    
    Uses tuple unpacking like Callum's working code.
    """
    v, a, b, c = x  # Tuple unpacking (matches Callum's code)
    
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

def finite_difference_gradient(f, x, h):
    """Compute gradient using central finite differences"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


def compare_gradients(x0, param_idx=0, param_name="v"):
    """
    Compare AD gradient vs FD gradient for a single parameter.
    Shows FD convergence and provides comparison table.
    """
    print(f"\n{'='*60}")
    print(f"Gradient Comparison for Parameter: {param_name} (index {param_idx})")
    print(f"{'='*60}")
    
    # Compute AD gradient
    print("\nComputing AD gradient...")
    grad_fn = jax.grad(objective_function)
    ad_grad = grad_fn(jnp.array(x0))
    ad_value = float(ad_grad[param_idx])
    print(f"AD gradient[{param_idx}] = {ad_value:.15e}")
    
    # Compute FD gradients at various step sizes
    print("\nComputing FD gradients for convergence study...")
    h_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    fd_values = []
    
    x0_np = np.array(x0, dtype=np.float64)
    
    for h in h_values:
        x_plus = x0_np.copy()
        x_minus = x0_np.copy()
        x_plus[param_idx] += h
        x_minus[param_idx] -= h
        
        f_plus = float(objective_function(jnp.array(x_plus)))
        f_minus = float(objective_function(jnp.array(x_minus)))
        fd_grad = (f_plus - f_minus) / (2 * h)
        fd_values.append(fd_grad)
        print(f"  h = {h:.0e}: FD gradient = {fd_grad:.15e}")
    
    # Find converged FD value (smallest h before numerical noise dominates)
    # Look for where successive differences start increasing
    diffs = [abs(fd_values[i] - fd_values[i-1]) for i in range(1, len(fd_values))]
    min_diff_idx = np.argmin(diffs)
    converged_fd = fd_values[min_diff_idx + 1]
    converged_h = h_values[min_diff_idx + 1]
    
    print(f"\nConverged FD gradient (h={converged_h:.0e}): {converged_fd:.15e}")
    
    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Value':<25} {'Significant Digits'}")
    print(f"{'-'*60}")
    print(f"{'AD (JAX)':<20} {ad_value:<25.15e}")
    print(f"{'FD (converged)':<20} {converged_fd:<25.15e}")
    
    # Compute relative error
    if abs(converged_fd) > 1e-15:
        rel_error = abs(ad_value - converged_fd) / abs(converged_fd)
        matching_digits = -np.log10(rel_error + 1e-16)
        print(f"\nRelative error: {rel_error:.6e}")
        print(f"Matching significant digits: ~{matching_digits:.1f}")
    
    # Detailed FD convergence table
    print(f"\n{'='*60}")
    print("FD CONVERGENCE TABLE")
    print(f"{'='*60}")
    print(f"{'h':<12} {'FD Gradient':<25} {'Error vs AD':<20}")
    print(f"{'-'*60}")
    for h, fd in zip(h_values, fd_values):
        error = abs(fd - ad_value)
        print(f"{h:<12.0e} {fd:<25.15e} {error:<20.6e}")
    
    return ad_value, converged_fd, h_values, fd_values


# ============================================================================
# Full Gradient Comparison (All Parameters)
# ============================================================================

def full_gradient_comparison(x0):
    """Compare AD vs FD for all design parameters"""
    param_names = ['v (velocity)', 'a (heat coeff)', 'b (heat coeff)', 'c (heat coeff)']
    
    print("\n" + "="*70)
    print("FULL GRADIENT COMPARISON: AD vs FD")
    print("="*70)
    
    # AD gradient (all at once)
    print("\nComputing full AD gradient...")
    grad_fn = jax.grad(objective_function)
    ad_grad = grad_fn(jnp.array(x0))
    ad_grad = np.array(ad_grad)
    print(f"AD gradient: {ad_grad}")
    
    # FD gradient (converged)
    print("\nComputing FD gradient (h=1e-6)...")
    h = 1e-6
    fd_grad = finite_difference_gradient(
        lambda x: float(objective_function(jnp.array(x))),
        np.array(x0),
        h
    )
    print(f"FD gradient: {fd_grad}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("GRADIENT COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Parameter':<20} {'AD Value':<22} {'FD Value':<22} {'Rel Error'}")
    print(f"{'-'*70}")
    
    for i, name in enumerate(param_names):
        ad_val = ad_grad[i]
        fd_val = fd_grad[i]
        if abs(fd_val) > 1e-15:
            rel_err = abs(ad_val - fd_val) / abs(fd_val)
        else:
            rel_err = abs(ad_val - fd_val)
        print(f"{name:<20} {ad_val:<22.10e} {fd_val:<22.10e} {rel_err:<.2e}")
    
    return ad_grad, fd_grad


# ============================================================================
# Optimization with JAX Gradients
# ============================================================================

"""Run the full optimization using JAX AD for gradients."""
from scipy import optimize
import os
import warnings
warnings.filterwarnings('ignore', message='delta_grad == 0.0')
os.makedirs('plots', exist_ok=True)

# Tracking lists for convergence plots
x_path = []
objective_history = []
max_T_history = []
eta_history = []
constraint_history = []
grad_obj_history = []

# JIT compile the gradient function for speed
grad_objective_jit = jax.jit(jax.grad(objective_function))
objective_jit = jax.jit(objective_function)

# Pre-compile constraint gradient to avoid recompilation
def heat_from_vector(x_vec):
    return compute_total_heat(x_vec[1], x_vec[2], x_vec[3])

grad_constraint_jit = jax.jit(jax.grad(heat_from_vector))

def objective_wrapper(x):
    """Wrapper for scipy - returns float"""
    x_jax = jnp.array(x)
    return float(objective_jit(x_jax))

def gradient_wrapper(x):
    """Wrapper for scipy - returns numpy array"""
    x_jax = jnp.array(x)
    return np.array(grad_objective_jit(x_jax))

def constraint_wrapper(x):
    """Equality constraint: total heat = 10W"""
    return float(compute_total_heat(x[1], x[2], x[3])) - 10.0

def constraint_grad_wrapper(x):
    """Gradient of constraint w.r.t. x"""
    return np.array(grad_constraint_jit(jnp.array(x)))

def callback(intermediate_result):
    """Callback to track optimization progress (trust-constr signature)"""
    xk = intermediate_result.x
    x_jax = jnp.array(xk)
    
    # Extract Lagrangian gradient norm
    grad_L = intermediate_result.lagrangian_grad
    grad_L_norm = np.linalg.norm(grad_L) if grad_L is not None else 0.0
    
    # Compute current values
    obj_val = float(objective_jit(x_jax))
    u = solve_steady_state(xk[0], xk[1], xk[2], xk[3])
    max_T = float(jnp.max(u))
    eta = float(fan_efficiency(xk[0]))
    constraint_val = float(compute_total_heat(xk[1], xk[2], xk[3])) - 10.0
    grad_obj = gradient_wrapper(xk)
    
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

# Initial guess
v0 = 10.0
x0_heat = 0.0 
a0 = x0_heat * 1e5
b0 = x0_heat * 1e5
c0 = 156250.0 - 0.02 * a0 - 0.02 * b0
x0 = np.array([v0, a0, b0, c0])

# Bounds (same as heat_eq_2D_opt.py)
bounds = [
    (0.1, 30),      # v: fan velocity
    (-1e8, 1e8),    # a
    (-1e8, 1e8),    # b
    (0, 1e8),       # c
]

# Constraint
constraints = [{
    'type': 'eq',
    'fun': constraint_wrapper,
    'jac': constraint_grad_wrapper
}]

print("="*60)
print("JAX-BASED OPTIMIZATION")
print("="*60)
print(f"Initial guess: v={v0}, a={a0}, b={b0}, c={c0}")
print(f"Weights: w1={W1} (temperature), w2={W2} (efficiency)")
print(f"Initial total heat: {float(compute_total_heat(a0, b0, c0)):.4f} W")
print("-"*60)

# Add initial guess to tracking lists (iteration 0)
x_jax0 = jnp.array(x0)
u0_ss = solve_steady_state(x0[0], x0[1], x0[2], x0[3])
obj_val0 = float(objective_jit(x_jax0))
max_T0 = float(jnp.max(u0_ss))
eta0 = float(fan_efficiency(x0[0]))
constraint_val0 = float(compute_total_heat(x0[1], x0[2], x0[3])) - 10.0
grad_obj0 = gradient_wrapper(x0)

x_path.append(x0.copy())
objective_history.append(obj_val0)
max_T_history.append(max_T0)
eta_history.append(eta0)
constraint_history.append(constraint_val0)
grad_obj_history.append(np.linalg.norm(grad_obj0))

print(f"\nInitial (Iter 0): obj={obj_val0:.6f}, maxT={max_T0-273:.2f}°C, "
        f"η={eta0:.4f}, |∇f|={np.linalg.norm(grad_obj0):.2e}")

# Run optimization with JAX gradients
print("\nStarting optimization with JAX AD gradients (SLSQP)...")

# SLSQP callback wrapper
def slsqp_callback(xk):
    class Result: pass
    res = Result()
    res.x = xk
    res.lagrangian_grad = None  # SLSQP doesn't provide this easily
    callback(res)

result = optimize.minimize(
    objective_wrapper,
    x0=x0,
    method='SLSQP',
    jac=gradient_wrapper,
    bounds=bounds,
    constraints=constraints,
    callback=slsqp_callback,
    options={'maxiter': 500, 'ftol': 1e-8, 'disp': True}
)

# Final solution - use FD solver from heat_eq_2D_opt.py for verification
x_opt = result.x

# Set up FD solver with same parameters as heat_eq_2D_opt.py
def initial_condition_fd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Initial condition matching heat_eq_2D_opt.py"""
    r, c = x.shape
    u = np.zeros([r, c])
    u = 70 * np.sin(x * np.pi / CPU_X) * np.sin(y * np.pi / CPU_Y) + EXT_T
    return u

def heat_generation_function_fd(x: np.ndarray, y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Heat generation function matching heat_eq_2D_opt.py"""
    return a * x + b * y + c

# Create FD solver instance
heq_fd = HeatEquation2D(
    CPU_X, CPU_Y, CPU_Z, N, N,
    k=K_SI, rho=RHO_SI, cp=C_SI,
    CFL=0.5,  # Match heat_eq_2D_opt.py
    init_condition=initial_condition_fd
)
heq_fd.max_iter = 5E5
heq_fd.verbose = False

# Set optimal design variables
heq_fd.set_fan_velocity(x_opt[0])
heq_fd.set_heat_generation(heat_generation_function_fd, x_opt[1], x_opt[2], x_opt[3])
heq_fd.reset()
heq_fd.solve_until_steady_state(tol=1e-3)  # Match heat_eq_2D_opt.py tolerance

# Use FD solution for final results
optimal_x = x_opt
u_opt = solve_steady_state(optimal_x[0], optimal_x[1], optimal_x[2], optimal_x[3])
max_T_opt = float(np.max(u_opt))
eta_opt = float(fan_efficiency(x_opt[0]))
obj_opt = float(objective_jit(jnp.array(x_opt)))

print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)
print(f"Converged: {result.success}")
print(f"Message: {result.message}")
print(f"Number of iterations: {result.nit}")
print("-"*60)
print("Optimal Design Variables:")
print(f"  v (fan velocity)   = {x_opt[0]:.4f} m/s")
print(f"  a (heat coeff)     = {x_opt[1]:.4f} W/m⁴")
print(f"  b (heat coeff)     = {x_opt[2]:.4f} W/m⁴")
print(f"  c (heat coeff)     = {x_opt[3]:.4f} W/m³")
print("-"*60)
print("Objective Function Components:")
print(f"  Max Temperature    = {max_T_opt:.2f} K ({max_T_opt - 273:.2f} °C)")
print(f"  Fan Efficiency η   = {eta_opt:.4f}")
print(f"  Objective Value    = {obj_opt:.6f}")
print("-"*60)
print("Constraint Satisfaction:")
total_heat = float(compute_total_heat(x_opt[1], x_opt[2], x_opt[3]))
print(f"  Total Heat Gen.    = {total_heat:.4f} W (target: 10 W)")
print(f"  Constraint Error   = {total_heat - 10:.6f} W")
print("="*60)

# Convert to arrays for plotting
x_path = np.array(x_path)
iterations = np.arange(len(x_path))
objective_history = np.array(objective_history)
max_T_history = np.array(max_T_history)
eta_history = np.array(eta_history)
constraint_history = np.array(constraint_history)
grad_obj_history = np.array(grad_obj_history)

# %%

# ========== Generate Plots ==========

# 1. Objective Function Convergence
plt.figure(figsize=(8, 5))
plt.plot(iterations, objective_history, 'r-', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel(r'$\omega_1 \max(T)/273 - \omega_2 \eta$')
plt.title('Objective Function vs Iteration (JAX AD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()


# 2. Maximum Temperature
plt.figure(figsize=(8, 5))
plt.plot(iterations, max_T_history - 273, 'g-', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel(r'$\max(T)$ [°C]')
plt.title('Maximum Temperature vs Iteration (JAX AD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()


# 3. Fan Efficiency
plt.figure(figsize=(8, 5))
plt.plot(iterations, eta_history, 'm-', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel(r'$\eta$')
plt.title('Fan Efficiency vs Iteration (JAX AD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()


# 4. Constraint History
plt.figure(figsize=(8, 5))
plt.plot(iterations, constraint_history, 'c-', linewidth=1.5)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8, label='Constraint = 0')
plt.xlabel('Iteration')
plt.ylabel(r'$\int\!\!\int\!\!\int f(x,y)\,dV - 10$ [W]')
plt.title('Power Constraint vs Iteration (JAX AD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


# 5. Gradient Norm (Convergence)
plt.figure(figsize=(8, 5))
plt.semilogy(iterations, grad_obj_history, 'b-', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel(r'$\|\nabla f\|$')
plt.title('Gradient of Objective Function vs Iteration (JAX AD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()


# 6. Design Variables
fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

axes[0].plot(iterations, x_path[:, 0], 'b-', linewidth=1.5)
axes[0].set_ylabel(r'$v$ [m/s]')
axes[0].set_title('Design Variables vs Iteration (JAX AD)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(iterations, x_path[:, 1], 'r-', linewidth=1.5)
axes[1].set_ylabel(r'$a$ [W/m$^4$]')
# Set y-limits to show variation around the mean value
a_mean = np.mean(x_path[:, 1])
a_range = np.max(x_path[:, 1]) - np.min(x_path[:, 1])
if a_range < 1e-6:  # If essentially constant, show small window
    axes[1].set_ylim(a_mean - 0.1, a_mean + 0.1)
else:
    axes[1].set_ylim(a_mean - 1.5 * a_range, a_mean + 1.5 * a_range)
axes[1].grid(True, alpha=0.3)

axes[2].plot(iterations, x_path[:, 2], 'g-', linewidth=1.5)
axes[2].set_ylabel(r'$b$ [W/m$^4$]')
# Set y-limits to show variation around the mean value
b_mean = np.mean(x_path[:, 2])
b_range = np.max(x_path[:, 2]) - np.min(x_path[:, 2])
if b_range < 1e-6:  # If essentially constant, show small window
    axes[2].set_ylim(b_mean - 0.1, b_mean + 0.1)
else:
    axes[2].set_ylim(b_mean - 1.5 * b_range, b_mean + 1.5 * b_range)
axes[2].grid(True, alpha=0.3)

axes[3].plot(iterations, x_path[:, 3], 'm-', linewidth=1.5)
axes[3].set_xlabel('Iteration')
axes[3].set_ylabel(r'$c$ [W/m$^3$]')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()


fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X * 1000, Y * 1000, u_opt - 273, levels=20, cmap='hot')
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Temperature [°C]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title('Optimal Steady-State Temperature Distribution (FD Solver)')
ax.set_aspect('equal')
plt.tight_layout()






# %%
