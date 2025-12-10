"""
JAX-differentiable 2D Heat Equation Solver for MECH 579 Final Project

This version is fully functional (no class mutation) so JAX can trace through
and compute gradients via automatic differentiation.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

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
CFL = 0.25  # Slightly increased for faster evolution
MAX_ITER = 500  # Increased to better match FD convergence (FD runs until tol=1e-3)  

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
    """Compute total heat generation using trapezoidal rule"""
    q = heat_generation(X, Y, a, b, c) * dx * dy * CPU_Z
    i0, iN, j0, jN = 0, N - 1, 0, N - 1
    
    # Trapezoidal weights
    weights = jnp.ones_like(q)
    weights = weights.at[i0, :].multiply(0.5)
    weights = weights.at[iN, :].multiply(0.5)
    weights = weights.at[:, j0].multiply(0.5)
    weights = weights.at[:, jN].multiply(0.5)
    
    return jnp.sum(q * weights)


def h_boundary(u):
    """Natural convection heat transfer coefficient (Churchill-Chu)"""
    # Use film temperature for properties
    T_film = (u + EXT_T) / 2
    T_film = jnp.maximum(T_film, 1.0)  # Safe lower bound
    beta = 1.0 / T_film
    
    # Temperature difference (use abs to handle both heating and cooling)
    dT = jnp.abs(u - EXT_T)
    dT = jnp.maximum(dT, 1e-6)  # Avoid zero
    
    rayleigh = 9.81 * beta * dT * dx**3 / (EXT_NU**2) * EXT_PR
    rayleigh = jnp.maximum(rayleigh, 1.0)  # Minimum Rayleigh for stability
    
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

def step_forward(u, v, a, b, c):
    """Single time step update - pure function, returns new u"""
    i0, j0, iN, jN = 0, 0, N - 1, N - 1
    
    # Compute heat transfer coefficients
    h_b = h_boundary(u)
    h_t = h_top(v)
    e_dot = heat_generation(X, Y, a, b, c)
    
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
        tau * dx * (old_u[i0, 2:] - 2*old_u[i0, 1:-1] + old_u[i0, :-2]) / dy +
        2 * tau * dy * (old_u[i0+1, 1:-1] - old_u[i0, 1:-1]) / dx +
        tau * h_t[i0, 1:-1] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[i0, 1:-1]) +
        tau * e_dot[i0, 1:-1] / K_SI * dx * dy
    )
    new_u = new_u.at[i0, 1:-1].set(left)
    
    # Right boundary (i=N-1)
    right = (
        old_u[iN, 1:-1] +
        2 * tau * h_b[iN, 1:-1] / K_SI * dy * (EXT_T - old_u[iN, 1:-1]) +
        tau * dx * (old_u[iN, 2:] - 2*old_u[iN, 1:-1] + old_u[iN, :-2]) / dy +
        2 * tau * dy * (old_u[iN-1, 1:-1] - old_u[iN, 1:-1]) / dx +
        tau * h_t[iN, 1:-1] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[iN, 1:-1]) +
        tau * e_dot[iN, 1:-1] / K_SI * dx * dy
    )
    new_u = new_u.at[iN, 1:-1].set(right)
    
    # Bottom boundary (j=0)
    bottom = (
        old_u[1:-1, j0] +
        2 * tau * h_b[1:-1, j0] / K_SI * dx * (EXT_T - old_u[1:-1, j0]) +
        tau * dy * (old_u[2:, j0] - 2*old_u[1:-1, j0] + old_u[:-2, j0]) / dx +
        2 * tau * dx * (old_u[1:-1, j0+1] - old_u[1:-1, j0]) / dy +
        tau * h_t[1:-1, j0] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[1:-1, j0]) +
        tau * e_dot[1:-1, j0] / K_SI * dx * dy
    )
    new_u = new_u.at[1:-1, j0].set(bottom)
    
    # Top boundary (j=N-1)
    top = (
        old_u[1:-1, jN] +
        2 * tau * h_b[1:-1, jN] / K_SI * dx * (EXT_T - old_u[1:-1, jN]) +
        tau * dy * (old_u[2:, jN] - 2*old_u[1:-1, jN] + old_u[:-2, jN]) / dx +
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


def solve_steady_state(v, a, b, c, num_iter=MAX_ITER):
    """
    Solve to steady state using jax.lax.fori_loop (JAX-differentiable).
    
    Uses fixed iteration count so JAX can differentiate through it.
    Returns the steady-state temperature field.
    """
    u0 = initial_condition()
    
    def body_fun(i, u):
        return step_forward(u, v, a, b, c)
    
    final_u = lax.fori_loop(0, num_iter, body_fun, u0)
    
    return final_u


def solve_steady_state_with_error(v, a, b, c, num_iter=MAX_ITER):
    """
    Solve to steady state and return both the field and the convergence error.
    
    Returns (u_final, error) where error = ||u_new - u_old||_inf
    """
    u0 = initial_condition()
    u_prev = u0
    
    def body_fun(i, u):
        return step_forward(u, v, a, b, c)
    
    final_u = lax.fori_loop(0, num_iter, body_fun, u0)
    
    # Compute error by doing one more step
    u_one_more = step_forward(final_u, v, a, b, c)
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
    v, a, b, c = x[0], x[1], x[2], x[3]
    
    # Solve for steady-state temperature
    u = solve_steady_state(v, a, b, c)
    
    # Compute objective components
    max_T = jnp.max(u)
    eta = fan_efficiency(v)
    
    return W1 * max_T / 273.0 - W2 * eta


def constraint_total_heat(x):
    """Equality constraint: total heat = 10W"""
    a, b, c = x[1], x[2], x[3]
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

def run_optimization():
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
        
        # Print progress
        print(f"  Iter {len(x_path):3d}: obj={obj_val:.6f}, maxT={max_T-273:.2f}°C, "
              f"η={eta:.4f}, |∇f|={np.linalg.norm(grad_obj):.2e}, |∇L|={grad_L_norm:.2e}, "
              f"vars: v={xk[0]:.2f}, a={xk[1]:.2f}, b={xk[2]:.2f}, c={xk[3]:.2f}")
        if len(x_path) <= 2:  # Print gradient breakdown for first few iterations
            print(f"      Gradients: ∇f_v={grad_obj[0]:.6e}, ∇f_a={grad_obj[1]:.6e}, "
                  f"∇f_b={grad_obj[2]:.6e}, ∇f_c={grad_obj[3]:.6e}")
        
        return False  # Don't stop optimization
    
    # Initial guess (same as heat_eq_2D_opt.py)
    v0 = 15.0
    a0 = -30.0
    b0 = -30.0
    c0 = 156250.0  # ≈ NOT 10W total, just for testing
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
    
    # Final solution
    x_opt = result.x
    u_opt = solve_steady_state(x_opt[0], x_opt[1], x_opt[2], x_opt[3])
    max_T_opt = float(jnp.max(u_opt))
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
    
    # ========== Generate Plots ==========
    
    # 1. Objective Function Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, objective_history, 'r-', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\omega_1 \max(T)/273 - \omega_2 \eta$')
    plt.title('Objective Function vs Iteration (JAX AD)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/objective_function_AD.png', dpi=150)
    plt.close()
    
    # 2. Maximum Temperature
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, max_T_history - 273, 'g-', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\max(T)$ [°C]')
    plt.title('Maximum Temperature vs Iteration (JAX AD)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/max_temperature_AD.png', dpi=150)
    plt.close()
    
    # 3. Fan Efficiency
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, eta_history, 'm-', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\eta$')
    plt.title('Fan Efficiency vs Iteration (JAX AD)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/fan_efficiency_AD.png', dpi=150)
    plt.close()
    
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
    plt.savefig('plots/power_constraint_AD.png', dpi=150)
    plt.close()
    
    # 5. Gradient Norm (Convergence)
    plt.figure(figsize=(8, 5))
    plt.semilogy(iterations, grad_obj_history, 'b-', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|\nabla f\|$')
    plt.title('Gradient of Objective Function vs Iteration (JAX AD)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/gradient_objective_AD.png', dpi=150)
    plt.close()
    
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
    plt.savefig('plots/design_vars_AD.png', dpi=150)
    plt.close()
    
    # 7. Optimal Temperature Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X * 1000, Y * 1000, u_opt - 273, levels=20, cmap='hot')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Temperature [°C]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('Optimal Steady-State Temperature Distribution (JAX AD)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('plots/optimal_temperature_AD.png', dpi=150)
    plt.close()
    
    print("\nAll plots saved to plots/ directory.")
    
    return result, x_path, objective_history


# ============================================================================
# Convergence Comparison
# ============================================================================

def compare_convergence(v, a, b, c):
    """
    Compare steady-state convergence between JAX (fixed iterations) and 
    estimate what error the JAX solver achieves.
    
    Returns convergence metrics for the JAX solver.
    """
    print("\n" + "="*60)
    print("STEADY-STATE CONVERGENCE COMPARISON")
    print("="*60)
    
    # JAX version with error computation
    print(f"\nJAX Solver (MAX_ITER={MAX_ITER}, CFL={CFL}):")
    u_jax, error_jax = solve_steady_state_with_error(v, a, b, c, num_iter=MAX_ITER)
    max_T_jax = float(jnp.max(u_jax))
    
    print(f"  Final max temperature: {max_T_jax:.2f} K ({max_T_jax-273:.2f} °C)")
    print(f"  Convergence error (||u_new - u_old||_inf): {error_jax:.6e}")
    
    # Check convergence at different iteration counts
    print(f"\nConvergence at different iteration counts:")
    for n_iter in [5000, 10000, 20000, 50000, MAX_ITER]:
        if n_iter <= MAX_ITER:
            u_test, err_test = solve_steady_state_with_error(v, a, b, c, num_iter=n_iter)
            max_T_test = float(jnp.max(u_test))
            print(f"  {n_iter:6d} iterations: error={err_test:.6e}, maxT={max_T_test:.2f}K")
    
    # Compare with FD target tolerance
    fd_tolerance = 1e-3
    print(f"\nFD Version target tolerance: {fd_tolerance:.0e}")
    print(f"JAX Version final error:      {error_jax:.6e}")
    
    if error_jax < fd_tolerance:
        print(f"✓ JAX solver is MORE converged than FD target")
    elif error_jax < 10 * fd_tolerance:
        print(f"≈ JAX solver is SIMILARLY converged to FD target")
    else:
        print(f"✗ JAX solver is LESS converged than FD target")
        print(f"  Consider increasing MAX_ITER or adjusting CFL")
    
    return u_jax, error_jax


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check command line args
    if len(sys.argv) > 1 and sys.argv[1] == '--optimize':
        # Run full optimization
        run_optimization()
    elif len(sys.argv) > 1 and sys.argv[1] == '--compare-convergence':
        # Compare convergence with FD version
        v0, a0, b0, c0 = 15.0, 10.0, 10.0, 156250.0
        compare_convergence(v0, a0, b0, c0)
    else:
        # Run AD vs FD comparison (default)
        # Initial design point
        v0 = 15.0
        a0 = 10.0
        b0 = 10.0
        c0 = 10.0 / (CPU_X * CPU_Y * CPU_Z)  # ≈ 156250 W/m³
        
        x0 = [v0, a0, b0, c0]
        
        print("="*60)
        print("JAX Automatic Differentiation Test")
        print("="*60)
        print(f"Design point: v={v0}, a={a0}, b={b0}, c={c0:.2f}")
        print(f"Total heat at x0: {float(compute_total_heat(a0, b0, c0)):.4f} W")
        
        # First, test that the solver works
        print("\nSolving heat equation to steady state...")
        u_ss = solve_steady_state(v0, a0, b0, c0)
        print(f"Max temperature: {float(jnp.max(u_ss)):.2f} K ({float(jnp.max(u_ss))-273:.2f} °C)")
        print(f"Objective value: {float(objective_function(jnp.array(x0))):.6f}")
        
        # Detailed comparison for one parameter (v)
        ad_v, fd_v, h_vals, fd_vals = compare_gradients(x0, param_idx=0, param_name="v (velocity)")
        
        # Full gradient comparison
        ad_full, fd_full = full_gradient_comparison(x0)
        
        # Plot FD convergence
        import os
        os.makedirs('plots', exist_ok=True)
        
        errors = [abs(fd - ad_v) for fd in fd_vals]
        if any(e > 0 for e in errors):
            plt.figure(figsize=(10, 6))
            plt.loglog(h_vals, errors, 'bo-', label='|FD - AD|')
            plt.xlabel('Step size h')
            plt.ylabel('Absolute error |FD - AD|')
            plt.title('Finite Difference Convergence to AD Gradient (parameter v)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('plots/fd_convergence.png', dpi=150)
            plt.close()
        
        print("\n" + "="*60)
        print("Analysis complete. Plot saved to plots/fd_convergence.png")
        print("="*60)
        print("\nTo run optimization, use: python jax_ad_heat.py --optimize")

