# %%
# Packages
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
    v, a, b, c = x 

    # v = v * 20
    # a = a * 50
    # b = b * 50
    # c = c * 10**5
    
    # Solve for steady-state temperature
    u = solve_steady_state(v, a, b, c)
    
    # Compute objective components
    max_T = jnp.max(u)
    eta = fan_efficiency(v)
    
    return W1 * max_T / 273.0 - W2 * eta


def constraint_total_heat(x):
    """Equality constraint: total heat = 10W"""
    v, a, b, c = x  # Tuple unpacking

    # v = v * 20
    # a = a * 50
    # b = b * 50
    # c = c * 10**5

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
# Functions

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


# %%
# Convergence Analysis

# Run AD vs FD comparison (default)
# Initial design point
v0 = 15.0
a0 = 10.0
b0 = 10.0
c0 = 10.0 / (CPU_X * CPU_Y * CPU_Z)  # ≈ 156250 W/m³

x0 = [v0, a0, b0, c0]

x_opt = [20.0, -50.0, -50.0, 153320.0]

xs = np.linspace(x0, x_opt, 5)

global_error = []

iters = [50, 100, 250, 500, 750, 1000, 10000, 50000]

for x in xs:
    v, a, b, c = x

    print(f"x value: {x}")

    error_list = []
    for n_iter in iters:
        u, error = solve_steady_state_with_error(v, a, b, c, num_iter=n_iter)
        error_list.append(error)
        print(f"  {n_iter:6d} iterations: error={error:.6e}, maxT={float(jnp.max(u)):.2f}K")

    global_error.append(np.array(error_list))

mean_error = np.mean(np.array(global_error), axis=0)



fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(iters, mean_error)
ax.set_xlabel('time steps')
ax.set_ylabel('mean steady-state error')
ax.grid(True, alpha=0.3)
fig.tight_layout()


# %%
# Sensitivity Analysis

def sensitivity(AD_objective_grad):
    # Run AD vs FD comparison (default)
    # Initial design point
    v0 = 15.0
    a0 = 10.0
    b0 = 10.0
    c0 = 10.0 / (CPU_X * CPU_Y * CPU_Z)  # ≈ 156250 W/m³

    x0 = [v0, a0, b0, c0]

    x_opt = [20.0, -50.0, -50.0, 153320.0]

    xs = np.linspace(x0, x_opt, 5)

    AD_jac_obj = []

    param_names = ['v (velocity)', 'a (heat coeff)', 'b (heat coeff)', 'c (heat coeff)']

    print(f"\n{'='*70}")
    print("GRADIENT COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Parameter':<20} {'AD Value':<22} {'FD Value':<22} {'Abs Error'}")
    print(f"{'-'*70}")

    for i, x in enumerate(xs):
        ad_obj = AD_objective_grad(x)
        AD_jac_obj.append(ad_obj)
        fd_obj = finite_difference_gradient(objective_function, x, 10e-3)

        print(f'x = {x}')

        for i, name in enumerate(param_names):
            ad_val = ad_obj[i]
            fd_val = fd_obj[i]
            abs_error = np.abs(ad_val - fd_val)
            print(f"{name:<20} {ad_val:<22.10e} {fd_val:<22.10e} {abs_error:<.2e}")

    print(f'Mean sensitivity during optimization: {np.mean(AD_jac_obj[:-1], axis=0)}')

print('objective sensitivity')
sensitivity(AD_objective_grad)

print('constraint sensitivity')
sensitivity(AD_constraint_grad)


# %%

fig, ax = plt.subplots(figsize=(10, 6))

for i, x0 in enumerate(xs):
    v0, a0, b0, c0 = x0
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
        ax.loglog(h_vals, errors, 'bo-', label=f'x = {i}')
        
ax.set_xlabel('Step size h')
ax.set_ylabel('Absolute error |FD - AD|')
# ax.set_title('Finite Difference Convergence to AD Gradient (parameter v)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

# %%
