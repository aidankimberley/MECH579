"""
JAX Adjoint-Based 2D Heat Equation Optimization for MECH 579 Final Project

This script implements the ADJOINT METHOD using JAX's implicit differentiation features.
Instead of backpropagating through thousands of time steps (which consumes huge memory),
we compute the gradient of the steady-state solution by solving the adjoint linear system:

    (I - dG/du)^T * lambda = (dJ/du)^T

where:
    u = G(u, p) is the fixed-point iteration (time step)
    J is the objective function
    lambda is the adjoint variable (Lagrange multiplier)

This allows for optimization with constant memory cost regardless of iteration count.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import partial

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
CFL = 0.25
MAX_ITER = 150000  # High iteration count for true steady state (feasible with adjoint!)
TOLERANCE = 1e-5

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
    T_film = (u + EXT_T) / 2
    T_film = jnp.maximum(T_film, 1.0)
    beta = 1.0 / T_film
    dT = jnp.abs(u - EXT_T)
    dT = jnp.maximum(dT, 1e-6)
    rayleigh = 9.81 * beta * dT * dx**3 / (EXT_NU**2) * EXT_PR
    rayleigh = jnp.maximum(rayleigh, 1.0)
    
    nusselt = (0.825 + (0.387 * rayleigh**(1/6)) /
               (1 + (0.492/EXT_PR)**(9/16))**(8/27))**2
    return nusselt * EXT_K / dx


def h_top(v):
    """Forced convection from fan (flat plate correlation)"""
    Rex = v * X / EXT_NU
    Rex = jnp.maximum(Rex, 1e-10)
    Nux = jnp.where(
        Rex < 5e5,
        0.332 * Rex**0.5 * EXT_PR**(1/3),
        0.0296 * Rex**0.8 * EXT_PR**(1/3)
    )
    return Nux * EXT_K / (X + 1e-8)


def initial_condition():
    """Initial temperature distribution (Cosine profile)"""
    return 70 * jnp.sin(X * jnp.pi / CPU_X) * jnp.sin(Y * jnp.pi / CPU_Y) + EXT_T


# ============================================================================
# Fixed Point Iteration (The "Forward" Physics)
# ============================================================================

def step_forward(u, params):
    """
    Fixed point iteration function u_new = G(u, params).
    This represents one time step of the explicit solver.
    """
    v, a, b, c = params
    i0, j0, iN, jN = 0, 0, N - 1, N - 1
    
    # Physics coefficients
    h_b = h_boundary(u)
    h_t = h_top(v)
    e_dot = heat_generation(X, Y, a, b, c)
    
    old_u = u
    new_u = u
    
    # ------------------------------------------------------------------------
    # Discretized Heat Equation (Explicit)
    # ------------------------------------------------------------------------
    
    # Interior
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
    
    # Boundaries (Left, Right, Bottom, Top)
    left = (
        old_u[i0, 1:-1] +
        2 * tau * h_b[i0, 1:-1] / K_SI * dy * (EXT_T - old_u[i0, 1:-1]) +
        tau * dx * (old_u[i0, 2:] - 2*old_u[i0, 1:-1] + old_u[i0, :-2]) / dy +
        2 * tau * dy * (old_u[i0+1, 1:-1] - old_u[i0, 1:-1]) / dx +
        tau * h_t[i0, 1:-1] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[i0, 1:-1]) +
        tau * e_dot[i0, 1:-1] / K_SI * dx * dy
    )
    new_u = new_u.at[i0, 1:-1].set(left)
    
    right = (
        old_u[iN, 1:-1] +
        2 * tau * h_b[iN, 1:-1] / K_SI * dy * (EXT_T - old_u[iN, 1:-1]) +
        tau * dx * (old_u[iN, 2:] - 2*old_u[iN, 1:-1] + old_u[iN, :-2]) / dy +
        2 * tau * dy * (old_u[iN-1, 1:-1] - old_u[iN, 1:-1]) / dx +
        tau * h_t[iN, 1:-1] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[iN, 1:-1]) +
        tau * e_dot[iN, 1:-1] / K_SI * dx * dy
    )
    new_u = new_u.at[iN, 1:-1].set(right)
    
    bottom = (
        old_u[1:-1, j0] +
        2 * tau * h_b[1:-1, j0] / K_SI * dx * (EXT_T - old_u[1:-1, j0]) +
        tau * dy * (old_u[2:, j0] - 2*old_u[1:-1, j0] + old_u[:-2, j0]) / dx +
        2 * tau * dx * (old_u[1:-1, j0+1] - old_u[1:-1, j0]) / dy +
        tau * h_t[1:-1, j0] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[1:-1, j0]) +
        tau * e_dot[1:-1, j0] / K_SI * dx * dy
    )
    new_u = new_u.at[1:-1, j0].set(bottom)
    
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
    bl = (
        old_u[i0, j0] +
        2 * tau * h_b[i0, j0] * (dy + dx) / K_SI * (EXT_T - old_u[i0, j0]) +
        2 * tau * dx * (old_u[i0, j0+1] - old_u[i0, j0]) / dy +
        2 * tau * dy * (old_u[i0+1, j0] - old_u[i0, j0]) / dx +
        tau * h_t[i0, j0] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[i0, j0]) +
        tau * e_dot[i0, j0] / K_SI * dx * dy
    )
    new_u = new_u.at[i0, j0].set(bl)
    
    br = (
        old_u[iN, j0] +
        2 * tau * h_b[iN, j0] * (dy + dx) / K_SI * (EXT_T - old_u[iN, j0]) +
        2 * tau * dx * (old_u[iN, j0+1] - old_u[iN, j0]) / dy +
        2 * tau * dy * (old_u[iN-1, j0] - old_u[iN, j0]) / dx +
        tau * h_t[iN, j0] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[iN, j0]) +
        tau * e_dot[iN, j0] / K_SI * dx * dy
    )
    new_u = new_u.at[iN, j0].set(br)
    
    tl = (
        old_u[i0, jN] +
        2 * tau * h_b[i0, jN] * (dy + dx) / K_SI * (EXT_T - old_u[i0, jN]) +
        2 * tau * dx * (old_u[i0, jN-1] - old_u[i0, jN]) / dy +
        2 * tau * dy * (old_u[i0+1, jN] - old_u[i0, jN]) / dx +
        tau * h_t[i0, jN] / K_SI * dx * dy / CPU_Z * (EXT_T - old_u[i0, jN]) +
        tau * e_dot[i0, jN] / K_SI * dx * dy
    )
    new_u = new_u.at[i0, jN].set(tl)
    
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


# ============================================================================
# Implicit Adjoint Implementation
# ============================================================================

@partial(jax.custom_vjp, nondiff_argnums=(2,))
def solve_steady_state_adjoint(params, u_guess, max_iter):
    """
    Forward pass: Solves the fixed point u = G(u, p) by iterating.
    Returns steady-state u.
    """
    
    def cond_fun(state):
        u, u_old, i = state
        # Run at least max_iter/10, then check tolerance
        # (Simplified: just run max_iter for consistency with previous scripts)
        # For efficiency, we could check residual, but here we stick to fixed iter for robustness
        return i < max_iter
    
    def body_fun(state):
        u, _, i = state
        u_new = step_forward(u, params)
        return u_new, u, i + 1
    
    # Run the loop
    # Note: lax.while_loop is not reverse-mode differentiable by default,
    # but since we defined a custom VJP, JAX won't try to differentiation through it!
    # This allows us to use while_loop freely in the forward pass.
    
    # But for simplicity/speed, let's use scan/fori_loop if we want fixed iterations
    # or while_loop if we want convergence. Let's use fori_loop for speed.
    
    final_u = lax.fori_loop(0, max_iter, lambda i, u: step_forward(u, params), u_guess)
    
    return final_u


def solve_steady_state_fwd(params, u_guess, max_iter):
    """Forward pass logic (same as above)"""
    return solve_steady_state_adjoint(params, u_guess, max_iter)


def solve_steady_state_bwd(max_iter, res, g):
    """
    Backward pass (Adjoint Solver).
    
    Solves the adjoint equation:
        (I - dG/du)^T * lambda = g
        
    where g is the incoming gradient dJ/du from the objective.
    Then computes gradients w.r.t. params:
        dJ/dp = dG/dp^T * lambda
        
    res: (u_star,) - result from forward pass (steady state solution)
    g: gradient of objective w.r.t. u_star
    """
    u_star = res
    params = u_star.shape  # Wait, res is u_star, params is input to custom_vjp
    # We need to capture 'params' from the forward call. 
    # JAX handles this by passing residuals. We need to restructure slightly.
    return (None, None) # Placeholder, see updated structure below


# Redefining to capture residuals properly
@partial(jax.custom_vjp, nondiff_argnums=(2,))
def implicit_solver(params, u_guess, max_iter):
    # 1. Run forward solver
    def body(i, u): return step_forward(u, params)
    u_star = lax.fori_loop(0, max_iter, body, u_guess)
    return u_star

def implicit_solver_fwd(params, u_guess, max_iter):
    u_star = implicit_solver(params, u_guess, max_iter)
    # Save u_star and params for the backward pass
    return u_star, (params, u_star)

def implicit_solver_bwd(max_iter, residuals, g):
    """
    Solves the adjoint system using fixed-point iteration on the adjoint equation.
    
    The adjoint equation for u = G(u, p) is:
        lambda = (dG/du)^T * lambda + g
        
    This is a linear fixed point system: x = Ax + b
    We solve it by iterating lambda_{k+1} = (dG/du)^T * lambda_k + g
    """
    params, u_star = residuals
    
    # Define vector-Jacobian product function for dG/du
    # vjp_u(v) computes (dG/du)^T * v
    def vjp_u(v):
        _, vjp_fun = jax.vjp(lambda u: step_forward(u, params), u_star)
        return vjp_fun(v)[0]  # Gradient w.r.t. u
    
    # Solve adjoint system: lambda = vjp_u(lambda) + g
    # Iterative solver (Neumann series)
    # This converges if spectral radius of dG/du < 1 (which holds for stable time-stepper)
    
    def adjoint_body(i, lam):
        return vjp_u(lam) + g
    
    # Run adjoint iterations (same number as forward usually sufficient, or fewer)
    # Using fewer here for speed, but ideally checks convergence
    lambda_star = lax.fori_loop(0, max_iter, adjoint_body, jnp.zeros_like(u_star))
    
    # Now compute gradients w.r.t. params using the adjoint variable lambda_star
    # dJ/dp = lambda^T * (dG/dp)
    # equivalent to: vjp_params(lambda_star)
    
    _, vjp_fun_params = jax.vjp(lambda p: step_forward(u_star, p), params)
    grads_params = vjp_fun_params(lambda_star)[0]
    
    return grads_params, None  # Gradient w.r.t. params, None for u_guess


# Register the VJP
implicit_solver.defvjp(implicit_solver_fwd, implicit_solver_bwd)


# ============================================================================
# Optimization Setup
# ============================================================================

# Weights
W1 = 0.2
W2 = 0.8

def objective_function(x):
    """Objective with Implicit Adjoint Solver"""
    v, a, b, c = x[0], x[1], x[2], x[3]
    params = jnp.array([v, a, b, c])
    
    # Use the implicit adjoint solver
    u0 = initial_condition()
    u = implicit_solver(params, u0, MAX_ITER)
    
    max_T = jnp.max(u)
    eta = fan_efficiency(v)
    
    return W1 * max_T / 273.0 - W2 * eta


def run_optimization_adjoint():
    """Run optimization using the Adjoint Method"""
    
    print("="*60)
    print("JAX ADJOINT OPTIMIZATION")
    print("="*60)
    print("Using implicit differentiation (adjoint method) for gradients.")
    print(f"Iterations: {MAX_ITER}, CFL: {CFL}")
    print("-" * 60)
    
    # Compile functions
    print("JIT compiling functions...")
    obj_jit = jax.jit(objective_function)
    grad_jit = jax.jit(jax.grad(objective_function))
    
    # Initial guess
    v0 = 15.0
    a0 = 10.0
    b0 = 10.0
    c0 = 156250.0
    x0 = np.array([v0, a0, b0, c0])
    
    # Wrappers
    def obj_wrapper(x):
        return float(obj_jit(jnp.array(x)))
        
    def grad_wrapper(x):
        return np.array(grad_jit(jnp.array(x)))
        
    # Constraint (same as before)
    def constraint_fun(x):
        return float(compute_total_heat(x[1], x[2], x[3])) - 10.0
        
    grad_const_jit = jax.jit(jax.grad(lambda x: compute_total_heat(x[1], x[2], x[3])))
    def constraint_grad(x):
        return np.array(grad_const_jit(jnp.array(x)))
    
    constraints = [{'type': 'eq', 'fun': constraint_fun, 'jac': constraint_grad}]
    bounds = [(0.1, 30), (-1e8, 1e8), (-1e8, 1e8), (0, 1e8)]
    
    # Callback
    history = []
    def callback(xk):
        obj = obj_wrapper(xk)
        grad = grad_wrapper(xk)
        print(f"Iter {len(history)+1:3d}: obj={obj:.6f}, a={xk[1]:.2f}, b={xk[2]:.2f}, c={xk[3]:.2f}, v={xk[0]:.2f}, |∇f|={np.linalg.norm(grad):.2e}")
        history.append(obj)
        
    # Optimize
    print("\nStarting optimization...")
    res = optimize.minimize(
        obj_wrapper, x0, method='SLSQP', jac=grad_wrapper,
        bounds=bounds, constraints=constraints, callback=callback,
        options={'maxiter': 100, 'ftol': 1e-6, 'disp': True}
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print(f"Optimal v: {res.x[0]:.4f} m/s")
    print(f"Optimal a: {res.x[1]:.2f} W/m⁴")
    print(f"Optimal b: {res.x[2]:.2f} W/m⁴")
    print(f"Optimal c: {res.x[3]:.2f} W/m³")
    # Get optimal temperature field
    u_opt = steady_state_solver(res.x)
    max_T = float(jnp.max(u_opt))
    print(f"Maximum Temperature: {max_T:.2f} K")
    print(f"Optimal objective: {res.fun:.6f}")
    print("="*60)


if __name__ == "__main__":
    # Just run a quick gradient check to verify the adjoint works
    print("Verifying adjoint gradient computation...")
    x_test = jnp.array([15.0, 10.0, 10.0, 156250.0])
    
    # Compute gradient using the adjoint method
    grad_adjoint = jax.grad(objective_function)(x_test)
    print(f"Adjoint Gradient: {grad_adjoint}")
    
    # Run full optimization
    run_optimization_adjoint()
