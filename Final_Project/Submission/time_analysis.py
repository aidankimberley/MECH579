# %%
# Imports
import numpy as np # numpy for vectorization
from collections.abc import Callable # For type hints
import matplotlib.pyplot as plt
from scipy import optimize
import argparse
from functools import partial
# Import FD solver for final solution verification
from heat_eq_2D_opt import HeatEquation2D
import jax
import jax.numpy as jnp
from jax import lax
# Enable float64 for better numerical precision
jax.config.update("jax_enable_x64", True)
import time


# %% 

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

    v = v * 20
    a = a * 50
    b = b * 50
    c = c * 10**5
    
    # Solve for steady-state temperature
    u = solve_steady_state(v, a, b, c)
    
    # Compute objective components
    max_T = jnp.max(u)
    eta = fan_efficiency(v)
    
    return W1 * max_T / 273.0 - W2 * eta


def constraint_total_heat(x):
    """Equality constraint: total heat = 10W"""
    v, a, b, c = x  # Tuple unpacking

    v = v * 20
    a = a * 50
    b = b * 50
    c = c * 10**5

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

OPTIMIZATION_METHOD = 'trust-constr'


class HeatEquation2D:
    """Heat Equation Solver for MECH 579 Final Project

    This class will construct and solve the unsteady heat equation
    with Robin BCs as described in the assignment.
    """
    def __init__(self, x:float, y:float, height:float , n_x:int, n_y:int,
                     k:float=1.0, rho:float=1.0, cp:float=1.0,
                     CFL:float=0.1, init_condition:Callable[[np.ndarray,np.ndarray], np.ndarray] = lambda x,y: np.sin(x+y)):
        """Intializition function for the heat equation

        Parameters

        ------

        x (float): Physical Size of CPU in x-direction [m]

        y (float): Physical Size of CPU in y-direction [m]

        n_x (int): Number of grid points in x-direction [m]

        n_y (int): Number of grid points in y-direction [m]

        k (float): The heat transfer coefficient of the CPU [W/[mK]]

        rho (float): Constant density of CPU [kg/m^3]

        cp (float): Specific heat capacity of CPU [kJ/[kgK]]

        CFL (float): Courant-Friedrichs-Lewy Number

        init_condition (function(x,y)): Initial condition of the CPU
        """
        ## MESHING variables
        self.n_x = n_x
        self.n_y = n_y
        self.boundary_conditions = []
        # Physical locations
        x_axis = np.linspace(0, x, self.n_x)
        y_axis = np.linspace(0, y, self.n_y)
        self.X, self.Y = np.meshgrid(x_axis, y_axis, indexing='ij')
        self.dx = x_axis[1] - x_axis[0]
        self.dy = y_axis[1] - y_axis[0]
        # Variables of Mesh size
        self.u = np.zeros((self.n_x, self.n_y))
        self.h_top_values = np.zeros((self.n_x, self.n_y))
        self.h_boundary_values = np.zeros((self.n_x, self.n_y))

        ## Heat Generation Properties
        self.heat_generation_function = lambda x, y, a, b, c: a * x + b * y + c  # Can be changed
        self.heat_gen_a = 0
        self.heat_gen_b = 0
        self.heat_gen_c = 0
        self.heat_generation_total = 0

        ## Material Properties
        self.k = k
        self.rho = rho
        self.cp = cp
        self.thermal_alpha = self.k / (self.rho * self.cp)
        self.height = height #m

        ## Temporal Properties
        self.CFL = CFL
        self.dt = self.CFL * (self.dx * self.dy) / self.thermal_alpha
        self.current_time = 0
        self.steady_state_error = 1E2 # Large inital number to ensure that the problem will continue
        self.max_iter = 5E4
        self.init_condition = init_condition
        self.apply_initial_conditions()

        ## External Variables of Air
        self.ext_k = 0.02772  # W/m/K Thermal Coeffcient
        self.ext_Pr = 0.7215  # Prantl Number
        self.ext_nu = 1.506 * 10 ** (-5)  # m^2/s Kinematic Viscosity
        self.ext_T = 273 + 20  # K Temperature

        ## Fan Variables
        self.v = 10 # m/s Air Velocity
        self.fan_efficiency_func = lambda v: -0.002*v**2 + 0.08*v
        self.fan_efficiency = self.fan_efficiency_func(self.v)

        self.verbose = False

    def set_initial_conditions(self,initial_conditions:Callable[[np.ndarray,np.ndarray],np.ndarray]):
        """Sets the initial condition

        Parameters

        ------

        initial_conditions(function(x,y)): Initial condition of the CPU
        """
        self.init_condition = initial_conditions

    def apply_initial_conditions(self):
        """Applies the initial condition into self.u"""
        self.u = self.init_condition(self.X,self.Y)

    def reset(self):
        """Resets the heat equation"""
        self.apply_initial_conditions()
        self.current_time = 0
        self.steady_state_error = 1E2

    def set_heat_generation(self, heat_generation_function: Callable[[np.ndarray,np.ndarray,float,float,float], np.ndarray],
                            a: float, b: float, c: float):
        """Sets the heat generation function and associated variables

        Parameters

        ------

        heat_generation_function (function(x,y,a,b,c)): Function that dictates the heat generation by the CPU

        integrated_total (float): Total integrated value

        a, b, c (float): Variables associated with the heat generation function
        """
        self.heat_generation_function = heat_generation_function
        self.heat_gen_a = a
        self.heat_gen_b = b
        self.heat_gen_c = c
        heat_generation_matrix = self.heat_generation_function(self.X,self.Y,self.heat_gen_a,self.heat_gen_b,self.heat_gen_c) * self.dx * self.dy *self.height
        i0, iN, j0 ,jN = 0, self.n_x - 1, 0 , self.n_y - 1
        # Boundaries with one side
        heat_generation_matrix[i0,:] /= 2
        heat_generation_matrix[iN,:] /= 2
        heat_generation_matrix[j0,:] /= 2
        heat_generation_matrix[jN,:] /= 2
        # Boundaries with two sides
        heat_generation_matrix[i0,j0] /= 2
        heat_generation_matrix[iN,jN] /= 2
        heat_generation_matrix[iN,j0] /= 2
        heat_generation_matrix[i0,jN] /= 2
        self.heat_generation_total = np.sum(np.sum(heat_generation_matrix))

    def set_fan_velocity(self, v: float):
        """Sets the fan velocity

        Parameters

        ------

        v (float): Variable associated with the fan velocity
        """
        self.v = v
        self.fan_efficiency = self.fan_efficiency_func(self.v)


    def h_boundary(self,u: np.ndarray):
        """Calculates the convective heat transfer coefficient at the boundaries

        Parameters

        ------

        u (np.ndarray): Current Temperature Mesh
        """
        beta = 1/((u+self.ext_T)/2)
        rayleigh = 9.81*beta*(u-self.ext_T)*self.dx**3/(self.ext_nu**2)*self.ext_Pr
        nusselt = (0.825 + (0.387*rayleigh**(1/6))/
                   (1+(0.492/self.ext_Pr)**(9/16))**(8/27))**2
        return nusselt*self.ext_k/self.dx

    def h_top(self,x: np.ndarray,u):
        """Calculates the convective heat transfer coefficient from the fan velocity

        Parameters

        ------

        x (np.ndarray): x position

        u (np.ndarray): UNUSED
        """
        Rex = self.v*x/self.ext_nu
        r,c = Rex.shape
        Nux = np.zeros((r,c))
        for i in range(r):
            for j in range(c):
                if Rex[i,j] < 5E5:
                    Nux[i,j] = 0.332*Rex[i,j]**0.5*self.ext_Pr**(1/3)
                else:
                    Nux[i,j] = 0.0296*Rex[i,j]**0.8*self.ext_Pr**(1/3)
        h = Nux*self.ext_k/(x + 1E-5)
        return h

    def calculate_h(self):
        """Calculates all necessary convective heat transfer coefficients"""
        self.h_top_values = self.h_top(self.X,self.u)
        self.h_boundary_values = self.h_boundary(self.u)

    def apply_boundary_conditions(self, old_u):
        """Calculates the change in temperature at the boundary.

        Parameters

        -----

        old_u (np.ndarray): Current Temperature Mesh
        """
        e_dot = self.heat_generation_function(self.X, self.Y, self.heat_gen_a, self.heat_gen_b, self.heat_gen_c)
        tau = self.thermal_alpha * self.dt / (self.dx*self.dy)
        i0,j0,iN,jN = 0, 0, self.n_x-1, self.n_y-1
        # Left
        self.u[i0,1:-1] = (old_u[i0,1:-1] +
                            2 * tau * self.h_boundary_values[i0,1:-1]/self.k * self.dy * (self.ext_T - old_u[i0,1:-1]) +
                            tau * self.dx * (old_u[i0,2:] - old_u[i0,1:-1]) / self.dy +
                            tau * self.dx * (old_u[i0,1:-1] - old_u[i0,2:]) / self.dy +
                            2 * tau * self.dy * (old_u[i0 + 1, 1:-1] - old_u[i0, 1:-1]) / self.dx +
                            tau * self.h_top_values[i0,1:-1]/self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[i0,1:-1]) +
                            tau * e_dot[i0,1:-1] / self.k * self.dx * self.dy)

        # Right
        self.u[iN, 1:-1] = (old_u[iN, 1:-1] +
                            2 * tau * self.h_boundary_values[iN, 1:-1] / self.k * self.dy * (self.ext_T - old_u[iN, 1:-1]) +
                            tau * self.dx * (old_u[iN, 2:] - old_u[iN, 1:-1]) / self.dy +
                            tau * self.dx * (old_u[iN, 1:-1] - old_u[iN, 2:]) / self.dy +
                            2 * tau * self.dy * (old_u[iN- 1, 1:-1] - old_u[iN,1:-1]) / self.dx +
                            tau * self.h_top_values[iN, 1:-1] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN, 1:-1]) +
                            tau * e_dot[iN, 1:-1] / self.k * self.dx * self.dy)

        # Bottom
        self.u[1:-1,j0] = (old_u[1:-1,j0] +
                            2 * tau * self.h_boundary_values[1:-1,j0] / self.k * self.dx * (self.ext_T - old_u[1:-1,j0]) +
                            tau * self.dy * (old_u[2:,j0] - old_u[1:-1,j0]) / self.dx +
                            tau * self.dy * (old_u[1:-1,j0] - old_u[2:,j0]) / self.dx +
                            2 * tau * self.dx * (old_u[1:-1,j0 + 1] - old_u[1:-1,j0]) / self.dy +
                            tau * self.h_top_values[1:-1,j0] / self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[1:-1,j0]) +
                            tau * e_dot[1:-1,j0] / self.k * self.dx * self.dy)

        # Top
        self.u[1:-1,jN] = (old_u[1:-1,jN] +
                            2 * tau * self.h_boundary_values[1:-1,jN] / self.k * self.dx * (self.ext_T - old_u[1:-1,jN]) +
                            tau * self.dy * (old_u[2:,jN] - old_u[1:-1,jN]) / self.dx +
                            tau * self.dy * (old_u[1:-1,jN] - old_u[2:,jN]) / self.dx +
                            2 * tau * self.dx * (old_u[1:-1,jN - 1] - old_u[1:-1,jN]) / self.dy +
                            tau * self.h_top_values[1:-1,jN] / self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[1:-1,jN]) +
                            tau * e_dot[1:-1, jN] / self.k * self.dx * self.dy)

        ## Bottom Left Corner
        self.u[i0,j0] = (old_u[i0,j0] +
                         2 * tau * self.h_boundary_values[i0,j0] * self.dy / self.k * (self.ext_T - old_u[i0,j0]) +
                         2 * tau * self.h_boundary_values[i0,j0] * self.dx / self.k * (self.ext_T - old_u[i0,j0]) +
                         2 * tau * self.dx * (old_u[i0,j0+1] - old_u[i0,j0]) / self.dy +
                         2 * tau * self.dy * (old_u[i0+1,j0] - old_u[i0,j0]) / self.dx +
                         tau * self.h_top_values[i0,j0] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[i0,j0]) +
                         tau * e_dot[i0,j0] / self.k * self.dx * self.dy)
        ## Bottom Right Corner
        self.u[iN,j0] = (old_u[iN,j0] +
                         2 * tau * self.h_boundary_values[iN,j0] * self.dy / self.k * (self.ext_T - old_u[iN,j0]) +
                         2 * tau * self.h_boundary_values[iN,j0] * self.dx / self.k * (self.ext_T - old_u[iN,j0]) +
                         2 * tau * self.dx * (old_u[iN,j0+1] - old_u[iN,j0]) / self.dy +
                         2 * tau * self.dy * (old_u[iN-1,j0] - old_u[iN,j0]) / self.dx +
                         tau * self.h_top_values[iN,j0] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN,j0]) +
                         tau * e_dot[iN,j0] / self.k * self.dx * self.dy)
        ## Top Left Corner
        self.u[i0,jN] = (old_u[i0,jN] +
                         2 * tau * self.h_boundary_values[i0,jN] * self.dy / self.k * (self.ext_T - old_u[i0,jN]) +
                         2 * tau * self.h_boundary_values[i0,jN] * self.dx / self.k * (self.ext_T - old_u[i0,jN]) +
                         2 * tau * self.dx * (old_u[i0,jN-1] - old_u[i0,jN]) / self.dy +
                         2 * tau * self.dy * (old_u[i0+1,jN] - old_u[i0,jN]) / self.dx +
                         tau * self.h_top_values[i0,jN] / self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[i0,jN]) +
                         tau * e_dot[i0,jN] / self.k * self.dx * self.dy)
        ## Top Right Corner
        self.u[iN,jN] = (old_u[iN,jN] +
                         2 * tau * self.h_boundary_values[iN,jN] * self.dy / self.k * (self.ext_T - old_u[iN,jN]) +
                         2 * tau * self.h_boundary_values[iN,jN] * self.dx / self.k * (self.ext_T - old_u[iN,jN]) +
                         2 * tau * self.dx * (old_u[iN,jN-1] - old_u[iN,jN]) / self.dy +
                         2 * tau * self.dy * (old_u[iN-1,jN] - old_u[iN,jN]) / self.dx +
                         tau * self.h_top_values[iN,jN] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN,jN]) +
                         tau * e_dot[iN,jN] / self.k * self.dx * self.dy)
        return

    def step_forward_in_time(self):
        """Steps forward in time 1 timestep"""
        self.calculate_h()
        old_u = self.u.copy()
        self.apply_boundary_conditions(old_u)
        tau = self.thermal_alpha * self.dt / (self.dx * self.dy)
        self.u[1:-1, 1:-1] = (old_u[1:-1, 1:-1] +
                                    tau * (
                                            self.dy * (old_u[2:, 1:-1] - 2 * old_u[1:-1, 1:-1] + old_u[0:-2, 1:-1]) / self.dx  +
                                            self.dx * (old_u[1:-1, 2:] - 2 * old_u[1:-1, 1:-1] + old_u[1:-1, 0:-2]) / self.dy
                                    ) + tau * (self.h_top_values[1:-1, 1:-1] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[1:-1, 1:-1]) +
                                    self.dx * self.dy / self.k * self.heat_generation_function(self.X[1:-1, 1:-1],self.Y[1:-1, 1:-1],self.heat_gen_a,self.heat_gen_b,self.heat_gen_c)))
        self.steady_state_error = np.linalg.norm(self.u - old_u,np.inf)
        self.current_time += self.dt

    def solve_until_steady_state(self, tol: float = 1e-3):
        """Solves until steady state is reached

        Parameters

        ------

        tol (float, optional): Tolerance until steady state
        """
        iter = 0
        self.step_forward_in_time()
        while self.steady_state_error > tol and iter < self.max_iter:
            self.step_forward_in_time()
            iter += 1
            if (iter % 1000) == 0 and self.verbose:
                print(f"Iteration: {iter}, Error: {self.steady_state_error}")


    def solve_until_time(self,final_time: float):
        """Solves until time is reached

        Parameters

        ------

        final_time (float): Final time of simulation
        """
        iter = 0
        while self.current_time < final_time:
            self.step_forward_in_time()
            iter += 1
            if (iter % 1000) == 0 and self.verbose:
                print(f"Iteration: {iter}, Time: {self.current_time}")

# %% 
# AD Optimization

x_path = []
norm_grad_l_AD = []
norm_grad_f = []
objective_history = []
max_T_history = []
eta_history = []
constraint_history = []
grad_obj_history = []
    

t0 = time.perf_counter()
def callback(intermediate_result):
    """Callback to track optimization progress (trust-constr signature)"""
    xk = intermediate_result.x
    x_jax = jnp.array(xk)

    grad_obj = objective_grad_jax(xk)

    v, a, b, c = xk
    xk = np.array([v*20, a*50, b*50, c*10**5])
    
    # Extract Lagrangian gradient norm
    grad_L = intermediate_result.lagrangian_grad
    grad_L_norm = np.linalg.norm(grad_L) if grad_L is not None else 0.0
    norm_grad_l_AD.append(grad_L_norm)
    
    # Compute current values
    obj_val = float(objective_function(x_jax))
    u = solve_steady_state(xk[0], xk[1], xk[2], xk[3])
    max_T = float(jnp.max(u))
    eta = float(fan_efficiency(xk[0]))
    constraint_val = float(compute_total_heat(xk[1], xk[2], xk[3])) - 10.0
    
    
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
    
    t_AD.append(time.perf_counter() - t0)

    return False  # Don't stop optimization

## Bounds for inputs
# v: fan velocity [0, 30] m/s (physical limits)
# a, b: heat distribution coefficients (unconstrained)
# c: base heat generation (can be negative if a,b contribute enough)
bounds = [
    (0.1, 1.5),        # v: fan velocity (avoid 0 for numerical stability)
    (-1e8, 1e8),      # a
    (-1e8, 1e8),      # b
    (0, 1e8),      # c
]

## Setting the constraints
constraints = [
    {'type': 'eq', 'fun': constraint_total_heat, 'jac': AD_constraint_grad},
]
## Creating the initial guess

v0 = 0.5  # m/s - start at reasonable fan velocity
a0 = 0.1   # Start with uniform heat generation
b0 = 0.1
c0 = 1.0  # This gives approximately 10W total
x0 = [v0, a0, b0, c0]


t_AD = [0]
print(f"Initial guess: v={v0} m/s, a={a0}, b={b0}, c={c0}")
print(f"Starting optimization with w1={W1}, w2={W2}...")
print("-" * 50)
## Optimize
optimization_result_AD = optimize.minimize(
    objective_function,
    method='trust-constr',
    x0=x0,
    bounds=bounds,
    jac=AD_objective_grad,
    hess='2-point',
    constraints=constraints,
    callback=callback,
    options={'maxiter': 50, 'verbose': 3}
)
t_AD_tot = time.perf_counter() - t0

# %%
# FD optimization

# Physical Dimensions
cpu_x = 0.04  # m
cpu_y = 0.04  # m
cpu_z = 0.04  # m
N = 25

# Temporal Parameters
CFL = 0.5
# Silicon Constants
k_si = 149
rho_si = 2323
c_si = 19.789 / 28.085 * 1000  # J/(kgK)


# Tracking lists for convergence plots
x_path = []
norm_grad_l_FD = []
norm_grad_f = []
objective_history = []
max_T_history = []
eta_history = []
constraint_history = []
    

def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r, c = x.shape
    u = np.zeros([r, c])
    ## Cosine Case
    u = 70 * np.sin(x * np.pi / cpu_x) * np.sin(y * np.pi / cpu_y) + 293
    return u


def heat_generation_function(x: np.ndarray, y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x + b * y + c
## Problem Set up
heq = HeatEquation2D(cpu_x,cpu_y,cpu_z, N,N,
                    k=k_si,rho=rho_si,cp=c_si,
                    init_condition=initial_condition)
# Test values for a,b,c
test_a = 1*10**6
test_b = 1*10**6
test_c = (1.5625*10**5 - 0.02*test_b - 0.02*test_a)
## Fan velocity for test
fan_velocity = 10.0
heq.set_heat_generation(heat_generation_function,test_a,test_b,test_c)
heq.set_fan_velocity(fan_velocity)
## Skip initial condition plot during optimization
# fig, ax = plt.subplots()
# contour1 = ax.contourf(heq.X,heq.Y,heq.u - 273)
# fig.colorbar(contour1,ax=ax)
# plt.show()
## Setting objective function
heq.max_iter = 5E5
w1 = 0.2
w2 = 1 - w1
global_tolerance = 1E-3


def objective_function(x):
    """Objective Function
    
    Minimizes: w1 * max(T) - w2 * eta(v)
    
    Parameters
    ------
    x[0] (float): Velocity of Fan (v)
    x[1] (float): a coefficient of heat generation
    x[2] (float): b coefficient of heat generation
    x[3] (float): c coefficient of heat generation
    """
    v, a, b, c = x  # Tuple unpacking

    v = v * 20
    a = a * 50
    b = b * 50
    c = c * 10**5

    # Must update heq with current design variables and solve!
    heq.set_fan_velocity(v)
    heq.set_heat_generation(heat_generation_function, a, b, c)
    heq.reset()
    heq.solve_until_steady_state(tol=global_tolerance)
    
    # Now compute objective with the solved temperature field
    max_T = np.max(heq.u)
    eta = heq.fan_efficiency
    
    return w1 * max_T/273 - w2 * eta

## Bounds for inputs
# v: fan velocity [0, 30] m/s (physical limits)
# a, b: heat distribution coefficients (unconstrained)
# c: base heat generation (can be negative if a,b contribute enough)
bounds = [
    (0.1/20, 1.5),        # v: fan velocity (avoid 0 for numerical stability)
    (-1e8, 1e8),      # a
    (-1e8, 1e8),      # b
    (0, 1e8),      # c
]

def constraint_one(x):
    """Constraint for total power generation by the CPU = 10W
    
    The constraint is: ∫∫∫ f(x,y) dV = 10
    Returns 0 when satisfied (equality constraint)
    
    Parameters
    -------
    x[1] (float): a coefficient of heat generation
    x[2] (float): b coefficient of heat generation
    x[3] (float): c coefficient of heat generation
    """
    v, a, b, c = x  # Tuple unpacking

    v = v * 20
    a = a * 50
    b = b * 50
    c = c * 10**5

    # Update heat generation to get the integrated total
    heq.set_heat_generation(heat_generation_function, a, b, c)
    return heq.heat_generation_total - 10

## Setting the constraints
constraints = [
    {'type': 'eq', 'fun': constraint_one},
]
## Creating the initial guess
# For 10W total power in 0.04³ m³ volume with a=b=0: c = 10 / (0.04 * 0.04 * 0.04) ≈ 156250 W/m³
v0 = 0.5  # m/s - start at reasonable fan velocity
a0 = 0.1   # Start with uniform heat generation
b0 = 0.1
c0 = 1.0  # This gives approximately 10W total
x0 = [v0, a0, b0, c0]
heq.verbose = False
    

t0 = time.perf_counter()
t_FD = [0]
def callback(intermediate_result):
    x_path.append(intermediate_result.x.copy())
    norm_grad_l_FD.append(np.linalg.norm(intermediate_result.lagrangian_grad))
    # For trust-constr, jac can be a tuple (objective_grad, constraint_jacs)
    jac = intermediate_result.jac
    if isinstance(jac, (tuple, list)):
        jac = jac[0]
    norm_grad_f.append(np.linalg.norm(np.asarray(jac).flatten()))
    # Track objective, max T, efficiency, and constraint
    objective_history.append(intermediate_result.fun)
    max_T_history.append(np.max(heq.u))
    eta_history.append(heq.fan_efficiency)
    constraint_history.append(heq.heat_generation_total - 10)
    t_FD.append(time.perf_counter() - t0)

optimization_result_FD = optimize.minimize(
    objective_function,
    method='trust-constr',
    x0=x0,
    bounds=bounds,
    constraints=constraints,
    callback=callback,
    options={'maxiter': 500, 'verbose': 2}
)
t_FD_tot = time.perf_counter() - t0

# %% 
# convergence of Lagrangian with time



fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(t_AD[:-1], norm_grad_l_AD, label='AD')
ax.semilogy(t_FD[:-1], norm_grad_l_FD, label='FD')
ax.set_xlabel('time (s)')
ax.set_ylabel(r'$\|\nabla \mathcal{L}\|$')
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()


# %%
