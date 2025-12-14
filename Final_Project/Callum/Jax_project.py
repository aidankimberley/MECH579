import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
import numpy as np
from collections.abc import Callable
from functools import partial
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


mpl.rcParams['axes.formatter.useoffset'] = False

class HeatEquation2D:

    def __init__(self, x:float, y:float, height:float , n_x:int, n_y:int,
                     k:float=1.0, rho:float=1.0, cp:float=1.0,
                     CFL:float=0.1, init_condition:Callable[[jnp.ndarray,jnp.ndarray], jnp.ndarray] = lambda x,y: jnp.sin(x+y)):

        ## MESHING variables
        self.n_x = n_x
        self.n_y = n_y
        self.boundary_conditions = []
        # Physical locations
        x_axis = jnp.linspace(0, x, self.n_x)
        y_axis = jnp.linspace(0, y, self.n_y)
        self.X, self.Y = jnp.meshgrid(x_axis, y_axis, indexing='ij')
        self.dx = x_axis[1] - x_axis[0]
        self.dy = y_axis[1] - y_axis[0]
        # Variables of Mesh size
        self.u = jnp.zeros((self.n_x, self.n_y))
        self.h_top_values = jnp.zeros((self.n_x, self.n_y))
        self.h_boundary_values = jnp.zeros((self.n_x, self.n_y))

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

    def set_initial_conditions(self,initial_conditions:Callable[[jnp.ndarray,jnp.ndarray],jnp.ndarray]):
        self.init_condition = initial_conditions

    def apply_initial_conditions(self):
        """Applies the initial condition into self.u"""
        self.u = self.init_condition(self.X,self.Y)

    def reset(self):
        """Resets the heat equation"""
        self.apply_initial_conditions()
        self.current_time = 0
        self.steady_state_error = 1E2

    def set_heat_generation(self, heat_generation_function: Callable[[jnp.ndarray,jnp.ndarray,float,float,float], jnp.ndarray],
                            a: float, b: float, c: float):
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
        self.heat_generation_total = jnp.sum(jnp.sum(heat_generation_matrix))

    def set_fan_velocity(self, v: float):
        self.v = v
        self.fan_efficiency = self.fan_efficiency_func(self.v)


    def h_boundary(self,u: jnp.ndarray):
        beta = 1/((u+self.ext_T)/2)
        rayleigh = 9.81*beta*(u-self.ext_T)*self.dx**3/(self.ext_nu**2)*self.ext_Pr
        nusselt = (0.825 + (0.387*rayleigh**(1/6))/
                   (1+(0.492/self.ext_Pr)**(9/16))**(8/27))**2
        return nusselt*self.ext_k/self.dx

    def h_top(self,x: jnp.ndarray, u):
        Rex = self.v*x/self.ext_nu
        r,c = Rex.shape
        Nux = jnp.zeros((r,c))
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
        self.steady_state_error = jnp.linalg.norm(self.u - old_u,jnp.inf)
        self.current_time += self.dt

    def solve_until_steady_state(self, tol: float = 1e-3):
        iter = 0
        self.step_forward_in_time()
        while self.steady_state_error > tol and iter < self.max_iter:
            self.step_forward_in_time()
            iter += 1
            if (iter % 1000) == 0 and self.verbose:
                print(f"Iteration: {iter}, Error: {self.steady_state_error}")


    def solve_until_time(self,final_time: float):
        iter = 0
        while self.current_time < final_time:
            self.step_forward_in_time()
            iter += 1
            if (iter % 1000) == 0 and self.verbose:
                print(f"Iteration: {iter}, Time: {self.current_time}")


# -------------------------------------------------------------------------
# JAX "physics core" – pure functions used in objective/constraint
# -------------------------------------------------------------------------
def fan_efficiency_func(v):
    return -0.002 * v**2 + 0.08 * v


def build_params(v, a, b, c_coef, heq: HeatEquation2D):
    """Pack PDE parameters into a dict for JAX solver."""
    return {
        "dx": heq.dx,
        "dy": heq.dy,
        "dt": heq.dt,
        "k": heq.k,
        "thermal_alpha": heq.thermal_alpha,
        "height": heq.height,
        "ext_k": heq.ext_k,
        "ext_Pr": heq.ext_Pr,
        "ext_nu": heq.ext_nu,
        "ext_T": heq.ext_T,
        "v": v,
        "heat_gen_a": a,
        "heat_gen_b": b,
        "heat_gen_c": c_coef,
    }


def h_boundary_jax(u: jnp.ndarray, params):
    dx = params["dx"]
    ext_k = params["ext_k"]
    ext_Pr = params["ext_Pr"]
    ext_nu = params["ext_nu"]
    ext_T = params["ext_T"]

    beta = 1.0 / ((u + ext_T) / 2.0)

    rayleigh = 9.81 * beta * (u - ext_T) * dx**3 / (ext_nu**2) * ext_Pr
    rayleigh = jnp.maximum(rayleigh, 0.0)
    nusselt = (
        0.825
        + (0.387 * rayleigh ** (1.0 / 6.0))
        / (1.0 + (0.492 / ext_Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    ) ** 2
    return nusselt * ext_k / dx


def h_top_jax(x: jnp.ndarray, u: jnp.ndarray, params):
    ext_k = params["ext_k"]
    ext_Pr = params["ext_Pr"]
    ext_nu = params["ext_nu"]
    v = params["v"]

    x_eff = x + 1e-6
    Rex = v * x_eff / ext_nu
    Nux = jnp.where(
        Rex < 5e5,
        0.332 * jnp.sqrt(Rex) * ext_Pr ** (1.0 / 3.0),
        0.0296 * Rex ** 0.8 * ext_Pr ** (1.0 / 3.0),
    )
    h = Nux * ext_k / (x + 1e-5)
    return h


def heat_generation_jax(X, Y, params):
    return (
        params["heat_gen_a"] * X
        + params["heat_gen_b"] * Y
        + params["heat_gen_c"]
    )


def heat_generation_total_jax(X, Y, params):
    dx = params["dx"]
    dy = params["dy"]
    height = params["height"]

    heat_matrix = heat_generation_jax(X, Y, params) * dx * dy * height

    i0, j0 = 0, 0
    iN, jN = heat_matrix.shape[0] - 1, heat_matrix.shape[1] - 1

    # Faces
    heat_matrix = heat_matrix.at[i0, :].divide(2.0)
    heat_matrix = heat_matrix.at[iN, :].divide(2.0)
    heat_matrix = heat_matrix.at[j0, :].divide(2.0)
    heat_matrix = heat_matrix.at[jN, :].divide(2.0)

    # Corners
    heat_matrix = heat_matrix.at[i0, j0].divide(2.0)
    heat_matrix = heat_matrix.at[iN, jN].divide(2.0)
    heat_matrix = heat_matrix.at[iN, j0].divide(2.0)
    heat_matrix = heat_matrix.at[i0, jN].divide(2.0)

    return jnp.sum(heat_matrix)


def step_forward_jax(u: jnp.ndarray,
                     X: jnp.ndarray,
                     Y: jnp.ndarray,
                     params) -> jnp.ndarray:
    dx = params["dx"]
    dy = params["dy"]
    k = params["k"]
    thermal_alpha = params["thermal_alpha"]
    height = params["height"]
    ext_T = params["ext_T"]

    dt = params["dt"]
    tau = thermal_alpha * dt / (dx * dy)

    # convection coefficients and source
    h_top_values = h_top_jax(X, u, params)
    h_boundary_values = h_boundary_jax(u, params)
    e_dot = heat_generation_jax(X, Y, params)

    i0, j0 = 0, 0
    iN, jN = u.shape[0] - 1, u.shape[1] - 1

    new_u = u

    # boundaries
    # Left
    left_val = (
        u[i0, 1:-1] +
        2 * tau * h_boundary_values[i0, 1:-1] / k * dy * (ext_T - u[i0, 1:-1]) +
        tau * dx * (u[i0, 2:] - u[i0, 1:-1]) / dy +
        tau * dx * (u[i0, 1:-1] - u[i0, 2:]) / dy +
        2 * tau * dy * (u[i0 + 1, 1:-1] - u[i0, 1:-1]) / dx +
        tau * h_top_values[i0, 1:-1] / k * dx * dy / height * (ext_T - u[i0, 1:-1]) +
        tau * e_dot[i0, 1:-1] / k * dx * dy
    )
    new_u = new_u.at[i0, 1:-1].set(left_val)

    # Right
    right_val = (
        u[iN, 1:-1] +
        2 * tau * h_boundary_values[iN, 1:-1] / k * dy * (ext_T - u[iN, 1:-1]) +
        tau * dx * (u[iN, 2:] - u[iN, 1:-1]) / dy +
        tau * dx * (u[iN, 1:-1] - u[iN, 2:]) / dy +
        2 * tau * dy * (u[iN - 1, 1:-1] - u[iN, 1:-1]) / dx +
        tau * h_top_values[iN, 1:-1] / k * dx * dy / height * (ext_T - u[iN, 1:-1]) +
        tau * e_dot[iN, 1:-1] / k * dx * dy
    )
    new_u = new_u.at[iN, 1:-1].set(right_val)

    # Bottom
    bottom_val = (
        u[1:-1, j0] +
        2 * tau * h_boundary_values[1:-1, j0] / k * dx * (ext_T - u[1:-1, j0]) +
        tau * dy * (u[2:, j0] - u[1:-1, j0]) / dx +
        tau * dy * (u[1:-1, j0] - u[2:, j0]) / dx +
        2 * tau * dx * (u[1:-1, j0 + 1] - u[1:-1, j0]) / dy +
        tau * h_top_values[1:-1, j0] / k * dx * dy / height * (ext_T - u[1:-1, j0]) +
        tau * e_dot[1:-1, j0] / k * dx * dy
    )
    new_u = new_u.at[1:-1, j0].set(bottom_val)

    # Top
    top_val = (
        u[1:-1, jN] +
        2 * tau * h_boundary_values[1:-1, jN] / k * dx * (ext_T - u[1:-1, jN]) +
        tau * dy * (u[2:, jN] - u[1:-1, jN]) / dx +
        tau * dy * (u[1:-1, jN] - u[2:, jN]) / dx +
        2 * tau * dx * (u[1:-1, jN - 1] - u[1:-1, jN]) / dy +
        tau * h_top_values[1:-1, jN] / k * dx * dy / height * (ext_T - u[1:-1, jN]) +
        tau * e_dot[1:-1, jN] / k * dx * dy
    )
    new_u = new_u.at[1:-1, jN].set(top_val)

    # Bottom-left corner
    bl_val = (
        u[i0, j0] +
        2 * tau * h_boundary_values[i0, j0] * dy / k * (ext_T - u[i0, j0]) +
        2 * tau * h_boundary_values[i0, j0] * dx / k * (ext_T - u[i0, j0]) +
        2 * tau * dx * (u[i0, j0 + 1] - u[i0, j0]) / dy +
        2 * tau * dy * (u[i0 + 1, j0] - u[i0, j0]) / dx +
        tau * h_top_values[i0, j0] / k * dx * dy / height * (ext_T - u[i0, j0]) +
        tau * e_dot[i0, j0] / k * dx * dy
    )
    new_u = new_u.at[i0, j0].set(bl_val)

    # Bottom-right corner
    br_val = (
        u[iN, j0] +
        2 * tau * h_boundary_values[iN, j0] * dy / k * (ext_T - u[iN, j0]) +
        2 * tau * h_boundary_values[iN, j0] * dx / k * (ext_T - u[iN, j0]) +
        2 * tau * dx * (u[iN, j0 + 1] - u[iN, j0]) / dy +
        2 * tau * dy * (u[iN - 1, j0] - u[iN, j0]) / dx +
        tau * h_top_values[iN, j0] / k * dx * dy / height * (ext_T - u[iN, j0]) +
        tau * e_dot[iN, j0] / k * dx * dy
    )
    new_u = new_u.at[iN, j0].set(br_val)

    # Top-left corner
    tl_val = (
        u[i0, jN] +
        2 * tau * h_boundary_values[i0, jN] * dy / k * (ext_T - u[i0, jN]) +
        2 * tau * h_boundary_values[i0, jN] * dx / k * (ext_T - u[i0, jN]) +
        2 * tau * dx * (u[i0, jN - 1] - u[i0, jN]) / dy +
        2 * tau * dy * (u[i0 + 1, jN] - u[i0, jN]) / dx +
        tau * h_top_values[i0, jN] / k * dx * dy / height * (ext_T - u[i0, jN]) +
        tau * e_dot[i0, jN] / k * dx * dy
    )
    new_u = new_u.at[i0, jN].set(tl_val)

    # Top-right corner
    tr_val = (
        u[iN, jN] +
        2 * tau * h_boundary_values[iN, jN] * dy / k * (ext_T - u[iN, jN]) +
        2 * tau * h_boundary_values[iN, jN] * dx / k * (ext_T - u[iN, jN]) +
        2 * tau * dx * (u[iN, jN - 1] - u[iN, jN]) / dy +
        2 * tau * dy * (u[iN - 1, jN] - u[iN, jN]) / dx +
        tau * h_top_values[iN, jN] / k * dx * dy / height * (ext_T - u[iN, jN]) +
        tau * e_dot[iN, jN] / k * dx * dy
    )
    new_u = new_u.at[iN, jN].set(tr_val)

    # Interior
    interior = (
        u[1:-1, 1:-1] +
        tau * (
            dy * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1]) / dx +
            dx * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]) / dy
        ) +
        tau * (
            h_top_values[1:-1, 1:-1] / k * dx * dy / height * (ext_T - u[1:-1, 1:-1]) +
            dx * dy / k * e_dot[1:-1, 1:-1]
        )
    )
    new_u = new_u.at[1:-1, 1:-1].set(interior)

    return new_u


@partial(jax.jit, static_argnums=(4,))
def run_solver_jax(u0: jnp.ndarray,
                   X: jnp.ndarray,
                   Y: jnp.ndarray,
                   params,
                   n_steps: int) -> jnp.ndarray:

    def body_fun(u, _):
        u_next = step_forward_jax(u, X, Y, params)
        return u_next, None

    u_final, _ = lax.scan(body_fun, u0, xs=None, length=n_steps)
    return u_final

# Optimization with JAX-based objective/constraint
def run_ad():
    # Setup plots directory
    save_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(save_dir, exist_ok=True)

    cpu_x = 0.04
    cpu_y = 0.04
    cpu_z = 0.04
    N = 25

    CFL = 0.5
    k_si = 149.0
    rho_si = 2323.0
    c_si = 19.789 / 28.085 * 1000.0  # J/(kg*K)

    # Initial condition
    def initial_condition(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        u = 70.0 * jnp.sin(x * jnp.pi / cpu_x) * jnp.sin(y * jnp.pi / cpu_y) + 293.0
        return u

    def heat_generation_function(x: jnp.ndarray, y: jnp.ndarray,
                                 a: float, b: float, c: float) -> jnp.ndarray:
        return a * x + b * y + c

    heq = HeatEquation2D(cpu_x, cpu_y, cpu_z, N, N,
                         k=k_si, rho=rho_si, cp=c_si,
                         init_condition=initial_condition)

    # Visual check of initial condition
    fig, ax = plt.subplots()
    contour1 = ax.contourf(heq.X, heq.Y, heq.u - 273.0)
    fig.colorbar(contour1, ax=ax)
    plt.title("Initial condition (°C)")
    plt.savefig(os.path.join(save_dir, 'initial_condition.png'))
    plt.show()

    # Optimization parameters
    heq.max_iter = 5e4
    w1 = 0.2
    w2 = 1.0 - w1

    iter_log = {
        "x": [],
        "J": [],
        "T_max": [],
        "eta": [],
        "power": [],
        "constraint": [],
        "grad_norm": [],
    }


    def objective_function(x):
        v, a, b, c_coef = x
        params = build_params(v, a, b, c_coef, heq)

        u0 = initial_condition(heq.X, heq.Y)
        n_steps = int(heq.max_iter)

        u_final = run_solver_jax(u0, heq.X, heq.Y, params, n_steps)

        T_max = jnp.max(u_final)
        eta = fan_efficiency_func(v)

        return w1 * T_max / 273.0 - w2 * eta

    bounds = [
        (0, 30),
        (-jnp.inf, jnp.inf),
        (-jnp.inf, jnp.inf),
        (0, jnp.inf),
    ]

    def constraint_one(x):
        v, a, b, c_coef = x
        params = build_params(v, a, b, c_coef, heq)
        power_total = heat_generation_total_jax(heq.X, heq.Y, params)
        return 10.0 - power_total

    obj_grad = jax.grad(objective_function)
    c1_grad = jax.grad(constraint_one)


    def obj_grad_jax(x):
        return np.array(obj_grad(x), dtype=float)


    def c1_grad_jax(x):
        return np.array(c1_grad(x), dtype=float)


    def callback(x):
        v, a, b, c_coef = x
        params = build_params(v, a, b, c_coef, heq)

        u0 = initial_condition(heq.X, heq.Y)
        n_steps = int(heq.max_iter)
        u_final = run_solver_jax(u0, heq.X, heq.Y, params, n_steps)

        J = objective_function(x)
        T_max = jnp.max(u_final)
        eta = fan_efficiency_func(v)
        power = heat_generation_total_jax(heq.X, heq.Y, params)
        c_val = constraint_one(x)

        g_f = obj_grad_jax(x)
        g_c = c1_grad_jax(x)

        # Solve for λ and L
        denom = float(np.dot(g_c, g_c)) + 1e-12
        lam = float(np.dot(g_f, g_c) / denom)
        gL = g_f - lam * g_c
        gL_norm = float(np.linalg.norm(gL))
        log_gL_norm = float(np.log(gL_norm + 1e-16))

        iter_log["x"].append(np.array(x, dtype=float))
        iter_log["J"].append(float(J))
        iter_log["T_max"].append(float(T_max))
        iter_log["eta"].append(float(eta))
        iter_log["power"].append(float(power))
        iter_log["constraint"].append(float(c_val))
        iter_log["grad_norm"].append(log_gL_norm)

    constraints = [
        {"type": "eq", "fun": constraint_one, "jac": c1_grad_jax}
    ]

    # Initial guess
    v0 = 10
    x0_heat = 0.0
    x0 = [v0, x0_heat * 10 ** 5, x0_heat * 10 ** 5, (156250 - 0.02 * x0_heat * 10 ** 5 - 0.02 * x0_heat * 10 ** 5)]
    heq.verbose = False

    callback(np.array(x0, dtype=float))

    # Run optimization
    optimization_result = optimize.minimize(
        objective_function,
        x0,
        method='SLSQP',
        bounds=bounds,
        jac=obj_grad_jax,
        constraints=constraints,
        callback=callback,
        options={"maxiter": 200},  # keep this moderate
    )

    # Build optimal solution
    v_opt, a_opt, b_opt, c_opt = optimization_result.x
    params_opt = build_params(v_opt, a_opt, b_opt, c_opt, heq)

    u0 = initial_condition(heq.X, heq.Y)
    n_steps = int(heq.max_iter)
    u_opt = run_solver_jax(u0, heq.X, heq.Y, params_opt, n_steps)

    print("status:", optimization_result.status)
    print("message:", optimization_result.message)
    print("nit:", optimization_result.nit)
    print("fun at x0:", objective_function(x0))
    print("fun at optimum:", optimization_result.fun)

    print(
        f"Optimization result: {float(objective_function(optimization_result.x))}\n"
        f"v: {v_opt} m/s, "
        f"a: {a_opt}, "
        f"b: {b_opt}, "
        f"c: {c_opt}\n"
        f"Total Heat Generation: {float(heat_generation_total_jax(heq.X, heq.Y, params_opt))} "
        f"Constraint: {float(constraint_one(optimization_result.x))}\n"
    )

    # Plot optimal temperature field
    fig, ax = plt.subplots()
    contour3 = ax.contourf(heq.X, heq.Y, u_opt - 273.0)
    fig.colorbar(contour3, ax=ax)
    plt.title("Optimal temperature field (°C)")
    plt.savefig(os.path.join(save_dir, 'optimal_temperature_field.png'))
    plt.show()

    x_list = iter_log["x"]
    J_list = iter_log["J"]
    T_max_list = iter_log["T_max"]
    eta_list = iter_log["eta"]
    P_list = iter_log["power"]
    cons_list = iter_log["constraint"]
    ln_list = iter_log["grad_norm"]

    iters = np.arange(len(J_list))
    if len(x_list) > 0:
        x_array = np.vstack(x_list)
        v_list = x_array[:, 0]
        a_list = x_array[:, 1]
        b_list = x_array[:, 2]
        c_list = x_array[:, 3]
    else:
        v_list = a_list = b_list = c_list = np.array([])

    # Plots
    # 1) Objective
    fig1, ax1 = plt.subplots()
    ax1.plot(iters, J_list, label="Objective Function")
    ax1.set_title("Objective Function - AD")
    ax1.set_xlabel("Iteration [n]")
    ax1.set_ylabel("$ω_1$maxT−$ω_2$$\eta$")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, 'objective_function_AD.png'))

    # 2) ||∇L||
    fig2, ax2 = plt.subplots()
    ax2.plot(iters, ln_list, label="ln(||∇L||)")
    ax2.set_title("Lagrangian Gradient Norm - AD")
    ax2.set_xlabel("Iteration [n]")
    ax2.set_ylabel("ln(||∇L||)")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'gradient_lagrangian_AD.png'))

    # 3) T_max
    fig3, ax3 = plt.subplots()
    ax3.plot(iters, T_max_list, label=r"$T_{\max}$")
    ax3.set_title("Maximum Temperature - AD")
    ax3.set_xlabel("Iteration [n]")
    ax3.set_ylabel("Temperature [K]")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, 'max_temperature_AD.png'))

    # 4) Efficiency
    fig4, ax4 = plt.subplots()
    ax4.plot(iters, eta_list, label="Efficiency")
    ax4.set_title("Fan Efficiency - AD")
    ax4.set_xlabel("Iteration [n]")
    ax4.set_ylabel(r"$\eta$")
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, 'fan_efficiency_AD.png'))

    # 5) Power and constraint
    fig5, ax5 = plt.subplots(1, 2, figsize=(12, 6))
    ax5[0].plot(iters, P_list, label="Power")
    ax5[0].set_title("Power - AD")
    ax5[0].set_xlabel("Iteration [n]")
    ax5[0].set_ylabel("Power [W]")
    ax5[0].legend()
    ax5[0].grid(True)

    ax5[1].plot(iters, cons_list, label="Power Constraint (10 - P)")
    ax5[1].set_title("Constraint Value - AD")
    ax5[1].set_xlabel("Iteration [n]")
    ax5[1].set_ylabel("10 - P [W]")
    ax5[1].legend()
    ax5[1].grid(True)
    fig5.tight_layout()
    fig5.savefig(os.path.join(save_dir, 'power_constraint_AD.png'))

    # 6) Design parameters
    fig6, ax6 = plt.subplots(2, 2, figsize=(12, 6))
    ax6[0, 0].plot(iters, v_list, label="v")
    ax6[0, 0].legend()
    ax6[0, 0].set_xlabel("Iteration [n]")
    ax6[0, 0].set_ylabel("Velocity [m/s]")
    ax6[0, 0].grid(True)

    ax6[0, 1].plot(iters, a_list, label="a")
    ax6[0, 1].set_xlabel("Iteration [n]")
    ax6[0, 1].set_ylabel("[units]")
    ax6[0, 1].legend()
    ax6[0, 1].grid(True)

    ax6[1, 0].plot(iters, b_list, label="b")
    ax6[1, 0].set_xlabel("Iteration [n]")
    ax6[1, 0].set_ylabel("[units]")
    ax6[1, 0].legend()
    ax6[1, 0].grid(True)

    ax6[1, 1].plot(iters, c_list, label="c")
    ax6[1, 1].set_xlabel("Iteration [n]")
    ax6[1, 1].set_ylabel("[units]")
    ax6[1, 1].legend()
    ax6[1, 1].grid(True)

    fig6.suptitle("Design Parameters - AD")
    fig6.tight_layout()
    fig6.savefig(os.path.join(save_dir, 'design_parameters_AD.png'))
    plt.show()