# %%
# Imports
import numpy as np # numpy for vectorization
from collections.abc import Callable # For type hints
import matplotlib.pyplot as plt
from scipy import optimize
import argparse


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
if __name__ == "__main__":
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(description='Heat equation optimization with Pareto front generation')
    # parser.add_argument('--skip-initial', action='store_true',
    #                    help='Skip the initial optimization and only generate Pareto front')
    # parser.add_argument('--pareto-only', action='store_true',
    #                    help='Only generate Pareto front (alias for --skip-initial)')
    # args = parser.parse_args()
    
    # skip_initial = args.skip_initial or args.pareto_only
    skip_initial = False
    
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
    norm_grad_l = []
    norm_grad_f = []
    objective_history = []
    max_T_history = []
    eta_history = []
    constraint_history = []

    def callback(intermediate_result):
        x_path.append(intermediate_result.x.copy())
        norm_grad_l.append(np.linalg.norm(intermediate_result.lagrangian_grad))
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

    def callback2(xk):
        """Callback for SLSQP - records state after each iteration.
        
        Note: heq should already be updated from the last objective_function call,
        so we just read current values without re-solving.
        """
        x_path.append(xk.copy())
        # SLSQP doesn't provide gradient info in callback
        norm_grad_l.append(np.nan)
        norm_grad_f.append(np.nan)
        # Record current state (heq was already solved in objective_function)
        max_T = np.max(heq.u)
        eta = heq.fan_efficiency
        obj_val = w1 * max_T/273 - w2 * eta
        objective_history.append(obj_val)
        max_T_history.append(max_T)
        eta_history.append(eta)
        constraint_history.append(heq.heat_generation_total - 10)
        
        # Print progress
        print(f"  Iter {len(x_path):3d}: obj={obj_val:.4f}, maxT={max_T-273:.2f}°C, η={eta:.4f}, v={xk[0]:.2f}")

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
    
    if not skip_initial:
        print(f"Initial guess: v={v0} m/s, a={a0}, b={b0}, c={c0}")
        print(f"Starting optimization with w1={w1}, w2={w2}...")
        print("-" * 50)
        ## Optimize
        optimization_result = optimize.minimize(
            objective_function,
            method=OPTIMIZATION_METHOD,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            callback=callback,
            options={'maxiter': 500, 'verbose': 2}
        )

        ## Build and evaluate optimal solution
        heq.set_fan_velocity(20 * optimization_result.x[0])
        heq.set_heat_generation(heat_generation_function, 50*optimization_result.x[1], 50*optimization_result.x[2],
                                10**5 * optimization_result.x[3])
        heq.reset()
        heq.solve_until_steady_state(tol=global_tolerance)
        
        optimal_max_T = np.max(heq.u)
        optimal_eta = heq.fan_efficiency
        optimal_obj = w1 * optimal_max_T - w2 * optimal_eta
        
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
        print(f"  Total Heat Gen.    = {heq.heat_generation_total:.4f} W (target: 10 W)")
        print(f"  Constraint Error   = {heq.heat_generation_total - 10:.6f} W")
        print("=" * 50)

        # Convert tracking lists to numpy arrays
        x_path = np.array(x_path)
        iterations = np.arange(len(x_path))
        
        v_path = 20 * x_path[:, 0]
        a_path = 50 * x_path[:, 1]
        b_path = 50 * x_path[:, 2]
        c_path = 10**5 * x_path[:, 3]
        
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
        contour = ax.contourf(heq.X * 1000, heq.Y * 1000, heq.u - 273, levels=20, cmap='hot')
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
        else:
            print("Note: Gradient of objective not available for SLSQP method")

        print("\nAll plots saved to current directory.")
    else:
        print("Skipping initial optimization (use --skip-initial flag)...")





# %%
