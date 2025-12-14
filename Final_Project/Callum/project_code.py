from collections.abc import Callable
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
import numpy as np
import jax
import jax.numpy as jnp


mpl.rcParams['axes.formatter.useoffset'] = False

class HeatEquation2D:

    def __init__(self, x:float, y:float, height:float , n_x:int, n_y:int,
                     k:float=1.0, rho:float=1.0, cp:float=1.0,
                     CFL:float=0.1, init_condition:Callable[[np.ndarray,np.ndarray], np.ndarray] = lambda x,y: np.sin(x+y)):

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
        self.v = v
        self.fan_efficiency = self.fan_efficiency_func(self.v)


    def h_boundary(self,u: np.ndarray):
        beta = 1/((u+self.ext_T)/2)
        rayleigh = 9.81*beta*(u-self.ext_T)*self.dx**3/(self.ext_nu**2)*self.ext_Pr
        nusselt = (0.825 + (0.387*rayleigh**(1/6))/
                   (1+(0.492/self.ext_Pr)**(9/16))**(8/27))**2
        return nusselt*self.ext_k/self.dx

    def h_top(self,x: np.ndarray, u):
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


def run_fd():
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
    ## plotting initial conditions
    fig, ax = plt.subplots()
    contour1 = ax.contourf(heq.X,heq.Y,heq.u - 273)
    fig.colorbar(contour1,ax=ax)
    plt.show()
    ## Setting objective function
    heq.max_iter = 5e4
    w1 = 0.2
    w2 = 1 - w1
    global_tolerance = 1E-2

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
        heq.set_fan_velocity(v)
        heq.set_heat_generation(heat_generation_function, a, b, c_coef)
        heq.reset()
        heq.solve_until_steady_state(global_tolerance)
        T_max = np.max(heq.u)

        return w1 * T_max / 273.0 - w2 * heq.fan_efficiency


    bounds = [
        (0, 30),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (0, np.inf),
    ]

    def constraint_one(x):
        _, a, b, c_coef = x
        heq.set_heat_generation(heat_generation_function, a, b, c_coef)

        return 10 - heq.heat_generation_total


    def callback(x, state):
        J = objective_function(x)
        T_max = np.max(heq.u)
        eta = heq.fan_efficiency
        power = heq.heat_generation_total
        c_val = constraint_one(x)

        grad_L = state.lagrangian_grad
        grad_norm = np.log(np.linalg.norm(grad_L))

        iter_log['x'].append(x)
        iter_log['J'].append(J)
        iter_log['T_max'].append(T_max)
        iter_log['eta'].append(eta)
        iter_log['power'].append(power)
        iter_log['constraint'].append(c_val)
        iter_log['grad_norm'].append(grad_norm)

    constraints = [
        {'type': 'eq', 'fun': constraint_one},
    ]
    ## Initial guess
    v0 = 10
    x0_heat = 0
    x0 = [v0, x0_heat * 10 ** 5, x0_heat * 10 ** 5, (156250 - 0.02 * x0_heat * 10 ** 5 - 0.02 * x0_heat * 10 ** 5)]


    # JAX: PART C
    def objective_function_jax(x: jnp.ndarray):
        v = x[0]

        eta_v = -0.002 * v ** 2 + 0.08 * v
        return -w2 * eta_v

    obj_grad_jax = jax.grad(objective_function_jax)

    def central_difference_gradient(f, x, i, h=1e-6):
        x_plus = list(x)
        x_plus[i] += h
        x_minus = list(x)
        x_minus[i] -= h
        return (f(x_plus) - f(x_minus)) / (2 * h)

    # FD Convergence
    h_values = np.logspace(-3, -12, 10)
    fd_grad_v_values = []
    v_index = 0
    print("\n--- FD Convergence Check for dJ/dv at initial guess x0 ---")
    for h in h_values:
        fd_grad = central_difference_gradient(objective_function, x0, v_index, h=h)
        fd_grad_v_values.append(fd_grad)
        print(f"h: {h:.1e}, dJ/dv: {fd_grad:.8f}")

    h_converged = 1e-8
    fd_grad_converged_v = central_difference_gradient(objective_function, x0, v_index, h=h_converged)

    # AD vs FD Comparison for Part (c)
    ad_grad_eta_v = obj_grad_jax(jnp.array(x0))[v_index].item()

    # Finite Difference Gradient
    def algebraic_eta_only_function(x):
        v = x[0]
        eta = -0.002 * v ** 2 + 0.08 * v
        return -w2 * eta

    fd_grad_eta_v = central_difference_gradient(algebraic_eta_only_function, x0, v_index, h=1e-8)

    print("\n--- AD vs FD Comparison Table ---")
    print(f"Design Parameter: v (index {v_index})")
    print(f"Initial Guess x0: v={x0[0]:.2f}, a={x0[1]:.2e}, b={x0[2]:.2e}, c={x0[3]:.2f}")

    data = {
        "Method": ["Finite Difference (h=1e-8)", "Automatic Differentiation (JAX)"],
        "d/dv [-w2 * eta(v)]": [f"{fd_grad_eta_v:.10f}", f"{ad_grad_eta_v:.10f}"],
    }

    print("\n| ---------- Method ------ | ----------- Value ---------- |")
    print(f"| {data['Method'][0]} | {data['d/dv [-w2 * eta(v)]'][0]} |")
    print(f"| {data['Method'][1]} | {data['d/dv [-w2 * eta(v)]'][1]} |")
    print(f"\nConverged FD of FULL Objective (dJ/dv): {fd_grad_converged_v:.10f} (h={h_converged:.1e})")

    heq.verbose = False
    ## Optimize
    optimization_result = optimize.minimize(
        objective_function,
        x0,
        method='trust-constr',
        bounds=bounds,
        constraints=constraints,
        callback=callback
    )
    ## Build optimal solution
    heq.set_fan_velocity(optimization_result.x[0])
    heq.set_heat_generation(heat_generation_function, optimization_result.x[1], optimization_result.x[2],
                            optimization_result.x[3])
    print(
        f"Optimization result: {objective_function(optimization_result.x)}\n"
        f"v: {optimization_result.x[0]} m/s, "
        f"a: {optimization_result.x[1]}, "
        f"b: {optimization_result.x[2]}, "
        f"c: {optimization_result.x[3]}"
        f"\n"
        f"Constraints:\n"
        f"Total Heat Generation: {heq.heat_generation_total} Constraint: {constraint_one(optimization_result.x)}\n"
    )

    x_list = iter_log['x']
    J_list = iter_log['J']
    T_max_list = iter_log['T_max']
    eta_list = iter_log['eta']
    P_list = iter_log['power']
    cons_list = iter_log['constraint']
    gn_list = iter_log['grad_norm']

    iters = np.arange(len(J_list))

    x_array = np.array(x_list)

    v_list = x_array[:, 0]
    a_list = x_array[:, 1]
    b_list = x_array[:, 2]
    c_list = x_array[:, 3]

    # 1) Objective function
    fig1, ax1 = plt.subplots()
    ax1.plot(iters, J_list, label='Objective Function')
    ax1.set_title('Objective Function - FD')
    ax1.set_xlabel('Iteration [n]')
    ax1.set_ylabel('$ω_1$maxT−$ω_2$$\eta$')
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()

    # 2) ||∇L||
    fig2, ax2 = plt.subplots()
    ax2.plot(iters, gn_list, label='ln(||∇L||)')
    ax2.set_title('Lagrangian Gradient Norm - FD')
    ax2.set_xlabel('Iteration [n]')
    ax2.set_ylabel('ln(||∇L||)')
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()

    # 3) T_max
    fig3, ax3 = plt.subplots()
    ax3.plot(iters, T_max_list, label='$T_{max}$')
    ax3.set_title('Maximum Temperature - FD')
    ax3.set_xlabel('Iteration [n]')
    ax3.set_ylabel('Temperature [K]')
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()

    # 4) Efficiency
    fig4, ax4 = plt.subplots()
    ax4.plot(iters, eta_list, label='Efficiency')
    ax4.set_title('Efficiency')
    ax4.set_xlabel('Iteration [n]')
    ax4.set_ylabel(r"$\eta$")
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()

    fig5, ax5 = plt.subplots(1, 2, figsize=(12, 6))
    ax5[0].plot(iters, P_list, label='Power')
    ax5[0].set_title('Power')
    ax5[0].set_xlabel('Iteration [n]')
    ax5[0].set_ylabel('Power [W]')
    ax5[0].legend()
    ax5[0].grid(True)

    ax5[1].plot(iters, cons_list, label='Power Constraint')
    ax5[1].set_title('Power Constraint')
    ax5[1].set_xlabel('Iteration [n]')
    ax5[1].set_ylabel('Power [W]')
    ax5[1].legend()
    ax5[1].grid(True)
    fig5.tight_layout()

    fig6, ax6 = plt.subplots(2, 2, figsize=(12, 6))
    ax6[0, 0].plot(iters, v_list, label="v - FD")
    ax6[0, 0].legend()
    ax6[0, 0].set_xlabel('Iteration [n]')
    ax6[0, 0].set_ylabel('Velocity [m/s]')
    ax6[0, 0].grid()

    ax6[0, 1].plot(iters, a_list, label="a")
    ax6[0, 1].set_xlabel('Iteration [n]')
    ax6[0, 1].set_ylabel('[units]')
    ax6[0, 1].legend()
    ax6[0, 1].grid()

    ax6[1, 0].plot(iters, b_list, label="b")
    ax6[1, 0].set_xlabel('Iteration [n]')
    ax6[1, 0].set_ylabel('[units]')
    ax6[1, 0].legend()
    ax6[1, 0].grid()

    ax6[1, 1].plot(iters, c_list, label="c")
    ax6[1, 1].set_xlabel('Iteration [n]')
    ax6[1, 1].set_ylabel('[units]')
    ax6[1, 1].legend()
    ax6[1, 1].grid()

    fig6.suptitle("Design Parameters - FD")
    fig6.tight_layout()

    ## Plot optimal solution
    fig, ax = plt.subplots()
    contour3 = ax.contourf(heq.X, heq.Y, heq.u - 273)
    fig.colorbar(contour3, ax=ax)
    plt.show()