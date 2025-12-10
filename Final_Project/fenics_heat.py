"""
FEniCS reimplementation of the 2D heat conduction + Robin BC problem.

Physics mirrors the original finite-difference model:
- Steady-state conduction in a 0.04 m x 0.04 m plate of thickness H = 0.04 m.
- Volumetric heat generation q(x, y) = a*x + b*y + c [W/m^3].
- Natural convection on all boundaries (temperature-dependent h_boundary).
- Forced convection on the "top" via a fan: h_top depends on velocity v and x.

Note: Natural convection makes the weak form nonlinear through h(T). We solve
with a Newton iteration via FEniCS' NonlinearVariationalSolver.
"""

from dolfin import (  # type: ignore
    RectangleMesh,
    Point,
    FunctionSpace,
    Function,
    TestFunction,
    TrialFunction,
    Measure,
    Constant,
    Expression,
    dot,
    grad,
    conditional,
    gt,
    assemble,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    derivative,
)
import numpy as np


# Geometry
Lx = 0.04
Ly = 0.04
H = 0.04  # thickness used to convert volumetric source to total Watts

# Material properties (silicon, from original script)
k_si = 149.0
rho_si = 2323.0
cp_si = 19.789 / 28.085 * 1000.0  # J/(kg K)

# External air properties
ext_k = 0.02772
ext_Pr = 0.7215
ext_nu = 1.506e-5
ext_T = 273.0 + 20.0

# Fan efficiency model (kept for parity)
fan_efficiency = lambda v: -0.002 * v ** 2 + 0.08 * v


def make_h_top_expression(v: float):
    """Piecewise Nusselt correlation for forced convection."""
    # Re = v * x / nu; use x-coordinate as characteristic length
    return Expression(
        "Re < Recrit ? 0.332*pow(Re,0.5)*pow(Pr, 1.0/3.0) : 0.0296*pow(Re,0.8)*pow(Pr, 1.0/3.0)",
        degree=2,
        Recrit=5e5,
        Pr=ext_Pr,
        nu=ext_nu,
        v=v,
        Re=0.0,  # placeholder, see below
    )


def solve_temperature(v: float, a: float, b: float, c: float, nx: int = 50, ny: int = 50):
    """
    Solve steady-state temperature field for given design variables.

    Returns
    -------
    T : Function
        Temperature field
    stats : dict
        {"max_T": float, "eta": float, "total_heat_W": float}
    """
    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)
    V = FunctionSpace(mesh, "P", 1)

    # Trial/unknown and test
    T = Function(V)
    dT = TrialFunction(V)
    v_test = TestFunction(V)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    # Source term q(x,y) = a*x + b*y + c
    q_expr = Expression("a*x[0] + b*x[1] + c", degree=2, a=a, b=b, c=c)

    # Forced convection (fan) on boundaries: h_top depends on x
    h_top = Expression(
        "Re < Recrit ? 0.332*pow(Re,0.5)*pow(Pr, 1.0/3.0) : 0.0296*pow(Re,0.8)*pow(Pr, 1.0/3.0)",
        degree=2,
        Recrit=5e5,
        Pr=ext_Pr,
        v=v,
        nu=ext_nu,
    )

    # In FEniCS Expressions, we can refer to x[0]; Re is evaluated at run time
    h_top.Re = None  # silence mypy; FEniCS handles attribute dynamically

    # Characteristic length: use min element size as a proxy (consistent with FD dx)
    h_char = Constant(min(Lx / nx, Ly / ny))

    # Natural convection h_boundary(T) (nonlinear in T)
    beta = 1.0 / ((T + Constant(ext_T)) / 2.0)
    rayleigh = (
        9.81 * beta * (T - Constant(ext_T)) * h_char ** 3 / (ext_nu ** 2) * ext_Pr
    )
    nusselt_nat = (0.825 + (0.387 * rayleigh ** (1.0 / 6.0)) /
                   (1.0 + (0.492 / ext_Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    h_nat = nusselt_nat * ext_k / h_char

    # Weak form: k * grad T · grad v + (h_nat + h_top)*(T - Text) on boundary = q on domain
    a_form = k_si * dot(grad(T), grad(v_test)) * dx + (h_nat + h_top) * T * v_test * ds
    L_form = q_expr * v_test * dx + (h_nat + h_top) * Constant(ext_T) * v_test * ds

    F = a_form - L_form
    J = derivative(F, T, dT)

    problem = NonlinearVariationalProblem(F, T, bcs=[], J=J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1e-10
    prm["newton_solver"]["relative_tolerance"] = 1e-9
    prm["newton_solver"]["maximum_iterations"] = 40
    solver.solve()

    # Post-process
    T_array = T.vector().get_local()
    max_T = float(np.max(T_array))

    total_heat = float(assemble(q_expr * dx)) * H  # integrate over volume
    eta = float(fan_efficiency(v))

    return T, {"max_T": max_T, "eta": eta, "total_heat_W": total_heat}


def run_example():
    # Example inputs (from the original script)
    v0 = 15.0
    a0 = 0.0
    b0 = 0.0
    c0 = 156250.0
    T, stats = solve_temperature(v0, a0, b0, c0)
    print("Max T [K]:", stats["max_T"])
    print("Fan efficiency:", stats["eta"])
    print("Total heat [W]:", stats["total_heat_W"])


if __name__ == "__main__":
    run_example()

