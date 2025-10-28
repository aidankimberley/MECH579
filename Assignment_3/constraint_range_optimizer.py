import numpy as np
import jax.numpy as jnp
import scipy.optimize
import matplotlib.pyplot as plt
import brequet_range_optimizer as breg_opt
import SQP
from SQP import SQP, quadratic_penalty_inequality

#%%
#constants
eps = 1e-10
e=0.8
CD0=0.0083
AR = 10
S = 100 #m^2
Wf = 162400 #kg
Wfuel = 146571 #kg
At = 1.3295 #m^2
FAR =0.1
### SPEED OF SOUND ###
c=341
######################
Wi = Wf + Wfuel
fuel_fraction = 0.75
Wf_75 = Wf + fuel_fraction*Wfuel
Lift = (Wf_75)*9.81

#%%
#constraints
c_1 = lambda x: 2e4 - x[1] #20000 max altitude (inequality: h ≤ 20000)
c_2 = lambda x: 150 - x[0] #150 max m/s (inequality: V ≤ 150)
c_ineq_array = [c_1, c_2]  # Inequality constraints

# For scipy compatibility, ensure constraints work with numpy arrays
def c_1_numpy(x):
    return 2e4 - x[1]

def c_2_numpy(x):
    return 150 - x[0]
lamb0 = 0.0 #initial guess for Lagrange multipliers
x0 = jnp.array([130.0, 12000.0])  # Start closer to expected optimal

#%%
#run optimization
print("Running Custom SQP...")
'''
xk, lamb, x_path, gradient_norms = SQP(x0, lamb0, breg_opt.brequet_range_wrapper, 
                                       c_eq_array=None, c_ineq_array=c_ineq_array, 
                                       eps=1e-4, max_iter=500, track_data=True, BFGS=True)
'''
xk, x_path, gradient_norms = quadratic_penalty_inequality(x0, breg_opt.brequet_range_wrapper, c_ineq_array, eps=1e-2, max_iter=12, track_data=True, verbose=True)
print(f"Optimal solution: {xk}")
print(f"Converged in {len(gradient_norms)} iterations!")
print(f"Constraint violations:")
print(f"  Altitude constraint (h ≤ 20000): {xk[1] - 20000}")
print(f"  Velocity constraint (V ≤ 150): {xk[0] - 150}")
print(f"Final objective value: {breg_opt.brequet_range_wrapper(xk)}")
constraints = [
    {'type': 'ineq', 'fun': c_1_numpy},
    {'type': 'ineq', 'fun': c_2_numpy}
]

# Scipy comparison - convert JAX arrays to numpy
print("\n" + "="*50)
print("RUNNING SCIPY SLSQP FOR COMPARISON")
print("="*50)

def brequet_range_wrapper_numpy(x):
    """Numpy-compatible wrapper for scipy"""
    return float(breg_opt.brequet_range_wrapper(jnp.array(x)))

def brequet_gradient_numpy(x):
    """Compute gradient using JAX"""
    import jax
    grad_func = jax.grad(breg_opt.brequet_range_wrapper)
    grad = grad_func(jnp.array(x))
    return np.array(grad, dtype=np.float64)

# Convert to numpy arrays (use same initial point as custom SQP)
x0_numpy = np.array(x0, dtype=np.float64)

# Run scipy optimization with gradients
xk_scipy = scipy.optimize.minimize(
    brequet_range_wrapper_numpy, 
    x0_numpy, 
    method='SLSQP',
    jac=brequet_gradient_numpy,
    constraints=constraints, 
    options={'disp': True, 'maxiter': 200, 'ftol': 1e-9}
)

print("\n" + "="*50)
print("SCIPY RESULTS")
print("="*50)
print(f"Scipy Optimal solution: V = {xk_scipy.x[0]:.2f} m/s, h = {xk_scipy.x[1]:.2f} m")
print(f"Scipy Converged: {xk_scipy.success}")
print(f"Scipy Iterations: {xk_scipy.nit}")
print(f"Scipy Constraint violations:")
print(f"  Altitude constraint (h ≤ 20000): {xk_scipy.x[1] - 20000:.2f} m")
print(f"  Velocity constraint (V ≤ 150): {xk_scipy.x[0] - 150:.2f} m/s")
print(f"Scipy Maximum Range: {-brequet_range_wrapper_numpy(xk_scipy.x):.0f} m")
print("="*50)

print("\n" + "="*50)
print("COMPARISON: CUSTOM SQP vs SCIPY SLSQP")
print("="*50)
print(f"{'Metric':<30} {'Custom SQP':>15} {'Scipy SLSQP':>15}")
print("-"*62)
print(f"{'Optimal Velocity (m/s)':<30} {xk[0]:>15.2f} {xk_scipy.x[0]:>15.2f}")
print(f"{'Optimal Altitude (m)':<30} {xk[1]:>15.2f} {xk_scipy.x[1]:>15.2f}")
print(f"{'Maximum Range (m)':<30} {-breg_opt.brequet_range_wrapper(xk):>15.0f} {-brequet_range_wrapper_numpy(xk_scipy.x):>15.0f}")
print(f"{'Iterations':<30} {len(gradient_norms):>15} {xk_scipy.nit:>15}")
print(f"{'V constraint (V-150)':<30} {xk[0]-150:>15.2f} {xk_scipy.x[0]-150:>15.2f}")
print(f"{'h constraint (h-20000)':<30} {xk[1]-20000:>15.2f} {xk_scipy.x[1]-20000:>15.2f}")
print("="*62)

# Calculate differences
v_diff = abs(xk[0] - xk_scipy.x[0])
h_diff = abs(xk[1] - xk_scipy.x[1])
range_diff = abs(-breg_opt.brequet_range_wrapper(xk) - (-brequet_range_wrapper_numpy(xk_scipy.x)))

print(f"\nAbsolute Differences:")
print(f"  Velocity: {v_diff:.2f} m/s")
print(f"  Altitude: {h_diff:.2f} m")
print(f"  Range: {range_diff:.0f} m ({range_diff/-breg_opt.brequet_range_wrapper(xk)*100:.2f}%)")
print("="*50)
#%%
#plot results
from SQP import plot_convergence, plot_contour_with_path
plot_convergence(gradient_norms, algorithm_name="SQP", save_path=r"/Users/aidan1/Documents/McGill/MECH597/Assignment_3/convergence.png")
# plot_contour_with_path(
#     breg_opt.brequet_range_wrapper,
#     c_array,
#     x_path,
#     x_bounds=(0, 160),y_bounds=(0, 25000),
#     algorithm_name="SQP",
#     save_path=r"/Users/aidan1/Documents/McGill/MECH597/Assignment_3/range_contour.png"
# )
# Contour plot temporarily disabled for faster results
# breg_opt.plot_brequet_contour_with_path(breg_opt.brequet_range_wrapper, x_path, x_range=(10,300), y_range=(0,25000),show_plot=True)
print("\nPlot generation skipped for faster results.")