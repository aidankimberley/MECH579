import scipy
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import time
import brequet_range_equation_jax as brequet_range_equation

# Unconstrained optimization for comparison with neural network method
x0 = np.array([200.0, 1500.0], dtype=np.float64)

#grad range
def range_wrapper(x,fuel_percentage=0.75):
    S=100#m^2
    fuel_percentage = fuel_percentage
    return -brequet_range_equation.range(x[0],x[1],fuel_percentage,S)

jax_grad = jax.grad(range_wrapper)
def jac(x):
    # Ensure input is float64 before passing to JAX
    x_float = np.array(x, dtype=np.float64)
    return np.array(jax_grad(jnp.array(x_float, dtype=jnp.float64)), dtype=np.float64)

iterates = []
range_values = []
sqp_iteration_times = []

def callback(xk):
    # Track time when callback is called (this marks the end of an iteration)
    sqp_iteration_times.append(time.time())
    # Ensure we store as float64 array
    iterates.append(np.array(xk, dtype=np.float64))
    # Track range value (negative because we're minimizing -range)
    range_val = -range_wrapper(xk)
    range_values.append(range_val)

# Add initial condition to iterates before optimization starts
iterates.append(np.array(x0))
range_values.append(-range_wrapper(x0))

# Start timing before optimization
sqp_start_time = time.time()
sqp_iteration_times.append(sqp_start_time)  # Mark start time

# Run unconstrained optimization with callback
result = scipy.optimize.minimize(range_wrapper, x0=x0, jac=jac, method = 'SLSQP', callback=callback)
sqp_total_time = time.time() - sqp_start_time

# Calculate time per iteration from the times between consecutive callbacks
sqp_time_per_iteration = []
for i in range(1, len(sqp_iteration_times)):
    iter_time = sqp_iteration_times[i] - sqp_iteration_times[i-1]
    sqp_time_per_iteration.append(iter_time)
    
# If we only have start time (no callbacks), calculate from total time
if len(sqp_time_per_iteration) == 0 and result.nit > 0:
    avg_time_per_iter = sqp_total_time / result.nit
    sqp_time_per_iteration = [avg_time_per_iter] * result.nit

print("="*60)
print("UNCONSTRAINED SQP OPTIMIZATION")
print("="*60)
print(result)
print("SQP Total optimization time: {:.4f} s".format(sqp_total_time))
print("SQP Number of iterations: {}".format(len(sqp_time_per_iteration)))
print("SQP Average time per iteration: {:.4f} s ({:.2f} ms)".format(
    np.mean(sqp_time_per_iteration) if len(sqp_time_per_iteration) > 0 else 0,
    np.mean(sqp_time_per_iteration)*1000 if len(sqp_time_per_iteration) > 0 else 0))

# Save unconstrained SQP timing data
sqp_iterations = np.arange(len(range_values))
np.savez('plots/sqp_unconstrained_timing_data.npz', 
         iterations=sqp_iterations, 
         objective=range_values, 
         time_per_iter=sqp_time_per_iteration,
         total_time=sqp_total_time)

