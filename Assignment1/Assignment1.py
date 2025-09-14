'''
Assignment 1
Aidan Kimberley
09/10/2025
'''
#%%imports
#imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%
#Bisection Method

f1 = lambda x: x**2 -4*x+4 - np.log(x) #function to find the root of
f = lambda x: x+1-2*np.sin(np.pi*x)
a = 0.5 #left endpoint
b = 1 #right endpoint
eps = 10e-5 #tolerance
x0 = a



def bisection_method(f, a, b, eps):
    """
    Bisection method to find root of function f in interval [a,b]
    Returns: root, iterations, errors, midpoints, function_values
    """
    errors = []
    midpoints = []
    function_values = []
    n = 0
    
    while np.abs(b - a) > eps:
        r = (a + b) / 2
        error = np.abs(b - a) / 2
        fx = f(r)
        midpoints.append(r)
        errors.append(error)
        function_values.append(fx)
        if f(r) * f(a) > 0:
            a = r
        else:
            b = r
        n += 1

    return r, n, errors, midpoints, function_values

root, n, errors, midpoints, function_values = bisection_method(f, a, b, eps)
print("Bisection Method:")
print(f"Root: {root}")
print(f"Number of iterations: {n}")
# Plot convergence
plt.figure()
plt.semilogy(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Bisection Method Convergence')
plt.grid(True, which="both", ls="--")
plt.show()

# Create table with selected iterations
n_total = len(midpoints)
indices = [0, n_total // 3, 2 * n_total // 3, n_total - 1]

table_data = []
for idx in indices:
    x_val = midpoints[idx]
    fx_val = function_values[idx]
    error_val = errors[idx]
    table_data.append([x_val, fx_val, error_val])

df = pd.DataFrame(table_data, columns=['x', 'f(x)', 'Error'])

print("\nBisection Method Table (selected iterations):")
print(df.to_string(index=False))


#%%
#Fixed Point Iteration
g1 = lambda x: 2 - np.sqrt(np.log(x))
g = lambda x: 1/np.pi*np.arcsin((x+1)/2)
g3 = lambda x: 2*np.sin(np.pi*x) - 1


def fixed_point_iteration(g, x0, eps):
    x = x0
    n=1
    errors = []
    midpoints = []
    function_values = []
    while np.abs(g(x) - x) > eps:
        error = np.abs(g(x) - x)
        midpoints.append(x)
        errors.append(error)
        function_values.append(g(x))
        x = g(x)
        n+=1
    return x, n, errors, midpoints, function_values

root, n, errors, midpoints, function_values = fixed_point_iteration(g, x0, eps)

print("Fixed Point Iteration:")
print(f"Root: {root}")
print(f"Number of iterations: {n}")

#plotting convergence
plt.figure()
plt.semilogy(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Fixed Point Iteration Convergence')
plt.grid(True, which="both", ls="--")
plt.show()

# Create table with selected iterations
n_total = len(midpoints)
indices = [0, n_total // 3, 2 * n_total // 3, n_total - 1]

table_data = []
for idx in indices:
    x_val = midpoints[idx]
    fx_val = function_values[idx]
    error_val = errors[idx]
    table_data.append([x_val, fx_val, error_val])

df = pd.DataFrame(table_data, columns=['x', 'f(x)', 'Error'])

print("\nFixed Point Iteration Table (selected iterations):")
print(df.to_string(index=False))

#%%
#Newton's Method
f_prime1 = lambda x: 2*x -4 -1/x
f_prime = lambda x: 1 - 2*np.pi*np.cos(np.pi*x)

def newtons_method(f, f_prime, x0, eps):
    x = x0
    n=1
    errors = []
    midpoints = []
    function_values = []
    
    # Store initial values
    midpoints.append(x)
    function_values.append(f(x))
    errors.append(np.abs(f(x)))
    
    while np.abs(f(x)) > eps:
        error = np.abs(f(x))
        errors.append(error)
        x = x - f(x)/f_prime(x)
        midpoints.append(x)
        function_values.append(f(x))
        n += 1
    return x, n, errors, midpoints, function_values

root, n, errors, midpoints, function_values = newtons_method(f, f_prime, x0, eps)
print("Newton's Method:")
print(f"Root: {root}")
print(f"Number of iterations: {n}")
#plotting convergence
plt.figure()
plt.semilogy(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Newton\'s Method Convergence')
plt.grid(True, which="both", ls="--")
plt.show()

# Create table with selected iterations
n_total = len(midpoints)
indices = [0, n_total // 3, 2 * n_total // 3, n_total - 1]

table_data = []
for idx in indices:
    x_val = midpoints[idx]
    fx_val = function_values[idx]
    error_val = errors[idx]
    table_data.append([x_val, fx_val, error_val])

df = pd.DataFrame(table_data, columns=['x', 'f(x)', 'Error'])

print("\nNewton's Method Table (selected iterations):")
print(df.to_string(index=False))

#%%
#Secant Method
x1 = x0 + 0.1

def secant_method(f, x0, x1, eps):
    x = x1
    x_prev = x0
    n=1
    errors = []
    midpoints = []
    function_values = []

    # Store initial values
    midpoints.append(x)
    function_values.append(f(x))
    errors.append(np.abs(f(x)))

    while np.abs(f(x)) > eps:
        f_prime = (f(x) - f(x_prev)) / (x - x_prev)
        x_new = x - f(x)/f_prime
        error = np.abs(f(x))
        midpoints.append(x_new)
        errors.append(error)
        function_values.append(f(x_new))
        x_prev = x
        x = x_new
        n+=1
    return x, n, errors, midpoints, function_values

root, n, errors, midpoints, function_values = secant_method(f, x0, x1, eps)
print("Secant Method:")
print(f"Root: {root}")
print(f"Number of iterations: {n}")
#plotting convergence
plt.figure()
plt.semilogy(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Secant Method Convergence')
plt.grid(True, which="both", ls="--")
plt.show()

# Create table with selected iterations
n_total = len(midpoints)
indices = [0, n_total // 3, 2 * n_total // 3, n_total - 1]

table_data = []
for idx in indices:
    x_val = midpoints[idx]
    fx_val = function_values[idx]
    error_val = errors[idx]
    table_data.append([x_val, fx_val, error_val])

df = pd.DataFrame(table_data, columns=['x', 'f(x)', 'Error'])

print("\nSecant Method Table (selected iterations):")
print(df.to_string(index=False))