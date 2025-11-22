import brequet_range_equation
import numpy as np
import scipy.optimize as optimize
#x: [ 1.933e+02  1.486e+04]

def brequet_range_equation_wrapper(x):
    S=100#m^2
    fuel_percentage = 0.75
    return -brequet_range_equation.range(x[:,0],x[:,1],fuel_percentage,S)
    #     # Convert to numpy array and flatten to handle both 1D and 2D inputs
    # x = np.array(x).flatten()
    
    # # Now x is always 1D with shape (2,)
    # v = x[0]
    # h = x[1]
    # return -brequet_range_equation.range(v, h, fuel_percentage, S)


def obj_func(x):
    #f_val = (1.0-x[:,0])**2 + 100.0*(x[:,1] - x[:,0]**2)**2;
    return brequet_range_equation_wrapper(x);

print(obj_func(np.array([[182, 14235]])))

print(obj_func(np.array([[192, 14805]])))

print(obj_func(np.array([[186.5, 13683]])))

x0 = np.array([200.0, 1500.0], dtype=np.float64)

result = optimize.minimize(obj_func, x0, method='SLSQP')
print(result)