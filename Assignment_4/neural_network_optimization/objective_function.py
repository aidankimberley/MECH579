import os
import sys
import numpy as np

# Ensure the parent directory (Assignment_4) is on sys.path so we can import
# modules like brequet_range_equation that live alongside this package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import brequet_range_equation

def brequet_range_equation_wrapper(x):
    S=100#m^2
    fuel_percentage = 0.75
    return -brequet_range_equation.range(x[:,0],x[:,1],fuel_percentage,S)


def obj_func(x):
    #f_val = (1.0-x[:,0])**2 + 100.0*(x[:,1] - x[:,0]**2)**2;
    return brequet_range_equation_wrapper(x);
