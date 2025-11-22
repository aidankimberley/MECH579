import numpy as np

def constraint_1(x):
    #c_val = 1.0 - x[:,0] - x[:,1];
    c_val = -x[:,1]+2e4
    return c_val;


def constraint_2(x):
    #c_val = 1.0 - x[:,0]**2 - x[:,1]**2;
    c_val = -x[:,0]+540/3.6 #m/s
    return c_val;
