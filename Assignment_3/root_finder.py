#Find root of function
import numpy as np

f = lambda x: -200*x**3 -300*x**2 + +99*x +101
#Find root of function using bisection method

a = 0
b = 1
eps = 10e-5

def bisection_method(f, a, b, eps):
    while np.abs(b - a) > eps:
        c = (a + b) / 2
        if f(c) == 0:
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c

root = bisection_method(f, a, b, eps)
print(f"Root found at x = {root}")
a=-1
b=0
root = bisection_method(f, a, b, eps)
print(f"Root found at x = {root}")
a=1
b=2
root = bisection_method(f, a, b, eps)
print(f"Root found at x = {root}")
a=-2
b=-1
root = bisection_method(f, a, b, eps)
print(f"Root found at x = {root}")