#Graphing 

import numpy as np
import matplotlib.pyplot as plt

# Define the function
f = lambda x, y: x**3 - x + y**3 - y

# Create a grid of x and y values
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plot the surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Surface plot of $z = x^3 - x + y^3 - y$')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
