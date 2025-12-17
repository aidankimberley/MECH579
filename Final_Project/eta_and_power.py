# %%
# Imports
import numpy as np 
import matplotlib.pyplot as plt


# %%
# Plotting parameters

# Get the directory where this script is located
# script_dir = pathlib.Path(__file__).parent
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25

# %% 
# Functions
a = -50.0
b = -50.0
c = 153320.0
gen = lambda x, y: a * x + b * y + c

eff = lambda v: -0.002*v**2 + 0.08*v

# %% 

v = np.linspace(0,40, 1000)

fig, ax = plt.subplots()
fig.set_size_inches(height * gr, height, forward=True)
ax.plot(v, eff(v), label=r'$\eta(v)$')
ax.set_xlabel('v [m/s]')
ax.set_ylabel('efficiency') 
ax.legend()
fig.tight_layout()

CPU_X = 0.04  # m
CPU_Y = 0.04  # m
N = 25

x_axis = np.linspace(0, CPU_X, N)
y_axis = np.linspace(0, CPU_Y, N)
X, Y = np.meshgrid(x_axis, y_axis, indexing='ij')

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, gen(X,Y), levels=20, cmap="viridis")
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label("Power Density [W/m^3]")
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
fig.tight_layout()

# %%
