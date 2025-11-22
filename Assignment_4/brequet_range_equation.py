## Essential Imports
import numpy as np # for pi and atan
import matplotlib.pyplot as plt # for plotting contour


def density(h):
    ## Calculates the density at a height h
    # Inputs
    #   h (m), altitude: float
    # Output:
    #   density (kg/m^3): float
    return 1.2*(1-h*0.0065/288)**5.26

def mass_fuel_rate(v,h):
    ## Calculates the mass fuel rate using the mass flow rate of air hitting the turbine
    # Inputs:
    #   v (m/s), velocity: float
    #   h (m), altitude: float
    # Output:
    #   mass_fuel_rate (kg/s): float
    turbine_area = np.pi*0.92**2/2
    FAR = 1E-1
    return density(h)*v*turbine_area*FAR

def C_L(v,h,weight,S):
    ## Calculates the coefficient of lift
    # Inputs:
    #   v (m/s), velocity: float
    #   h (m), altitude: float
    #   weight (kg), weight of the aircraft at current time: float
    #   S (m^2), platform area: float
    # Output:
    #   coefficient of lift: float
    return 9.81*weight/(0.5*density(h)*v**2*S)

def C_Dw(v,h):
    ## Calculates the wave drag of the aircraft
    # Inputs:
    #   v (m/s), velocity: float
    #   h (m), altitude: float
    # Output:
    #   wave drag coefficient: float
    # Use np.arctan instead of np.atan (np.atan does not exist, np.arctan is correct)
    return 10*(np.arctan(10*((v/(343*0.7))**2-1)) + np.pi/2)

def C_D(v,h,weight,S):
    ## Calculates the coefficient of drag
    # Inputs:
    #   v (m/s), velocity: float
    #   h (m), altitude: float
    #   weight (kg), weight of the aircraft at current time: float
    #   S (m^2), platform area: float
    # Output:
    #   coefficient of drag: float
    e = 0.8
    AR = 10
    coeff_drag = 0.5/60
    coeff_drag += C_L(v,h,weight,S)**2/(np.pi*e*AR)
    return coeff_drag + C_Dw(v,h)

def ct(v,h,weight,S):
    ## Calculates the specific fuel consumption
    # Inputs:
    #   v (m/s), velocity: float
    #   h (m), altitude: float
    #   weight (kg), weight of the aircraft at current time: float
    #   S (m^2), platform area: float
    # Output:
    #   specific fuel consumption: float
    ct_value = mass_fuel_rate(v,h)/(0.5*density(h)*v**2*S*C_D(v,h,weight,S))
    return ct_value + 10**(-5)

def range(v,h,fuel_percentage,S):
    ##Calculates the range of the aircraft
    # Inputs:
    #   v (m/s), velocity: float
    #   h (m), altitude: float
    #   weight (kg), weight of the aircraft at current time: float
    #   S (m^2), platform area: float
    # Output:
    #   range (km): float
    W_f = 162400
    W_fuel = 0.8 * 183214
    weight = W_f + W_fuel*fuel_percentage
    # You can change this to whatever you want, it will only change the total range, not the optimal location
    W_i = W_f + W_fuel
    total_range = v/ct(v,h,weight,S)*C_L(v,h,weight,S)/C_D(v,h,weight,S)*np.log(W_i/W_f)
    return total_range / 1E3

""" Example usage for Question 2.1 in Assignment 2 
## Constants
S = 100
W_f = 162400
W_fuel = 0.8 * 183214

## Change this section
fuel_percentage = 0.75
weight_used = W_f + W_fuel*fuel_percentage

## Example for Assignment 2 Question 2.1
v_axis = np.linspace(0.1,300,1000) # divide by zero error at 0.0, setting to 0.1
h_axis = np.linspace(0.1,25000,1000)

V,H = np.meshgrid(v_axis,h_axis)

range_mesh = range(V,H,weight_used,S)

fig, ax = plt.subplots()#subplot_kw={'projection': '2d'})
contour_obj = ax.contourf(V,H,range_mesh,cmap=plt.get_cmap('jet'),levels = 20)
fig.colorbar(contour_obj,ax=ax,label="Range (km)")
ax.set_xlabel("v (m/s)")
ax.set_ylabel("h (m)")
plt.show()
# Test to ensure that the max is near the correct location (not exact!)
i,j = np.where(j    max(range_mesh)==range_mesh)
print(f"Velocity: {V[i,j]} m/s, Height: {H[i,j]} m, Range: {range_mesh[i,j]} km")
"""