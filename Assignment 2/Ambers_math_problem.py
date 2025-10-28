import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import time

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)

Doors = np.ones(100)  # 1 is closed, -1 is open

# Create a 10x10 grid for displaying doors
def draw_doors(doors, iteration):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title showing current iteration
    title = f'Iteration {iteration}: Toggling every {iteration} door(s)'
    if iteration == 0:
        title = 'Initial State: All Doors Closed'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Draw doors in a 10x10 grid
    for idx in range(100):
        row = 9 - (idx // 10)  # Start from top
        col = idx % 10
        
        # Color: green for closed (1), red for open (-1)
        color = 'green' if doors[idx] == 1 else 'red'
        
        # Draw rectangle for door
        rect = Rectangle((col, row), 0.9, 0.9, 
                         facecolor=color, 
                         edgecolor='black', 
                         linewidth=2)
        ax.add_patch(rect)
        
        # Add door number
        ax.text(col + 0.45, row + 0.45, str(idx + 1), 
               ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', label='Closed'),
        Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black', label='Open')
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(0, -0.02), ncol=2, fontsize=12)
    
    plt.draw()
    plt.pause(0.5)  # Pause for half a second

# Show initial state
draw_doors(Doors, 0)
time.sleep(1)  # Longer pause for initial state

# Animate the door toggling process
for i in range(1, 101):
    for j in range(100):
        if (j + 1) % i == 0:
            Doors[j] *= -1
    
    draw_doors(Doors, i)

# Final summary
final_open = np.where(Doors == -1)[0] + 1
final_closed = np.where(Doors == 1)[0] + 1

print(f"\nFinal Results:")
print(f"Open doors: {final_open}")
print(f"Closed doors: {final_closed}")
print(f"\nNumber of open doors: {len(final_open)}")
print(f"Number of closed doors: {len(final_closed)}")

plt.show()
