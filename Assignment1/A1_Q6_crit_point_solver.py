"""
Numerical Critical Point Solver for f(x,y) = x² + y³ - x²*y + x*y²
================================================================
This script implements numerical methods to find and classify critical points.
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """Function: f(x,y) = x² + y³ - x²*y + x*y²"""
    return x**2 + y**3 - x**2*y + x*y**2

def grad_f(x, y):
    """Gradient: ∇f = [2x - 2xy + y², 3y² - x² + 2xy]"""
    return np.array([2*x - 2*x*y + y**2, 3*y**2 - x**2 + 2*x*y])

def hessian_f(x, y):
    """Hessian matrix: H = [[2-2y, -2x+2y], [-2x+2y, 6y]]"""
    return np.array([[2 - 2*y, -2*x + 2*y], [-2*x + 2*y, 6*y]])

def newton_raphson_2d(f, grad_f, hessian_f, x0, y0, tol=1e-10, max_iter=100):
    """
    Newton-Raphson method for finding critical points in 2D
    
    Parameters:
    f: function
    grad_f: gradient function
    hessian_f: Hessian function
    x0, y0: initial guess
    tol: tolerance for convergence
    max_iter: maximum iterations
    
    Returns:
    x, y: critical point coordinates
    converged: boolean indicating convergence
    """
    x, y = x0, y0
    
    for i in range(max_iter):
        # Calculate gradient and Hessian
        grad = grad_f(x, y)
        hess = hessian_f(x, y)
        
        # Check if gradient is close to zero (critical point)
        if np.linalg.norm(grad) < tol:
            return x, y, True
        
        # Newton step: solve Hessian * delta = -gradient
        try:
            delta = np.linalg.solve(hess, -grad)
            x += delta[0]
            y += delta[1]
        except np.linalg.LinAlgError:
            # If Hessian is singular, use gradient descent
            alpha = 0.01  # step size
            print(f"Hessian is singular at ({x}, {y})")
            x -= alpha * grad[0]
            y -= alpha * grad[1]
    
    return x, y, False

def classify_critical_point(x, y):
    """
    Classify a critical point using the second derivative test
    
    Returns:
    classification: string describing the type of critical point
    det_H: determinant of Hessian
    tr_H: trace of Hessian
    """
    hess = hessian_f(x, y)
    det_H = np.linalg.det(hess)
    tr_H = np.trace(hess)
    eigenvals = np.linalg.eigvals(hess)
    
    if det_H > 0:
        if tr_H > 0:
            return "Local Minimum", det_H, tr_H, eigenvals
        else:
            return "Local Maximum", det_H, tr_H, eigenvals
    elif det_H < 0:
        return "Saddle Point", det_H, tr_H, eigenvals
    else:
        return "Degenerate", det_H, tr_H, eigenvals

def find_all_critical_points():
    """
    Find all critical points using multiple initial guesses
    """
    print("NUMERICAL CRITICAL POINT SOLVER")
    print("=" * 50)
    print("Function: f(x,y) = x² + y³ - x²*y + x*y²")
    print("Gradient: ∇f = [2x - 2xy + y², 3y² - x² + 2xy]")
    print("Hessian: H = [[2-2y, -2x+2y], [-2x+2y, 6y]]")
    print()
    
    # Multiple initial guesses to find all critical points
    initial_guesses = [
        (0.0, 0.0),   # Origin
        (0.5, 0.5),   # Positive quadrant
        (-0.5, 0.5),  # Negative x, positive y
        (0.5, -0.5),  # Positive x, negative y
        (-0.5, -0.5), # Negative quadrant
        (1.0, 1.0),   # Far positive
        (-1.0, 1.0),  # Far negative x, positive y
        (1.0, -1.0),  # Far positive x, negative y
        (-1.0, -1.0), # Far negative
        (0.0, 1.0),   # On y-axis
        (1.0, 0.0),   # On x-axis
        (-1.0, 0.0),  # On x-axis
        (0.0, -1.0),  # On y-axis
    ]
    
    critical_points = []
    tolerance = 1e-6
    
    print("Searching for critical points...")
    print("-" * 40)
    
    for i, (x0, y0) in enumerate(initial_guesses):
        x, y, converged = newton_raphson_2d(f, grad_f, hessian_f, x0, y0, tol=tolerance)
        
        if converged:
            # Check if this point is already found (within tolerance)
            is_duplicate = False
            for existing_x, existing_y, _, _, _, _, _ in critical_points:
                if abs(x - existing_x) < tolerance and abs(y - existing_y) < tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                classification, det_H, tr_H, eigenvals = classify_critical_point(x, y)
                f_val = f(x, y)
                critical_points.append((x, y, classification, det_H, tr_H, eigenvals, f_val))
                print(f"Found: ({x:.6f}, {y:.6f}) - {classification}")
    
    print(f"\nTotal unique critical points found: {len(critical_points)}")
    print()
    
    return critical_points

def plot_results(critical_points):
    """
    Create visualization with critical points marked
    """
    # Create a grid for the surface plot
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    
    # 3D Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    
    # Mark critical points
    colors = {'Local Minimum': 'red', 'Local Maximum': 'blue', 'Saddle Point': 'orange', 'Degenerate': 'purple'}
    markers = {'Local Minimum': 'o', 'Local Maximum': 's', 'Saddle Point': '^', 'Degenerate': 'd'}
    
    for x, y, classification, det_H, tr_H, eigenvals, f_val in critical_points:
        color = colors.get(classification, 'black')
        marker = markers.get(classification, 'o')
        ax1.scatter([x], [y], [f_val], color=color, s=100, marker=marker, 
                   label=f'{classification}', edgecolors='black', linewidth=1)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Surface Plot with Critical Points')
    
    # 2D Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, colors='black', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Mark critical points on contour plot
    for x, y, classification, det_H, tr_H, eigenvals, f_val in critical_points:
        color = colors.get(classification, 'black')
        marker = markers.get(classification, 'o')
        ax2.scatter(x, y, color=color, s=100, marker=marker, 
                   label=f'{classification}', edgecolors='black', linewidth=1)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot with Critical Points')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def print_detailed_results(critical_points):
    """
    Print detailed numerical results
    """
    print("\nDETAILED NUMERICAL RESULTS")
    print("=" * 60)
    
    for i, (x, y, classification, det_H, tr_H, eigenvals, f_val) in enumerate(critical_points, 1):
        print(f"Critical Point {i}:")
        print(f"  Coordinates: ({x:.8f}, {y:.8f})")
        print(f"  Function value: f({x:.6f}, {y:.6f}) = {f_val:.8f}")
        print(f"  Hessian determinant: det(H) = {det_H:.8f}")
        print(f"  Hessian trace: tr(H) = {tr_H:.8f}")
        print(f"  Eigenvalues: λ₁ = {eigenvals[0]:.8f}, λ₂ = {eigenvals[1]:.8f}")
        print(f"  Classification: {classification}")
        print(f"  Gradient magnitude: ||∇f|| = {np.linalg.norm(grad_f(x, y)):.2e}")
        print()

# Main execution
if __name__ == "__main__":
    # Find all critical points numerically
    critical_points = find_all_critical_points()
    
    # Print detailed results
    print_detailed_results(critical_points)
    
    # Create visualizations
    plot_results(critical_points)
    
    # Analytical analysis
    print("\nANALYTICAL ANALYSIS")
    print("=" * 50)
    print("Setting gradient to zero:")
    print("∂f/∂x = 2x - 2xy + y² = 0")
    print("∂f/∂y = 3y² - x² + 2xy = 0")
    print()
    print("From ∂f/∂x = 0: 2x(1-y) + y² = 0")
    print("Case 1: x = 0 → y² = 0 → y = 0 → Critical point: (0,0)")
    print("Case 2: x ≠ 0 → 2(1-y) + y²/x = 0")
    print()
    print("From ∂f/∂y = 0: 3y² - x² + 2xy = 0")
    print("At (0,0): ∂f/∂y = 0 ✓")
    print()
    print("Let's check other cases...")
    print("If y = 2/3, then from ∂f/∂x = 0: 2x(1/3) + 4/9 = 0 → x = -2/3")
    print("Checking ∂f/∂y at (-2/3, 2/3): 3(4/9) - 4/9 + 2(-2/3)(2/3) = 12/9 - 4/9 - 8/9 = 0 ✓")
    print("So (-2/3, 2/3) is also a critical point.")
    