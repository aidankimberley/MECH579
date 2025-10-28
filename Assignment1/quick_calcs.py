import numpy as np

def solve_ax_b(A, b):
    """Simple Ax=b solver"""
    A = np.array(A)
    b = np.array(b).flatten()
    x = np.linalg.solve(A, b)
    return x

# Example usage
if __name__ == "__main__":
    # Define your matrix A and vector b here
    A = [[4, 4,0], [4, -6,0], [0,0,2]]
    b = [4, 0,4]
    
    print(f"A = {A}")
    print(f"b = {b}")
    
    x = solve_ax_b(A, b)
    print(f"Solution x = {x}")
    
    # Verify: Ax should equal b
    verification = np.dot(A, x)
    print(f"Verification Ax = {verification}")