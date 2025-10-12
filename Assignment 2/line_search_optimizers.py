#%%
#imports
import numpy as np
import matplotlib.pyplot as plt



#minimize the rosenbrock function
#1 steepest descent
#2 nonlinear conjugate gradient
#3 quasi newton
#4 newtons method
#Use backtracking linesearch for all methods

def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

grad_rosenbrock = lambda x: np.array([-2*(1-x[0]) - 400*(x[1]-x[0]**2)*x[0], 200*(x[1]-x[0]**2)])

hess_rosenbrock = lambda x: np.array([[2-400*(x[1]-x[0]**2) + 800*x[0]**2, -400*x[0]],
                                         [-400*x[0],                        200]])

def alpha_backtracking(func,grad_func,pk,xk,alpha=0.5,c=0.5,rho=0.8,max_iter=100):
    i=0
    while func(xk+alpha*pk) >= func(xk) + c*alpha*grad_func(xk).T@pk:
        alpha = rho*alpha
        i+=1
        if i > max_iter:
            print(f"Alpha backtracking did not converge in {max_iter} iterations at {xk}")
            return alpha
    return alpha

def steepest_descent(x0,func,grad_func,tol=1e-6,max_iter=30000):
    xk = x0
    grad_history = [np.linalg.norm(grad_func(xk))]
    path_history = [xk.copy()]
    for i in range(max_iter):
        if i % 1000 == 0 and i > 0:
            print(f"Iteration {i}: xk = {xk}, grad norm = {np.linalg.norm(grad_func(xk))}")
        pk = -grad_func(xk)
        grad_history.append(np.linalg.norm(grad_func(xk)))
        alpha = alpha_backtracking(func,grad_func,pk,xk)
        xk = xk + alpha*pk
        path_history.append(xk.copy())
        if np.linalg.norm(grad_func(xk)) < tol:
            print(f"Steepest descent converged in {i} iterations at {xk}")
            return xk,i,grad_history,path_history
    print(f"Steepest descent did not converge in {max_iter} iterations at {xk}")
    return xk,i,grad_history,path_history


def nonlin_conj_grad(x0, func, grad_func, tol=1e-6, max_iter=30000):
    xk = x0
    pk = -grad_func(xk)
    grad_history = [np.linalg.norm(grad_func(xk))]
    path_history = [xk.copy()]
    for i in range(max_iter):
        if i % 1000 == 0 and i > 0:
            print(f"Iteration {i}: xk = {xk}, grad norm = {np.linalg.norm(grad_func(xk))}")
        gk = grad_func(xk)
        alpha = alpha_backtracking(func,grad_func,pk,xk)
        xk = xk + alpha*pk
        path_history.append(xk.copy())
        gk1 = grad_func(xk)
        bk = (gk1.T@gk1)/(gk.T@gk)
        pk = -gk1 + bk*pk
        grad_history.append(np.linalg.norm(grad_func(xk)))
        if np.linalg.norm(grad_func(xk)) < tol:
            print(f"Nonlinear conjugate gradient converged in {i} iterations at {xk}")
            return xk,i,grad_history,path_history
    print(f"Nonlinear conjugate gradient did not converge in {max_iter} iterations at {xk}")
    return xk,i,grad_history,path_history


def quasi_newton_bfgs(x0, func, grad_func, tol=1e-6, max_iter=3000):
    #H represents inverse of the Hessian
    xk = x0
    gk = grad_func(xk)
    Hk = np.eye(2)
    grad_history = [np.linalg.norm(gk)]
    path_history = [xk.copy()]
    for i in range(max_iter):
        if i % 1000 == 0 and i > 0:
            print(f"Iteration {i}: xk = {xk}, grad norm = {np.linalg.norm(gk)}")
        pk = -Hk@gk
        alpha = alpha_backtracking(func,grad_func,pk,xk)
        #alpha = 0.05
        xk = xk + alpha*pk
        path_history.append(xk.copy())
        gk1 = grad_func(xk)
        del_gk = gk1 - gk
        del_xk = alpha*pk
        
        # Check curvature condition: s_k^T y_k > 0
        # Only update H if this condition is satisfied
        sk_yk = del_xk.T @ del_gk
        if sk_yk > 1e-10:  # Add small threshold to avoid numerical issues
            # BFGS update
            rho = 1.0 / sk_yk
            Vk = np.eye(len(xk)) - rho * np.outer(del_xk, del_gk)
            Hk = Vk @ Hk @ Vk.T + rho * np.outer(del_xk, del_xk)
        
        gk=gk1
        grad_history.append(np.linalg.norm(grad_func(xk)))
        if np.linalg.norm(grad_func(xk)) < tol:
            print(f"Quasi-Newton BFGS converged in {i} iterations at {xk}")
            return xk,i,grad_history,path_history
    print(f"Quasi-Newton BFGS did not converge in {max_iter} iterations at {xk}")
    return xk,i,grad_history,path_history


def newtons_method(x0, func, grad_func, hess_func, tol=1e-6, max_iter=30000):
    xk =x0
    gk = grad_func(xk)
    grad_history = [np.linalg.norm(gk)]
    path_history = [xk.copy()]
    for i in range(max_iter):
        if i % 1000 == 0 and i > 0:
            print(f"Iteration {i}: xk = {xk}, grad norm = {np.linalg.norm(gk)}")
        pk = -np.linalg.inv(hess_func(xk))@gk
        alpha = alpha_backtracking(func,grad_func,pk,xk)
        xk = xk + alpha*pk
        path_history.append(xk.copy())
        gk = grad_func(xk)
        grad_history.append(np.linalg.norm(grad_func(xk)))
        if np.linalg.norm(grad_func(xk)) < tol:
            print(f"Newtons method converged in {i} iterations at {xk}")
            return xk,i,grad_history,path_history
    print(f"Newtons method did not converge in {max_iter} iterations at {xk}")
    return xk,i,grad_history,path_history

def plot_contour_with_path(func, path_history, title, x_range=(-2, 4), y_range=(-1, 5)):
    """Plot contour of function with optimization path overlaid"""
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Evaluate function on grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contours (log scale for Rosenbrock function)
    levels = np.logspace(-1, 3.5, 35)
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot optimization path
    path_array = np.array(path_history)
    ax.plot(path_array[:, 0], path_array[:, 1], 'ro-', linewidth=2, 
            markersize=4, label='Optimization Path', alpha=0.8)
    ax.plot(path_array[0, 0], path_array[0, 1], 'go', markersize=10, 
            label=f'Start: ({path_array[0, 0]:.2f}, {path_array[0, 1]:.2f})')
    ax.plot(path_array[-1, 0], path_array[-1, 1], 'r*', markersize=15, 
            label=f'End: ({path_array[-1, 0]:.3f}, {path_array[-1, 1]:.3f})')
    
    # Mark optimum at (1,1)
    ax.plot(1, 1, 'b*', markersize=15, label='True Optimum (1, 1)')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'{title}\nIterations: {len(path_history)-1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#%%
#steepest descent
x0 = np.array([2.0, 2.0])
xk,i,grad_history,path_history = steepest_descent(x0,rosenbrock,grad_rosenbrock)

#plot log(grad_history) vs iteration
plt.plot(np.log(grad_history))
plt.xlabel("Iteration")
plt.ylabel("Log(Gradient Norm)")
plt.title("Steepest Descent")
plt.show()

#plot contour with path
plot_contour_with_path(rosenbrock, path_history, "Steepest Descent")

#%%
#nonlinear conjugate gradient
x0 = np.array([2.0, 2.0])
xk,i,grad_history,path_history = nonlin_conj_grad(x0,rosenbrock,grad_rosenbrock)

#plot log(grad_history) vs iteration
plt.plot(np.log(grad_history))
plt.xlabel("Iteration")
plt.ylabel("Log(Gradient Norm)")
plt.title("Nonlinear Conjugate Gradient")
plt.show()

#plot contour with path
plot_contour_with_path(rosenbrock, path_history, "Nonlinear Conjugate Gradient")

#%%
#quasi newton
x0 = np.array([2.0, 2.0])
xk,i,grad_history,path_history = quasi_newton_bfgs(x0,rosenbrock,grad_rosenbrock)

#plot log(grad_history) vs iteration
plt.plot(np.log(grad_history))
plt.xlabel("Iteration")
plt.ylabel("Log(Gradient Norm)")
plt.title("Quasi-Newton BFGS")
plt.show()

#plot contour with path
plot_contour_with_path(rosenbrock, path_history, "Quasi-Newton BFGS")

#%%
#newtons method
x0 = np.array([2.0, 2.0])
xk,i,grad_history,path_history = newtons_method(x0,rosenbrock,grad_rosenbrock,hess_rosenbrock)

#plot log(grad_history) vs iteration
plt.plot(np.log(grad_history))
plt.xlabel("Iteration")
plt.ylabel("Log(Gradient Norm)")
plt.title("Newtons Method")
plt.show()

#plot contour with path
plot_contour_with_path(rosenbrock, path_history, "Newton's Method")
