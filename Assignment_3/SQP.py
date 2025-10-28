#minimize f wrt x subject to c_i(x) = 0 for all i
#%%
#Imports
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from line_search_optimizers import quasi_newton_bfgs

#%%
# Rosenbrock function
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# Constraints: c_i(x) = 0
c_hat1 = lambda x: 1 - x[0] - x[1]  # x[0] + x[1] = 1
c_hat2 = lambda x: x[0]**2 + x[1]**2 - 1  # x[0]^2 + x[1]^2 = 1

#%%
# SQP Algorithm with Multiple Constraints

def quadratic_penalty(x0, func, c_array, mu0=1.0, tau0=0.5, beta=10.0, eps=1e-4, max_iter=30, track_data=True):
    """
    Quadratic Penalty Method
    
    Parameters:
    -----------
    x0 : array-like
        Initial guess
    func : callable
        Objective function
    c_array : list of callables
        List of constraint functions c_i(x) = 0
    mu0 : float
        Initial penalty parameter
    beta : float
        Penalty parameter increase factor
    eps : float
        Convergence tolerance
    max_iter : int
        Maximum iterations per inner optimization
    track_data : bool
        If True, returns additional data for plotting
    
    Returns:
    --------
    xk : final solution
    x_path : list of optimization path points (if track_data=True)
    gradient_norms : list of gradient norms (if track_data=True)
    """
    # Tracking lists
    if track_data:
        x_path = [x0.copy()]
        gradient_norms = []
    
    
    xk = x0.copy()
    mu = mu0
    tau = tau0
    for outer_iter in range(max_iter):  # Maximum outer iterations
        # Define penalized objective function
        def penalized_objective(x):
            penalty = 0.0
            for c in c_array:
                penalty += c(x)**2
            return func(x) + mu/2 * penalty
        
        # Gradient of penalized function
        Q_grad = jax.grad(penalized_objective)
        if tau < eps:
            tau = eps
        #solve with quasi-newton bfgs
        try:
            xk,_,grad_history,path_history = quasi_newton_bfgs(xk, penalized_objective, Q_grad, tol=tau, max_iter=100)
            if track_data:
                # Flatten the path history and add to x_path
                for point in path_history:
                    x_path.append(point)
                # Flatten the gradient history and add to gradient_norms
                for grad_norm in grad_history:
                    gradient_norms.append(grad_norm)
        except Exception as e:
            print(f"Warning: Quasi-Newton failed: {e}")
            if track_data:
                # Add current point if optimization failed
                x_path.append(xk.copy())
                gradient_norms.append(np.linalg.norm(Q_grad(xk)))
        # Check outer convergence
        c_vals = jnp.array([c(xk) for c in c_array])
        constraint_violation = jnp.linalg.norm(c_vals)
        
        if constraint_violation < eps:
            break
        
        # Update penalty parameter
        mu *= beta
        tau /= (beta/2)
    if track_data:
        return xk, x_path, gradient_norms
    else:
        return xk


def quadratic_penalty_inequality(x0, func, c_ineq_array, mu0=1.0, tau0=0.5, beta=10.0, eps=1e-4, max_iter=30, track_data=True, verbose=False):
    """
    Quadratic Penalty Method
    
    Parameters:
    -----------
    x0 : array-like
        Initial guess
    func : callable
        Objective function
    c_array : list of callables
        List of constraint functions c_i(x) = 0
    mu0 : float
        Initial penalty parameter
    beta : float
        Penalty parameter increase factor
    eps : float
        Convergence tolerance
    max_iter : int
        Maximum iterations per inner optimization
    track_data : bool
        If True, returns additional data for plotting
    
    Returns:
    --------
    xk : final solution
    x_path : list of optimization path points (if track_data=True)
    gradient_norms : list of gradient norms (if track_data=True)
    """
    # Tracking lists
    if track_data:
        x_path = [x0.copy()]
        gradient_norms = []
    
    
    xk = x0.copy()
    mu = mu0
    tau = tau0
    for outer_iter in range(max_iter):  # Maximum outer iterations
        # Define penalized objective function
        def penalized_objective(x):
            penalty = 0.0
            for c_ineq in c_ineq_array:
                penalty += max(0, -c_ineq(x))**2
            return func(x) + mu/2 * penalty
        
        # Gradient of penalized function
        Q_grad = jax.grad(penalized_objective)
        if tau < eps:
            tau = eps
        #solve with quasi-newton bfgs
        try:
            xk,_,grad_history,path_history = quasi_newton_bfgs(xk, penalized_objective, Q_grad, tol=tau, max_iter=100)
            if track_data:
                # Flatten the path history and add to x_path
                for point in path_history:
                    x_path.append(point)
                # Flatten the gradient history and add to gradient_norms
                for grad_norm in grad_history:
                    gradient_norms.append(grad_norm)
        except Exception as e:
            print(f"Warning: Quasi-Newton failed: {e}")
            if track_data:
                # Add current point if optimization failed
                x_path.append(xk.copy())
                gradient_norms.append(np.linalg.norm(Q_grad(xk)))
        # Check outer convergence
        # --- Penalty method final convergence check: ---
        # Check *both* equality/inequality feasibility, stationarity, & penalty tightness
        c_ineq_vals = jnp.array([c(xk) for c in c_ineq_array]) if c_ineq_array else jnp.array([])

        ineq_feas = jnp.all(c_ineq_vals >= -eps) if c_ineq_vals.size > 0 else True
        
        # Compute (approximate) stationarity wrt penalized function
        gradQ = Q_grad(xk)
        stationarity = jnp.linalg.norm(gradQ)
        penalty_loose = (mu > 1e4)  # Arbitrary threshold for "very large penalty" (tune as appropriate)
        tight_enough = (tau < eps*2)
        if verbose:
            print(f"Inequality Feasibility: {ineq_feas}")
            print(f"Stationarity: {stationarity}")
            print(f"Penalty Loose: {penalty_loose}")
            print(f"Tight Enough: {tight_enough}")
        if ineq_feas and (tight_enough or penalty_loose):
            print("Final Convergence Check Passed")
            break
            
        
        # Update penalty parameter
        mu *= beta
        tau /= (beta/5)
    if track_data:
        return xk, x_path, gradient_norms
    else:
        return xk

# Test quadratic penalty method (removed for cleaner output)
def SQP(x0, lamb0, func, c_eq_array=None, c_ineq_array=None, eps=1e-3, max_iter=200, track_data=False, BFGS=True):
    """
    Sequential Quadratic Programming for:
    min f(x) subject to c_eq_i(x) = 0 for i = 1, ..., m
                    and c_ineq_j(x) <= 0 for j = 1, ..., p
    
    Parameters:
    -----------
    x0 : array-like, shape (n,)
        Initial guess for variables
    lamb0 : array-like, shape (m+p,) or scalar
        Initial guess for Lagrange multipliers
    func : callable
        Objective function f(x)
    c_eq_array : list of callables or None
        List of equality constraint functions [c_eq_1, c_eq_2, ..., c_eq_m]
    c_ineq_array : list of callables or None
        List of inequality constraint functions [c_ineq_1, c_ineq_2, ..., c_ineq_p]
    eps : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
    track_data : bool
        If True, returns additional data for plotting (x_path, gradient_norms)
    
    Returns:
    --------
    xk : final solution
    lamb : final Lagrange multipliers
    x_path : list of optimization path points (if track_data=True)
    gradient_norms : list of gradient norms (if track_data=True)
    """
    # Handle None inputs
    if c_eq_array is None:
        c_eq_array = []
    if c_ineq_array is None:
        c_ineq_array = []
    
    n = len(x0)  # Number of variables
    m = len(c_eq_array)  # Number of equality constraints
    p = len(c_ineq_array)  # Number of inequality constraints

    # Initialize
    xk = x0
    if jnp.isscalar(lamb0):
        lamb_eq = jnp.ones(m) * lamb0 if m > 0 else jnp.array([])
        lamb_ineq = jnp.ones(p) * lamb0 if p > 0 else jnp.array([])
    else:
        lamb_eq = jnp.array(lamb0[:m]) if m > 0 else jnp.array([])
        lamb_ineq = jnp.array(lamb0[m:m+p]) if p > 0 else jnp.array([])
    
    # Initialize tracking lists if needed
    if track_data:
        x_path = [xk.copy()]
        gradient_norms = []
    
    # Gradient and Hessian functions
    def lagrangian(x, lamb_eq, lamb_ineq):
        result = func(x)
        if m > 0:
            result += lamb_eq.T @ jnp.array([c(x) for c in c_eq_array])
        if p > 0:
            result += lamb_ineq.T @ jnp.array([c(x) for c in c_ineq_array])
        return result
    
    grad_func = jax.grad(func)
    if not BFGS:
        hess_lagrangian_func = jax.hessian(lagrangian)
    else:
        H_Lagrangian = jnp.eye(n)
    
    # Create gradient functions for each constraint
    grad_c_eq_funcs = [jax.grad(c) for c in c_eq_array]
    grad_c_ineq_funcs = [jax.grad(c) for c in c_ineq_array]
    
    for i in range(max_iter):
        # Evaluate objective and gradients
        grad_f = grad_func(xk)
        if not BFGS:
            H_Lagrangian = hess_lagrangian_func(xk, lamb_eq, lamb_ineq)
        
        # Calculate the value of each constraint at the current guess xk
        c_eq_vals = jnp.array([c(xk) for c in c_eq_array]) if m > 0 else jnp.array([])
        c_ineq_vals = jnp.array([c(xk) for c in c_ineq_array]) if p > 0 else jnp.array([])

        # For each constraint, get its gradient (derivative) at xk
        A_eq = jnp.array([grad_c(xk) for grad_c in grad_c_eq_funcs]) if m > 0 else jnp.zeros((0, n))
        A_ineq = jnp.array([grad_c(xk) for grad_c in grad_c_ineq_funcs]) if p > 0 else jnp.zeros((0, n))
        
        # KKT residual
        grad_L = grad_f.copy()
        if m > 0:
            grad_L = grad_L - A_eq.T @ lamb_eq
        if p > 0:
            grad_L = grad_L - A_ineq.T @ lamb_ineq
        
        constraint_violation = jnp.linalg.norm(c_eq_vals) + jnp.linalg.norm(jnp.maximum(0, c_ineq_vals), ord=1)
        kkt_norm = jnp.linalg.norm(grad_L) + constraint_violation
        
        # Track gradient norm if needed
        if track_data:
            gradient_norms.append(float(kkt_norm))
        
        # Progress output removed for cleaner output
        
        # Check convergence
        if kkt_norm < eps:
            print(f"\nConverged in {i} iterations!")
            lamb = jnp.concatenate([lamb_eq, lamb_ineq]) if (m > 0 or p > 0) else jnp.array([])
            if track_data:
                return xk, lamb, x_path, gradient_norms
            else:
                return xk, lamb
        
        # Identify active inequality constraints (violated or nearly active)
        # An inequality constraint c_i(x) <= 0 is active if c_i(x) >= -eps or lambda_i > 0
        if p > 0:
            # Active if: constraint violated OR multiplier is positive (constraint is binding)
            active_ineq_mask = (c_ineq_vals >= -eps * 10) | (lamb_ineq > eps * 0.1)
        else:
            active_ineq_mask = jnp.array([], dtype=bool)
        
        # Count active inequality constraints
        n_active = int(jnp.sum(active_ineq_mask)) if p > 0 else 0
        
        # Build KKT matrix including equality constraints and active inequality constraints
        total_constraints = m + n_active
        
        if total_constraints > 0:
            # Combine equality and active inequality constraints
            if m > 0 and n_active > 0:
                A_active_ineq = A_ineq[active_ineq_mask]
                A_combined = jnp.vstack([A_eq, A_active_ineq])
                c_combined = jnp.concatenate([c_eq_vals, c_ineq_vals[active_ineq_mask]])
            elif m > 0:
                A_combined = A_eq
                c_combined = c_eq_vals
            else:  # n_active > 0
                A_combined = A_ineq[active_ineq_mask]
                c_combined = c_ineq_vals[active_ineq_mask]
            
            zero_block = jnp.zeros((total_constraints, total_constraints))
            KKT_matrix = jnp.block([
                [H_Lagrangian, -A_combined.T],
                [-A_combined,  zero_block]
            ])
            
            rhs = jnp.concatenate([-grad_L, c_combined])
        else:
            # No active constraints, just use Newton step
            KKT_matrix = H_Lagrangian
            rhs = -grad_L
        
        # Solve KKT system for search direction
        try:
            # Check for NaN/Inf in KKT matrix and RHS
            if jnp.isnan(KKT_matrix).any() or jnp.isinf(KKT_matrix).any() or jnp.isnan(rhs).any() or jnp.isinf(rhs).any():
                print(f"Warning: NaN/Inf in KKT system at iteration {i}")
                break
            
            search_dir = jnp.linalg.solve(KKT_matrix, rhs)
            
            # Check for NaN/Inf in solution
            if jnp.isnan(search_dir).any() or jnp.isinf(search_dir).any():
                print(f"Warning: NaN/Inf in KKT solution at iteration {i}")
                break
                
        except jnp.linalg.LinAlgError:
            print(f"KKT matrix is singular at iteration {i}!")
            break
        except Exception as e:
            print(f"Error solving KKT system at iteration {i}: {e}")
            break
            
        pk = search_dir[:n]  # Step in x
        
        # Extract multiplier updates for equality and active inequality constraints
        if total_constraints > 0:
            lamb_updates = search_dir[n:n+total_constraints]
            
            if m > 0 and n_active > 0:
                plamb_eq = lamb_updates[:m]
                plamb_active_ineq = lamb_updates[m:]
                # Update only active inequality multipliers using numpy operations
                plamb_ineq = jnp.zeros(p)
                # Use boolean mask to assign values
                active_indices = jnp.where(active_ineq_mask)[0]
                plamb_ineq = plamb_ineq.at[active_indices].set(plamb_active_ineq)
            elif m > 0:
                plamb_eq = lamb_updates
                plamb_ineq = jnp.zeros(p) if p > 0 else jnp.array([])
            else:  # n_active > 0
                plamb_eq = jnp.array([]) if m == 0 else jnp.zeros(m)
                plamb_ineq = jnp.zeros(p)
                active_indices = jnp.where(active_ineq_mask)[0]
                plamb_ineq = plamb_ineq.at[active_indices].set(lamb_updates)
        else:
            plamb_eq = jnp.array([]) if m == 0 else jnp.zeros(m)
            plamb_ineq = jnp.zeros(p) if p > 0 else jnp.array([])
        
        # Check for NaN/Inf in search directions
        if jnp.isnan(pk).any() or jnp.isinf(pk).any() or jnp.isnan(plamb_eq).any() or jnp.isinf(plamb_eq).any():
            print(f"Warning: NaN/Inf in search directions at iteration {i}")
            break
        
        # Line search with L1 penalty merit function
        alpha = 1.0
        sigma = 1.0
        rho = 0.5
        tau = 0.5
        
        # Adaptive penalty parameter
        c_norm = jnp.linalg.norm(c_eq_vals, ord=1) + jnp.linalg.norm(jnp.maximum(0, c_ineq_vals), ord=1)
        if c_norm > 1e-10:
            numerator = grad_f @ pk + sigma/2 * pk @ H_Lagrangian @ pk
            denominator = (1-rho) * c_norm
            if abs(denominator) > 1e-12:  # Avoid division by very small numbers
                mu = numerator / denominator
                # Calculate maximum lambda value safely
                lamb_max = 0
                if m > 0:
                    lamb_max = max(lamb_max, jnp.abs(lamb_eq).max())
                if p > 0:
                    lamb_max = max(lamb_max, jnp.abs(lamb_ineq).max())
                mu = max(abs(mu), lamb_max + 1)
            else:
                # Calculate maximum lambda value safely
                lamb_max = 0
                if m > 0:
                    lamb_max = max(lamb_max, jnp.abs(lamb_eq).max())
                if p > 0:
                    lamb_max = max(lamb_max, jnp.abs(lamb_ineq).max())
                mu = lamb_max + 1
        else:
            # Calculate maximum lambda value safely
            lamb_max = 0
            if m > 0:
                lamb_max = max(lamb_max, jnp.abs(lamb_eq).max())
            if p > 0:
                lamb_max = max(lamb_max, jnp.abs(lamb_ineq).max())
            mu = lamb_max + 1
        
        # Merit function: phi(x) = f(x) + mu * ||c(x)||_1
        def phi(x):
            c_eq_x = jnp.array([c(x) for c in c_eq_array])
            c_ineq_x = jnp.array([c(x) for c in c_ineq_array])
            return func(x) + mu * jnp.linalg.norm(c_eq_x, ord=1) + jnp.linalg.norm(jnp.maximum(0, c_ineq_x), ord=1)
        
        # Directional derivative of merit function
        nu = 1e-4
        D = grad_f @ pk - mu * c_norm
        
        # Backtracking line search
        phi_k = phi(xk)

        for line_search_iter in range(20):
            xk_new = xk + alpha * pk
            lamb_eq_new = lamb_eq + alpha * plamb_eq
            lamb_ineq_new = jnp.maximum(0, lamb_ineq + alpha * plamb_ineq)  # Enforce non-negativity
            
            # Check for NaN in new values
            if jnp.isnan(xk_new).any() or jnp.isinf(xk_new).any() or jnp.isnan(lamb_eq_new).any() or jnp.isinf(lamb_eq_new).any():
                print(f"Warning: NaN/Inf in line search at iteration {i}, alpha={alpha}")
                alpha *= tau
                continue
            
            phi_new = phi(xk_new)
            
            # Check for NaN in phi_new
            if jnp.isnan(phi_new) or jnp.isinf(phi_new):
                print(f"Warning: NaN/Inf in phi_new at iteration {i}, alpha={alpha}")
                alpha *= tau
                continue
            
            if phi_new <= phi_k + nu * alpha * D:
                break
            alpha *= tau
        
        # Final check for valid alpha
        if alpha < 1e-10:
            print(f"Warning: Line search failed, alpha too small at iteration {i}")
            break
        # Update
        if BFGS: #Damped BFGS update
            sk = xk_new - xk
            # Calculate gradient difference for BFGS
            grad_L_new = grad_func(xk_new).copy()
            if m > 0:
                grad_L_new = grad_L_new - A_eq.T @ lamb_eq_new
            if p > 0:
                grad_L_new = grad_L_new - A_ineq.T @ lamb_ineq_new
            
            grad_L_old = grad_func(xk).copy()
            if m > 0:
                grad_L_old = grad_L_old - A_eq.T @ lamb_eq_new
            if p > 0:
                grad_L_old = grad_L_old - A_ineq.T @ lamb_ineq_new
            
            yk = grad_L_new - grad_L_old
            
            # Check for NaN or infinite values
            if jnp.isnan(sk).any() or jnp.isinf(sk).any() or jnp.isnan(yk).any() or jnp.isinf(yk).any():
                print(f"Warning: NaN/Inf detected in sk or yk at iteration {i}")
                break
            
            # Ensure sk and yk are column vectors
            sk = sk.reshape(-1, 1) if sk.ndim == 1 else sk
            yk = yk.reshape(-1, 1) if yk.ndim == 1 else yk
            
            skT_H_sk = sk.T @ H_Lagrangian @ sk
            skT_yk = sk.T @ yk
            
            # Check if step size is too small (convergence)
            sk_norm = jnp.linalg.norm(sk)
            if sk_norm < 1e-12:
                # Step size very small, check for convergence
                xk = xk_new
                lamb_eq = lamb_eq_new
                lamb_ineq = lamb_ineq_new
                if track_data:
                    x_path.append(xk.copy())
                
                # Check convergence with updated values
                grad_f_new = grad_func(xk)
                c_eq_vals_new = jnp.array([c(xk) for c in c_eq_array]) if m > 0 else jnp.array([])
                c_ineq_vals_new = jnp.array([c(xk) for c in c_ineq_array]) if p > 0 else jnp.array([])
                A_eq_new = jnp.array([grad_c(xk) for grad_c in grad_c_eq_funcs]) if m > 0 else jnp.zeros((0, n))
                A_ineq_new = jnp.array([grad_c(xk) for grad_c in grad_c_ineq_funcs]) if p > 0 else jnp.zeros((0, n))
                
                grad_L_new = grad_f_new.copy()
                if m > 0:
                    grad_L_new = grad_L_new - A_eq_new.T @ lamb_eq
                if p > 0:
                    grad_L_new = grad_L_new - A_ineq_new.T @ lamb_ineq
                
                constraint_violation_new = jnp.linalg.norm(c_eq_vals_new) + jnp.linalg.norm(jnp.maximum(0, c_ineq_vals_new), ord=1)
                kkt_norm_new = jnp.linalg.norm(grad_L_new) + constraint_violation_new
                
                if track_data:
                    gradient_norms.append(float(kkt_norm_new))
                
                if kkt_norm_new < eps:
                    print(f"\nConverged in {i+1} iterations!")
                    lamb = jnp.concatenate([lamb_eq, lamb_ineq]) if (m > 0 or p > 0) else jnp.array([])
                    if track_data:
                        return xk, lamb, x_path, gradient_norms
                    else:
                        return xk, lamb
                else:
                    # If KKT norm is not decreasing significantly, break to avoid infinite loop
                    if len(gradient_norms) > 5 and abs(gradient_norms[-1] - gradient_norms[-5]) < eps * 0.01:
                        print(f"KKT norm not decreasing significantly, breaking at iteration {i+1}")
                        break
                    continue
            
            # Check for numerical issues (only skip if truly problematic)
            if jnp.isnan(skT_H_sk) or jnp.isinf(skT_H_sk) or skT_H_sk <= 1e-20:
                print(f"Warning: Numerical issue with skT_H_sk at iteration {i}")
                print(f"skT_H_sk = {skT_H_sk}, sk norm = {sk_norm}")
                # Skip BFGS update for this iteration but continue
                print("Skipping BFGS update for this iteration")
                xk = xk_new
                lamb_eq = lamb_eq_new
                lamb_ineq = lamb_ineq_new
                if track_data:
                    x_path.append(xk.copy())
                continue
            
            if skT_yk > 0.2 * skT_H_sk:
                theta = 1.0
            else:
                denominator = skT_H_sk - skT_yk
                if abs(denominator) > 1e-12:  # Avoid division by very small numbers
                    theta = (0.8 * skT_H_sk) / denominator
                else:
                    theta = 1.0
            
            rk = theta * yk + (1-theta) * H_Lagrangian @ sk
            
            # Ensure rk is a column vector
            rk = rk.reshape(-1, 1) if rk.ndim == 1 else rk
            
            # Check for NaN in rk
            if jnp.isnan(rk).any() or jnp.isinf(rk).any():
                print(f"Warning: NaN/Inf detected in rk at iteration {i}")
                break
            
            # BFGS update formula
            skT_rk = sk.T @ rk
            
            # Check for numerical issues in denominators
            if abs(skT_rk) > 1e-16 and abs(skT_H_sk) > 1e-16:
                H_Lagrangian = H_Lagrangian - (H_Lagrangian @ sk @ sk.T @ H_Lagrangian) / skT_H_sk + (rk @ rk.T) / skT_rk
                
                # Check for NaN in updated Hessian
                if jnp.isnan(H_Lagrangian).any() or jnp.isinf(H_Lagrangian).any():
                    print(f"Warning: NaN/Inf detected in H_Lagrangian at iteration {i}")
                    H_Lagrangian = jnp.eye(n)  # Reset to identity
            else:
                print(f"Warning: Skipping BFGS update due to numerical issues at iteration {i}")
        xk = xk_new
        lamb_eq = lamb_eq_new
        lamb_ineq = lamb_ineq_new
        
        # Reset multipliers for inactive inequality constraints (complementarity)
        # If constraint is not active (c_i < -eps) and multiplier is small, set to zero
        if p > 0:
            # Get updated constraint values
            c_ineq_vals_new = jnp.array([c(xk) for c in c_ineq_array])
            inactive_mask = (c_ineq_vals_new < -eps * 10) & (lamb_ineq < eps * 0.1)
            lamb_ineq = jnp.where(inactive_mask, 0.0, lamb_ineq)
        
        # Track path if needed
        if track_data:
            x_path.append(xk.copy())
        
        # Safety check for NaN
        if jnp.isnan(xk).any() or jnp.isnan(lamb_eq).any() or jnp.isnan(lamb_ineq).any():
            print(f"NaN detected at iteration {i}!")
            break
    
    print(f"\nWarning: Loop Broke Before Convergence")
    lamb = jnp.concatenate([lamb_eq, lamb_ineq]) if (m > 0 or p > 0) else jnp.array([])
    if track_data:
        return xk, lamb, x_path, gradient_norms
    else:
        return xk, lamb

#%%
# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_convergence(gradient_norms, algorithm_name="SQP", save_path=None):
    """
    Plot convergence of gradient norms (log scale)
    
    Parameters:
    -----------
    gradient_norms : list or array
        List of gradient norms at each iteration
    algorithm_name : str
        Name of the algorithm for the plot title
    save_path : str, optional
        Path to save the plot (if None, just display)
    """
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy array and ensure positive values for log
    grad_norms = np.array(gradient_norms)
    grad_norms = np.maximum(grad_norms, 1e-16)  # Avoid log(0)
    
    iterations = np.arange(len(grad_norms))
    
    plt.semilogy(iterations, grad_norms, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Log of Gradient Norm', fontsize=12)
    plt.title(f'{algorithm_name} - Gradient Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    
    return plt.gcf()

def plot_contour_with_path(objective_func, constraint_funcs, x_path, 
                          x_bounds=None, y_bounds=None, 
                          algorithm_name="SQP", save_path=None):
    """
    Create contour plot of objective function with optimization path and constraints
    
    Parameters:
    -----------
    objective_func : callable
        Objective function f(x) where x is a 2D array
    constraint_funcs : list of callables
        List of constraint functions c_i(x) = 0
    x_path : list or array
        List of (x, y) points representing the optimization path
    x_bounds : tuple
        (x_min, x_max) for the plot bounds (if None, auto-calculate from path)
    y_bounds : tuple
        (y_min, y_max) for the plot bounds (if None, auto-calculate from path)
    algorithm_name : str
        Name of the algorithm for the plot title
    save_path : str, optional
        Path to save the plot (if None, just display)
    """
    # Auto-calculate bounds from path if not provided
    if x_bounds is None or y_bounds is None:
        if len(x_path) > 0:
            x_path_array = np.array(x_path)
            x_min, x_max = x_path_array[:, 0].min(), x_path_array[:, 0].max()
            y_min, y_max = x_path_array[:, 1].min(), x_path_array[:, 1].max()
            
            # Add some padding
            x_padding = (x_max - x_min) * 0.2
            y_padding = (y_max - y_min) * 0.2
            
            if x_bounds is None:
                x_bounds = (x_min - x_padding, x_max + x_padding)
            if y_bounds is None:
                y_bounds = (y_min - y_padding, y_max + y_padding)
            
        else:
            # Default bounds if no path
            x_bounds = (-10, 10)
            y_bounds = (-10, 10)
    
    # Create meshgrid for contour plot
    x = np.linspace(x_bounds[0], x_bounds[1], 100)
    y = np.linspace(y_bounds[0], y_bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate objective function on the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = jnp.array([X[i, j], Y[i, j]])
            Z[i, j] = objective_func(point)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot contour lines
    contour = plt.contour(X, Y, Z, levels=20, colors='gray', alpha=0.6, linewidths=0.5)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Fill contour plot
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Objective Function Value')
    
    # Plot constraints
    for i, constraint_func in enumerate(constraint_funcs):
        # For constraint c(x) = 0, plot the zero contour
        constraint_Z = np.zeros_like(X)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                point = jnp.array([X[row, col], Y[row, col]])
                constraint_Z[row, col] = constraint_func(point)
        
        constraint_contour = plt.contour(X, Y, constraint_Z, levels=[0], 
                                       colors=['red', 'blue', 'green', 'orange'][i % 4], 
                                       linewidths=3, linestyles=['-', '--', '-.', ':'][i % 4])
        plt.clabel(constraint_contour, inline=True, fontsize=10, 
                  fmt=f'c_{i+1}(x)=0', colors=['red', 'blue', 'green', 'orange'][i % 4])
    
    # Plot optimization path
    if len(x_path) > 0:
        x_path = np.array(x_path)
        plt.plot(x_path[:, 0], x_path[:, 1], 'ro-', linewidth=2, markersize=6, 
                label='Optimization Path', alpha=0.8)
        
        # Mark start and end points
        plt.plot(x_path[0, 0], x_path[0, 1], 'go', markersize=10, 
                label='Start', markeredgecolor='black', markeredgewidth=2)
        plt.plot(x_path[-1, 0], x_path[-1, 1], 'rs', markersize=10, 
                label='End', markeredgecolor='black', markeredgewidth=2)
    
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title(f'{algorithm_name} - Optimization Path', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Contour plot saved to: {save_path}")
    
    return plt.gcf()

def plot_comparison_convergence(gradient_norms_dict, save_path=None):
    """
    Plot convergence comparison for multiple algorithms
    
    Parameters:
    -----------
    gradient_norms_dict : dict
        Dictionary with algorithm names as keys and gradient norms as values
        e.g., {'SQP': [grad_norms], 'Quadratic Penalty': [grad_norms]}
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (algorithm, grad_norms) in enumerate(gradient_norms_dict.items()):
        grad_norms = np.array(grad_norms)
        grad_norms = np.maximum(grad_norms, 1e-16)  # Avoid log(0)
        iterations = np.arange(len(grad_norms))
        
        plt.semilogy(iterations, grad_norms, 
                    color=colors[i % len(colors)], 
                    linewidth=2, marker='o', markersize=4,
                    label=algorithm)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Log of Gradient Norm', fontsize=12)
    plt.title('Convergence Comparison - Gradient Norms', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return plt.gcf()

#%%
# =============================================================================
# Running on rosenbrock
# =============================================================================

if __name__ == "__main__":
    print("Optimization Algorithms Comparison")
    print("="*50)
    
    # Initial point
    x0 = jnp.array([10., 4.])
    
    # Constraint 1: x[0] + x[1] = 1
    print("\nConstraint 1: x[0] + x[1] = 1")
    print("-" * 30)
    
    c_array_1 = [c_hat1]
    
    # Run SQP on constraint 1
    print("Running SQP...")
    xk_sqp1, lamb_sqp1, x_path_sqp1, grad_norms_sqp1 = SQP(x0, 0.0, rosenbrock, c_eq_array=c_array_1, eps=1e-3, track_data=True)
    print(f"SQP: x = [{xk_sqp1[0]:.4f}, {xk_sqp1[1]:.4f}], iterations = {len(grad_norms_sqp1)}")
    
    # Run Quadratic Penalty on constraint 1
    print("Running Quadratic Penalty...")
    xk_penalty1, x_path_penalty1, grad_norms_penalty1 = quadratic_penalty(x0, rosenbrock, c_array=c_array_1, track_data=True)
    print(f"Penalty: x = [{xk_penalty1[0]:.4f}, {xk_penalty1[1]:.4f}], iterations = {len(grad_norms_penalty1)}")
    
    # Generate plots for constraint 1
    print("Generating plots...")
    #plot_convergence(grad_norms_sqp1, "SQP - Constraint 1", "sqp_constraint1_convergence.png")
    plot_contour_with_path(rosenbrock, c_array_1, x_path_sqp1, 
                          algorithm_name="SQP - Constraint 1", 
                          save_path="sqp_constraint1_contour.png")
    plot_contour_with_path(rosenbrock, c_array_1, x_path_penalty1, 
                          algorithm_name="Quadratic Penalty - Constraint 1", 
                          save_path="penalty_constraint1_contour.png",x_bounds=(-40,40), y_bounds=(-20,80))
    
    grad_norms_dict_1 = {
        'SQP': grad_norms_sqp1,
        'Quadratic Penalty': grad_norms_penalty1
    }
    plot_comparison_convergence(grad_norms_dict_1, "convergence_comparison_constraint1.png")
    
    # Constraint 2: x[0]^2 + x[1]^2 = 1
    print("\nConstraint 2: x[0]^2 + x[1]^2 = 1")
    print("-" * 30)
    
    c_array_2 = [c_hat2]
    
    # Run SQP on constraint 2
    print("Running SQP...")
    xk_sqp2, lamb_sqp2, x_path_sqp2, grad_norms_sqp2 = SQP(x0, 0.0, rosenbrock, c_eq_array=c_array_2, eps=1e-3, track_data=True)
    print(f"SQP: x = [{xk_sqp2[0]:.4f}, {xk_sqp2[1]:.4f}], iterations = {len(grad_norms_sqp2)}")
    
    # Run Quadratic Penalty on constraint 2
    print("Running Quadratic Penalty...")
    xk_penalty2, x_path_penalty2, grad_norms_penalty2 = quadratic_penalty(x0, rosenbrock, c_array_2, track_data=True)
    print(f"Penalty: x = [{xk_penalty2[0]:.4f}, {xk_penalty2[1]:.4f}], iterations = {len(grad_norms_penalty2)}")
    
    # Generate plots for constraint 2
    print("Generating plots...")
    plot_convergence(grad_norms_sqp2, "SQP - Constraint 2", "sqp_constraint2_convergence.png")
    plot_contour_with_path(rosenbrock, c_array_2, x_path_sqp2, 
                          algorithm_name="SQP - Constraint 2", 
                          save_path="sqp_constraint2_contour.png")
    plot_contour_with_path(rosenbrock, c_array_2, x_path_penalty2, 
                          algorithm_name="Quadratic Penalty - Constraint 2", 
                          save_path="penalty_constraint2_contour.png",x_bounds=(-20,20))
    grad_norms_dict_2 = {
        'SQP': grad_norms_sqp2,
        'Quadratic Penalty': grad_norms_penalty2
    }
    plot_comparison_convergence(grad_norms_dict_2, "convergence_comparison_constraint2.png")
    
    print("\n" + "="*50)
    print("Generated plots:")
    print("  • sqp_constraint1_convergence.png")
    print("  • sqp_constraint1_contour.png")
    print("  • sqp_constraint2_convergence.png")
    print("  • sqp_constraint2_contour.png")
    print("  • convergence_comparison_constraint1.png")
    print("  • convergence_comparison_constraint2.png")
    print("="*50)
    
 