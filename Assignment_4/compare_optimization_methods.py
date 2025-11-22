import numpy as np
import matplotlib.pyplot as plt
import os

# Load unconstrained SQP timing data for comparison
if os.path.exists('plots/sqp_unconstrained_timing_data.npz'):
    sqp_data = np.load('plots/sqp_unconstrained_timing_data.npz')
    sqp_iterations = sqp_data['iterations']
    sqp_objective = sqp_data['objective']
    sqp_time_per_iter = sqp_data['time_per_iter']
    sqp_total_time = sqp_data['total_time']
elif os.path.exists('plots/sqp_timing_data.npz'):
    # Fall back to constrained version if unconstrained not available
    sqp_data = np.load('plots/sqp_timing_data.npz')
    sqp_iterations = sqp_data['iterations']
    sqp_objective = sqp_data['objective']
    sqp_time_per_iter = sqp_data['time_per_iter']
    sqp_total_time = sqp_data['total_time']
else:
    print("SQP timing data not found. Please run scipy_range_opt_unconstrained.py first.")
    sqp_data = None

# Load Neural Network timing data (check both locations)
if os.path.exists('plots/nn_timing_data.npz'):
    nn_data = np.load('plots/nn_timing_data.npz')
elif os.path.exists('neural_network_optimization/plots/nn_timing_data.npz'):
    nn_data = np.load('neural_network_optimization/plots/nn_timing_data.npz')
else:
    print("Neural network timing data not found. Please run optimization_neural_network.py first.")
    nn_data = None

if nn_data is not None:
    nn_epochs = nn_data['epochs']
    nn_objective = nn_data['objective']
    nn_time_per_iter = nn_data['time_per_iter']

# Create comparison plots
if sqp_data is not None and nn_data is not None:
    # Plot 1: Objective function convergence comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot full SQP convergence
    ax.plot(sqp_iterations, sqp_objective, 'b-o', 
            linewidth=2, markersize=8, label='SQP Method', alpha=0.8, markevery=1)
    
    # For neural network, sample to show progression without overwhelming the plot
    # Show first 50, then every 100th, then every 1000th after 1000
    nn_indices = list(range(min(50, len(nn_epochs))))
    if len(nn_epochs) > 50:
        nn_indices.extend(list(range(50, min(1000, len(nn_epochs)), 100)))
    if len(nn_epochs) > 1000:
        nn_indices.extend(list(range(1000, len(nn_epochs), 1000)))
    nn_indices = sorted(list(set(nn_indices)))  # Remove duplicates and sort
    
    ax.plot(np.array(nn_epochs)[nn_indices], np.array(nn_objective)[nn_indices], 'r-s', 
            linewidth=2, markersize=6, label='Neural Network Method', alpha=0.8, markevery=1)
    
    ax.set_xlabel('Iteration/Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objective Function (Range in km)', fontsize=12, fontweight='bold')
    ax.set_title('Objective Function Convergence Comparison (Unconstrained Optimization)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/optimization_convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved convergence comparison plot to plots/optimization_convergence_comparison.png")
    plt.close()
    
    # Plot 2: Time per iteration comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot SQP time per iteration (align with iterations, skip initial point)
    sqp_iter_for_time = sqp_iterations[1:] if len(sqp_iterations) == len(sqp_time_per_iter) + 1 else sqp_iterations[:len(sqp_time_per_iter)]
    sqp_time_plot = sqp_time_per_iter[:len(sqp_iter_for_time)]
    ax.plot(sqp_iter_for_time, sqp_time_plot * 1000, 
            'b-o', linewidth=2, markersize=8, label='SQP Method', alpha=0.8, markevery=1)
    
    # For neural network, sample to show progression (use same indices as convergence plot)
    ax.plot(np.array(nn_epochs)[nn_indices], np.array(nn_time_per_iter)[nn_indices] * 1000, 
            'r-s', linewidth=2, markersize=6, label='Neural Network Method', alpha=0.8, markevery=1)
    
    ax.set_xlabel('Iteration/Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time per Iteration (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Time per Iteration Comparison (Unconstrained Optimization)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/time_per_iteration_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved time per iteration comparison plot to plots/time_per_iteration_comparison.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("OPTIMIZATION METHOD COMPARISON SUMMARY (UNCONSTRAINED)")
    print("="*60)
    print(f"\nUnconstrained SQP Method:")
    print(f"  Total iterations: {len(sqp_iterations)}")
    print(f"  Total time: {sqp_total_time:.4f} s")
    print(f"  Average time per iteration: {np.mean(sqp_time_per_iter):.6f} s ({np.mean(sqp_time_per_iter)*1000:.3f} ms)")
    print(f"  Final objective value: {sqp_objective[-1]:.2f} km")
    
    print(f"\nNeural Network Method:")
    print(f"  Total epochs: {len(nn_epochs)}")
    print(f"  Total time: {np.sum(nn_time_per_iter):.4f} s")
    print(f"  Average time per iteration: {np.mean(nn_time_per_iter):.6f} s ({np.mean(nn_time_per_iter)*1000:.3f} ms)")
    print(f"  Final objective value: {nn_objective[-1]:.2f} km")
    
    print(f"\nComparison:")
    print(f"  Speedup factor (time per iter): {np.mean(sqp_time_per_iter) / np.mean(nn_time_per_iter):.2f}x")
    print(f"  Objective difference: {abs(sqp_objective[-1] - nn_objective[-1]):.2f} km")
    print("="*60 + "\n")
    
elif sqp_data is None:
    print("Please run scipy_range_opt_unconstrained.py first to generate unconstrained SQP timing data.")
elif nn_data is None:
    print("Please run optimization_neural_network.py first to generate neural network timing data.")

