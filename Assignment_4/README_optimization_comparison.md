# Optimization Comparison Setup

To generate the comparison plots between SQP and Neural Network optimization methods:

1. **Run SQP optimization** (generates timing data):
   ```bash
   python scipy_range_opt.py
   ```
   This will create `plots/sqp_timing_data.npz`

2. **Run Neural Network optimization** (generates timing data):
   ```bash
   cd neural_network_optimization
   python optimization_neural_network.py
   ```
   This will create `plots/nn_timing_data.npz` and `plots/nn_optimization_convergence.png`

3. **Generate comparison plots**:
   ```bash
   python compare_optimization_methods.py
   ```
   This will create:
   - `plots/optimization_convergence_comparison.png` - Objective function convergence comparison
   - `plots/time_per_iteration_comparison.png` - Time per iteration comparison

Note: The neural network optimization may take several minutes to complete (currently set to 10,000 epochs for faster execution).

