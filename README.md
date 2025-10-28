# Assignment 3: Constrained Optimization Methods

This repository contains the implementation and analysis of two constrained optimization methods: Sequential Quadratic Programming (SQP) and the Quadratic Penalty Method.

## Files

### Core Implementation
- `SQP.py` - Main implementation containing both SQP and Quadratic Penalty algorithms
- `line_search_optimizers.py` - Quasi-Newton BFGS optimizer used by the penalty method

### Report Generation
- `Assignment_3_report.tex` - LaTeX source for the report (requires LaTeX installation)
- `generate_report.py` - Python script to generate PDF report using matplotlib
- `compile_report.sh` - Shell script to compile LaTeX to PDF (requires LaTeX)

### Generated Outputs
- `Assignment_3_report.pdf` - Complete PDF report with results and analysis
- Various `.png` files - Optimization plots and convergence comparisons

## Usage

### Running the Optimization Algorithms
```bash
cd /Users/aidan1/Documents/McGill/MECH597
source venv/bin/activate
python Assignment_3/SQP.py
```

This will:
- Run SQP and Quadratic Penalty methods on both constraint configurations
- Generate contour plots showing optimization paths
- Generate convergence comparison plots
- Display results summary

### Generating the PDF Report

**Option 1: Using Python (Recommended)**
```bash
python Assignment_3/generate_report.py
```

**Option 2: Using LaTeX (if installed)**
```bash
./compile_report.sh
```

## Results Summary

### Constraint 1: Linear Constraint (x₁ + x₂ = 1)
- **SQP**: Solution [0.6188, 0.3812]ᵀ, 12 iterations
- **Quadratic Penalty**: Solution [0.6188, 0.3812]ᵀ, 79 iterations

### Constraint 2: Nonlinear Constraint (x₁² + x₂² = 1)
- **SQP**: Solution [0.7864, 0.6177]ᵀ, 10 iterations
- **Quadratic Penalty**: Solution [0.7864, 0.6177]ᵀ, 55 iterations

## Key Findings

1. **Convergence Speed**: SQP demonstrates faster convergence with fewer iterations
2. **Solution Quality**: Both methods achieve nearly identical optimal solutions
3. **Algorithm Characteristics**: 
   - SQP: Faster but requires second-order derivatives
   - Quadratic Penalty: Slower but simpler implementation

## Dependencies

- Python 3.12+
- JAX (for automatic differentiation)
- NumPy
- Matplotlib
- SciPy (for optimization utilities)

## Report Contents

The generated PDF report includes:
- Abstract and problem formulation
- Methodology description
- Detailed results with tables and plots
- Analysis and discussion
- Conclusions and recommendations
- Code implementation snippets

All plots are automatically included showing:
- Optimization paths on contour plots
- Convergence behavior comparisons
- Algorithm performance metrics
