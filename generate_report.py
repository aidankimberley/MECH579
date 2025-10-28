#!/usr/bin/env python3
"""
Generate PDF Report for Assignment 3
This script creates a comprehensive PDF report from the optimization results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
import os

def create_report_pdf():
    """Create a comprehensive PDF report for Assignment 3"""
    
    # Check if plot files exist
    plot_files = [
        'sqp_constraint1_contour.png',
        'penalty_constraint1_contour.png', 
        'convergence_comparison_constraint1.png',
        'sqp_constraint2_contour.png',
        'penalty_constraint2_contour.png',
        'convergence_comparison_constraint2.png'
    ]
    
    missing_files = [f for f in plot_files if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: Missing plot files: {missing_files}")
        print("Please run SQP.py first to generate the plots.")
        return
    
    # Create PDF
    with PdfPages('Assignment_3_report.pdf') as pdf:
        
        # Title Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, 'MECH 597 Assignment 3', 
                fontsize=24, fontweight='bold', ha='center')
        ax.text(0.5, 0.75, 'Constrained Optimization Methods', 
                fontsize=18, ha='center')
        ax.text(0.5, 0.7, 'Sequential Quadratic Programming vs Quadratic Penalty Method', 
                fontsize=14, ha='center', style='italic')
        
        # Author and date
        ax.text(0.5, 0.4, 'Aidan [Your Last Name]', 
                fontsize=16, ha='center')
        ax.text(0.5, 0.35, f'Date: {datetime.now().strftime("%B %d, %Y")}', 
                fontsize=12, ha='center')
        
        # Course info
        ax.text(0.5, 0.2, 'McGill University', 
                fontsize=14, ha='center')
        ax.text(0.5, 0.15, 'Department of Mechanical Engineering', 
                fontsize=12, ha='center')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Abstract Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        ax.text(0.1, 0.9, 'Abstract', fontsize=18, fontweight='bold')
        
        abstract_text = """
This report presents a comparative analysis of two constrained optimization methods: 
Sequential Quadratic Programming (SQP) and the Quadratic Penalty Method. Both 
algorithms are applied to minimize the Rosenbrock function subject to different 
constraint configurations.

The objective function considered is:
f(x) = 100(x₂ - x₁²)² + (1 - x₁)²

Two constraint configurations are analyzed:
• Constraint 1: Linear constraint x₁ + x₂ = 1
• Constraint 2: Nonlinear constraint x₁² + x₂² = 1

Both algorithms start from the initial point x₀ = [10, 4]ᵀ.

Results show that SQP demonstrates faster convergence with fewer iterations 
(12 vs 79 for Constraint 1, 10 vs 55 for Constraint 2), while the Quadratic 
Penalty method shows more gradual but reliable convergence. Both methods 
achieve similar solution quality, converging to nearly identical optimal points.

The choice between methods depends on specific application requirements, 
including computational resources, derivative availability, and convergence 
speed requirements.
        """
        
        ax.text(0.1, 0.8, abstract_text, fontsize=11, va='top', wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Problem Formulation Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        ax.text(0.1, 0.9, 'Problem Formulation', fontsize=18, fontweight='bold')
        
        problem_text = """
Objective Function:
f(x) = 100(x₂ - x₁²)² + (1 - x₁)²

Constraint Configurations:

Constraint 1: Linear Constraint
x₁ + x₂ = 1

Constraint 2: Nonlinear Constraint  
x₁² + x₂² = 1

Initial Conditions:
x₀ = [10, 4]ᵀ

Algorithm Parameters:
• Convergence tolerance: ε = 1e-3
• Maximum iterations: 200 (SQP), 30 (Quadratic Penalty)
• Penalty parameter: μ₀ = 1.0, β = 10.0
        """
        
        ax.text(0.1, 0.8, problem_text, fontsize=11, va='top')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Results Summary Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        ax.text(0.1, 0.9, 'Results Summary', fontsize=18, fontweight='bold')
        
        # Results table
        ax.text(0.1, 0.8, 'Constraint 1: x₁ + x₂ = 1', fontsize=14, fontweight='bold')
        ax.text(0.1, 0.75, 'Method          Solution              Iterations', fontsize=12, fontweight='bold')
        ax.text(0.1, 0.72, 'SQP             [0.6188, 0.3812]ᵀ     12', fontsize=11)
        ax.text(0.1, 0.69, 'Quadratic Penalty [0.6188, 0.3812]ᵀ   79', fontsize=11)
        
        ax.text(0.1, 0.6, 'Constraint 2: x₁² + x₂² = 1', fontsize=14, fontweight='bold')
        ax.text(0.1, 0.55, 'Method          Solution              Iterations', fontsize=12, fontweight='bold')
        ax.text(0.1, 0.52, 'SQP             [0.7864, 0.6177]ᵀ     10', fontsize=11)
        ax.text(0.1, 0.49, 'Quadratic Penalty [0.7864, 0.6177]ᵀ   55', fontsize=11)
        
        ax.text(0.1, 0.35, 'Key Findings:', fontsize=14, fontweight='bold')
        findings_text = """
• Both algorithms successfully converged to similar solutions
• SQP demonstrates faster convergence with fewer iterations
• Quadratic Penalty method shows more gradual convergence
• Both methods achieve similar final solution quality
• SQP is more computationally efficient per problem
• Choice depends on derivative availability and speed requirements
        """
        
        ax.text(0.1, 0.3, findings_text, fontsize=11, va='top')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Add all the plot pages
        for i, plot_file in enumerate(plot_files):
            if os.path.exists(plot_file):
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                # Load and display the plot
                img = plt.imread(plot_file)
                ax.imshow(img, aspect='auto', extent=[0.05, 0.95, 0.1, 0.9])
                
                # Add title
                titles = {
                    'sqp_constraint1_contour.png': 'SQP Optimization Path - Constraint 1',
                    'penalty_constraint1_contour.png': 'Quadratic Penalty Optimization Path - Constraint 1',
                    'convergence_comparison_constraint1.png': 'Convergence Comparison - Constraint 1',
                    'sqp_constraint2_contour.png': 'SQP Optimization Path - Constraint 2',
                    'penalty_constraint2_contour.png': 'Quadratic Penalty Optimization Path - Constraint 2',
                    'convergence_comparison_constraint2.png': 'Convergence Comparison - Constraint 2'
                }
                
                ax.text(0.5, 0.95, titles.get(plot_file, plot_file), 
                       fontsize=14, fontweight='bold', ha='center')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        # Conclusion Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        ax.text(0.1, 0.9, 'Conclusion', fontsize=18, fontweight='bold')
        
        conclusion_text = """
Both SQP and Quadratic Penalty methods successfully solved the constrained 
optimization problems with the Rosenbrock function.

SQP Advantages:
• Faster convergence (fewer iterations)
• Direct handling of constraints
• Good theoretical properties

SQP Disadvantages:
• Requires second-order derivatives
• More complex implementation
• Potential numerical issues with Hessian

Quadratic Penalty Advantages:
• Simpler implementation
• Only requires first-order derivatives
• Robust convergence properties

Quadratic Penalty Disadvantages:
• Slower convergence
• Requires tuning penalty parameters
• May have numerical conditioning issues

The choice between methods depends on the specific requirements of the 
application, including computational resources, derivative availability, 
and convergence speed requirements.

Both algorithms demonstrated robust performance and achieved high-quality 
solutions for both linear and nonlinear constraint configurations.
        """
        
        ax.text(0.1, 0.8, conclusion_text, fontsize=11, va='top')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print("✅ Report generated successfully!")
    print("📄 Generated: Assignment_3_report.pdf")
    print("📊 Included plots:")
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            print(f"   ✓ {plot_file}")
        else:
            print(f"   ✗ {plot_file} (missing)")

if __name__ == "__main__":
    create_report_pdf()
