# SQP Inequality Constraint Fix Summary

## Problem
Your SQP algorithm was **not properly handling inequality constraints**. The optimization was returning:
- Optimal Velocity: **192.22 m/s** (violates V ≤ 150 m/s constraint)
- Optimal Altitude: 14802.45 m
- The algorithm incorrectly reported "Both constraints satisfied"

## Root Cause
The SQP implementation had several critical issues with inequality constraints:

1. **Inequality constraints were not included in the QP subproblem**: The KKT system only included equality constraints when computing search directions
2. **Lagrange multipliers for inequality constraints were never updated**: Line 260 had `plamb_ineq = jnp.zeros(p)` with comment "No step for inequality constraints for now"
3. **No active-set strategy**: There was no mechanism to identify and enforce active inequality constraints

## Solution
Implemented a proper **active-set SQP approach**:

### 1. Active Constraint Identification (Lines 217-228)
```python
# Identify active inequality constraints (violated or nearly active)
if p > 0:
    active_ineq_mask = (c_ineq_vals >= -eps * 10) | (lamb_ineq > eps * 0.1)
else:
    active_ineq_mask = jnp.array([], dtype=bool)
```

### 2. Include Active Constraints in KKT System (Lines 230-256)
- Combine equality constraints with active inequality constraints
- Build augmented KKT matrix including both types of constraints
- This forces the search direction to respect active inequality constraints

### 3. Update Multipliers for Active Constraints (Lines 279-301)
- Extract multiplier updates from KKT solution
- Update only active inequality multipliers
- Enforce non-negativity: `lamb_ineq_new = jnp.maximum(0, lamb_ineq + alpha * plamb_ineq)`

### 4. Complementarity Enforcement (Lines 507-513)
- Reset multipliers for inactive constraints to maintain complementarity
- `lamb_ineq = jnp.where(inactive_mask, 0.0, lamb_ineq)`

## Results - Before vs After

### BEFORE (Broken):
```
Optimal Velocity: 192.22 m/s  ❌ VIOLATES V ≤ 150 constraint
Optimal Altitude: 14802.45 m  ✓
Velocity constraint violation: +42.22 m/s
```

### AFTER (Fixed):
```
Optimal Velocity: 112.74 m/s  ✓ SATISFIES V ≤ 150 constraint
Optimal Altitude: 12547.70 m  ✓
Velocity constraint: -37.26 m/s (safely within bounds)
Altitude constraint: -7452.30 m (safely within bounds)
```

## Verification
Created `test_sqp_inequality.py` which verifies the fix with a simple test case:
- Objective: minimize (x0-2)² + (x1-3)²
- Constraints: x0 ≤ 1.5, x1 ≤ 2.5
- Expected optimal: [1.5, 2.5]
- **Result: ✓ PASS** - Found exact optimal with both constraints active

## Files Modified
1. `SQP.py` - Fixed inequality constraint handling (~150 lines modified)
2. `brequet_range_optimizer.py` - Fixed import-time execution issue
3. `constraint_range_optimizer.py` - Commented out crash-causing plot

## Technical Notes
The implementation uses an **active-set method** where:
- A constraint is considered "active" if it's violated (c_i ≥ -ε) OR has positive multiplier (λ_i > ε)
- Active inequality constraints are treated as temporary equality constraints in the QP subproblem
- Multipliers are projected to non-negative values (KKT complementarity condition)
- Inactive constraint multipliers are reset to zero

This is a standard approach in constrained optimization and ensures that inequality constraints are properly respected during optimization.

