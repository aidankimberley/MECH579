# SQP vs Scipy SLSQP Comparison Results

## Problem Statement
Optimize Breguet range for aircraft cruise with constraints:
- **Maximize**: Range (minimize -Range)
- **Subject to**: 
  - Velocity ≤ 150 m/s
  - Altitude ≤ 20,000 m

## Initial Guess
- V₀ = 130.0 m/s
- h₀ = 12,000.0 m

## Results Comparison

| Metric | Custom SQP | Scipy SLSQP | Difference |
|--------|------------|-------------|------------|
| **Optimal Velocity (m/s)** | 130.02 | 150.00 | 19.98 m/s |
| **Optimal Altitude (m)** | 12,007.81 | 11,271.98 | 735.83 m |
| **Maximum Range (m)** | 2,046,041 | **2,567,093** | **521,052 m (25.5%)** |
| **Iterations** | 500 (max) | 5 | 100x more! |
| **Convergence** | No (hit max_iter) | Yes | ✓ |
| **V Constraint Status** | -19.98 (slack) | -0.00 (active) | Not at boundary |
| **h Constraint Status** | -7,992 (slack) | -8,728 (slack) | Both satisfied |

## Key Findings

### 1. Custom SQP Not Converging to True Optimal
- **Issue**: Custom SQP stops at V = 130 m/s, well below the constraint boundary
- **True Optimal**: At V = 150 m/s (velocity constraint is active/binding)
- **Impact**: 25.5% suboptimal range (missing 521 km of range!)

### 2. Scipy SLSQP is Much More Efficient
- Scipy converges in **5 iterations** vs 500+ for custom SQP
- Scipy correctly identifies V = 150 m/s as the optimal (constraint boundary)
- Scipy properly handles the active constraint

### 3. Why Custom SQP Struggles

#### Possible Issues:
1. **Active-Set Strategy**: May not be aggressive enough in activating the velocity constraint
2. **Line Search**: Merit function may be too conservative
3. **BFGS Hessian Approximation**: May not be accurate enough for this problem
4. **Convergence Criteria**: "KKT norm not decreasing significantly" exits too early

#### Evidence:
```
Final Lagrange multipliers: [0, 41.24]
```
- λ₁ = 0 → altitude constraint inactive (correct)
- λ₂ = 41.24 → velocity constraint should be more active
- But V = 130 is still 20 m/s away from the constraint boundary!

## Theoretical Analysis

For the Breguet range equation with these constraints, the optimal should be at:
- **Maximum feasible velocity** (V = 150 m/s) because:
  - Range ∝ V/cₜ × (L/D) × ln(Wᵢ/Wf)
  - Higher velocity generally improves range (within constraint limits)
  - The constraint V ≤ 150 is the binding constraint

**Scipy Found This**, Custom SQP Did Not.

## Recommendations

### For Production Use:
✅ **Use Scipy's SLSQP** - It's faster, more robust, and finds the true optimal

### For Learning/Development:
If you want to improve your custom SQP:

1. **Looser Active-Set Criteria**: Make it easier for constraints to become active
   ```python
   active_ineq_mask = (c_ineq_vals >= -eps * 100) | (lamb_ineq > eps * 0.01)
   ```

2. **Less Aggressive Early Exit**: Remove or weaken the "KKT norm not decreasing" check
   ```python
   # Lines 411-414 in SQP.py - this may exit too early
   ```

3. **Try Different Initial Guess**: Start at V = 145 m/s (close to constraint)

4. **Use Exact Hessian** Instead of BFGS (set `BFGS=False`)

## Conclusion

Your inequality constraint fix **is working correctly** - the custom SQP now respects constraints properly (no violations). However, it's **not converging efficiently** to the true optimal for this specific problem.

**For your assignment**: I recommend reporting both results and discussing why the custom SQP struggles with this particular optimization landscape.

