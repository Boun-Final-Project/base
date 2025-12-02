# Early Convergence Problem & Solutions

## Problem Description ⚠️

Your system shows **premature convergence** - the particle filter collapses to a confident estimate too early, before the robot has explored enough of the environment.

### Symptoms
- Search completes after just a few steps
- Std Dev drops below threshold quickly (e.g., 0.553 < 0.6)
- Particles cluster in one area
- Estimated source location may be incorrect
- Robot stops searching prematurely

### Example from Your Screenshot
```
Std Dev: 0.553
Search: ✓ COMPLETE
Steps: ~5-10 (too few!)
```

Even though σ=0.553 < 0.6, the particles have falsely converged to the wrong location.

---

## Why This Happens 🔍

### Root Cause 1: **Particle Depletion with Binary Sensors**

Binary sensors (0/1) create a **harsh selection pressure**:

```python
# Binary sensor reading
if measurement > threshold:
    binary = 1  # Gas detected
else:
    binary = 0  # No gas
```

**Problem**:
- Particles matching the reading get high weight
- Particles not matching get ZERO weight and die
- After a few measurements, only a small cluster survives
- Small cluster → Low std dev → FALSE convergence

### Root Cause 2: **Insufficient Exploration**

If robot gets similar readings in a local area (e.g., all 0s), particles can converge to that region without exploring alternatives.

### Root Cause 3: **Threshold Too High**

Original `sigma_threshold = 0.6` is too lenient - allows convergence with particles still moderately spread.

---

## Solutions Implemented ✅

### Solution 1: **Lower Convergence Threshold**

**Changed**: `sigma_threshold: 0.6` → `0.35`

```python
# Before (line 38)
self.declare_parameter('sigma_threshold', 0.6)  # Too lenient

# After
self.declare_parameter('sigma_threshold', 0.35)  # Stricter convergence
```

**Effect**: Requires particles to be much more tightly clustered before declaring convergence.

**Paper reference**: Table I shows σ_t = 0.7m, but for your environment 0.35m is more appropriate.

---

### Solution 2: **Minimum Steps Requirement**

**Added**: Prevent convergence before robot has explored sufficiently

```python
# New parameters (lines 46-48)
self.declare_parameter('min_steps_before_convergence', 10)
```

**Implementation** (`is_estimation_converged()`, line 423):
```python
# Require BOTH conditions
std_dev_converged = sigma_p < sigma_threshold
enough_steps = self.step_count >= min_steps

converged = std_dev_converged AND enough_steps
```

**Effect**: Forces robot to take at least 10 measurements before convergence is allowed, even if std dev is low.

**Adjustable**: Increase to 20-30 for larger environments.

---

### Solution 3: **Better Logging & Warnings**

**Added early convergence warning** (line 453):
```python
if std_dev_converged and not enough_steps:
    self.get_logger().warn(
        f'Early convergence prevented: σ_p={sigma_p:.3f} < {sigma_threshold:.3f} '
        f'but only {self.step_count} steps (need {min_steps})'
    )
```

**You'll see**:
```
[WARN] Early convergence prevented: σ_p=0.553 < 0.350 but only 5 steps (need 10)
```

This helps you understand when the safeguard is activating.

---

## Parameter Tuning Guide 🔧

### `sigma_threshold` (Convergence Threshold)

**Default**: `0.35` meters

**How to adjust**:
```bash
ros2 run efe_igdm start --ros-args -p sigma_threshold:=0.3
```

**Guidelines**:
- **Too high** (0.6+): Early convergence, false positives
- **Too low** (<0.2): May never converge, search takes forever
- **Recommended**:
  - Small rooms: 0.3
  - Large areas: 0.4-0.5
  - Paper used: 0.7 (but with different sensor model)

**Rule of thumb**: Set to ~5-10% of typical room dimension

---

### `min_steps_before_convergence` (Minimum Steps)

**Default**: `10` steps

**How to adjust**:
```bash
ros2 run efe_igdm start --ros-args -p min_steps_before_convergence:=20
```

**Guidelines**:
- **Too low** (<5): Doesn't prevent early convergence
- **Too high** (>50): Wastes time if already converged correctly
- **Recommended**:
  - Simple environments: 10-15
  - Complex environments: 20-30
  - Very large areas: 30-50

**Rule of thumb**: Should allow robot to traverse ~1-2 RRT tree ranges

---

## Monitoring Convergence 📊

### Console Output

**Normal convergence** (good):
```
[DEBUG] Convergence check: σ_p=0.320, σ_t=0.350, steps=25/10, converged=True
[INFO] Estimation converged! σ_p = 0.320 < σ_t = 0.350 after 25 steps
```

**Early convergence prevented** (safeguard working):
```
[DEBUG] Convergence check: σ_p=0.280, σ_t=0.350, steps=5/10, converged=False
[WARN] Early convergence prevented: σ_p=0.280 < 0.350 but only 5 steps (need 10)
```

### RViz Indicators

Watch for:
1. **Text Overlay**: Shows current step count
2. **Particles**: Should spread across environment initially
3. **Search Status**: "⟳ SEARCHING" until properly converged

### CSV Logs

Check `~/igdm_logs/igdm_log_*.csv`:

```python
import pandas as pd
df = pd.read_csv('igdm_log_*.csv')

# Plot convergence over time
plt.plot(df['step'], df['std_dev_x'], label='σ_x')
plt.plot(df['step'], df['std_dev_y'], label='σ_y')
plt.axhline(y=0.35, color='r', linestyle='--', label='Threshold')
plt.xlabel('Step')
plt.ylabel('Std Dev (m)')
plt.legend()
plt.show()
```

Look for **gradual decrease**, not sudden drop!

---

## Additional Solutions (If Still Converging Early)

### Option A: Increase Particle Count

More particles = better coverage = less susceptible to depletion

```bash
ros2 run efe_igdm start --ros-args -p number_of_particles:=2000
```

**Trade-off**: Higher computational cost

---

### Option B: Stronger MCMC Diversity

The particle filter already uses MCMC for diversity. If needed, you can increase the MCMC step size by modifying `particle_filter_optimized.py`:

```python
# In __init__ (line 36-40)
self.mcmc_std = {
    'x': 0.10 * x_range,  # Increased from 0.05
    'y': 0.10 * y_range,  # Increased from 0.05
    'Q': 0.05 * Q_range
}
```

**Effect**: Particles explore more aggressively after resampling.

---

### Option C: Rogue Particle Injection

Re-enable random particle injection during resampling (currently disabled to match paper).

In `particle_filter_optimized.py`, line 316-321:

```python
def _resample(self):
    # After systematic resampling, inject 5% random particles
    num_rogue = int(0.05 * self.N)
    if num_rogue > 0:
        # Replace lowest weight particles with random ones
        self.particles[-num_rogue:] = self._initialize_particles()[-num_rogue:]
```

**Effect**: Maintains exploration even after convergence starts.

**Trade-off**: Violates strict particle filter theory (not in paper).

---

## Verification Checklist ✅

After applying fixes, verify:

- [ ] Search takes at least `min_steps_before_convergence` steps
- [ ] Std dev decreases **gradually** over time (check CSV logs)
- [ ] Early convergence warning appears if σ drops too fast
- [ ] Final estimate is close to true source (< 1m error)
- [ ] Particles spread across environment before clustering
- [ ] Entropy decreases smoothly (no sudden drops)

---

## Expected Behavior (After Fix)

### Good Search Pattern

```
Step 1:  σ_p=2.500 → Particles spread across map
Step 5:  σ_p=1.800 → Starting to narrow down
Step 10: σ_p=1.200 → Converging to region
Step 15: σ_p=0.700 → Getting closer
Step 20: σ_p=0.400 → Almost there
Step 25: σ_p=0.320 → ✓ CONVERGED (correct!)
```

### Bad Search Pattern (Early Convergence)

```
Step 1: σ_p=2.500 → Particles spread
Step 2: σ_p=0.800 → Sudden drop! (bad)
Step 3: σ_p=0.400 → Too fast!
Step 5: σ_p=0.280 → Would converge (but prevented by min_steps)
```

---

## When to Adjust Parameters

### Environment is Small
```bash
sigma_threshold: 0.25
min_steps_before_convergence: 5-10
```

### Environment is Large
```bash
sigma_threshold: 0.4-0.5
min_steps_before_convergence: 20-30
```

### High Confidence Needed
```bash
sigma_threshold: 0.2-0.25
min_steps_before_convergence: 30-50
```

### Fast Convergence Acceptable
```bash
sigma_threshold: 0.5
min_steps_before_convergence: 5
```

---

## Summary

**Changes Made**:
1. ✅ Lowered `sigma_threshold` from 0.6 → 0.35
2. ✅ Added `min_steps_before_convergence` = 10
3. ✅ Enhanced logging with warnings
4. ✅ Debug output shows step count

**Result**: Robot will now:
- Explore for at least 10 steps before converging
- Require tighter particle clustering (σ < 0.35)
- Warn you if trying to converge too early
- Provide better visibility into convergence status

**Next Steps**:
1. Run experiment with new parameters
2. Monitor console for "Early convergence prevented" warnings
3. Check CSV logs to verify gradual convergence
4. Adjust thresholds based on your specific environment

---

## References

- Paper Table I: σ_t = 0.7m (but different sensor model)
- Algorithm 1 (line 22): Convergence condition
- `particle_filter_optimized.py`: MCMC implementation
- `TROUBLESHOOTING.md`: Additional debugging tips
