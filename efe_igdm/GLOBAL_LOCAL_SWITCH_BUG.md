# Bug: Premature GLOBAL Mode Re-trigger After LOCAL Switch

## Symptom

Robot plans a global path with N nodes, reaches a waypoint, switches to LOCAL mode, takes **1 local step**, then immediately switches back to GLOBAL targeting the **same frontier**.

## Root Cause

When GLOBAL → LOCAL switch happens (due to high MI at a waypoint), `_handle_settling_complete` resets the `DeadEndDetector` to its initial state. The detector then fires after just **1–2 local steps**, before the robot has meaningfully explored.

### Step-by-step trace:

1. Robot in `GLOBAL`, MI > threshold at waypoint → `settling_start_time` set
2. `_handle_settling_complete` → switches to `LOCAL`, clears `global_path`, **resets dead-end detector**
3. **Step 1 LOCAL**: `is_dead_end()` called → `initialized=False` → sets threshold from BI₁, returns `False`
4. **Step 2 LOCAL**: `is_dead_end()` checks BI₂ < threshold(BI₁) → fires `True` if BI dropped
5. `_handle_dead_end_transition` → same frontiers detected (robot barely moved) → same frontier re-selected → back to `GLOBAL`

The dead-end detector expects to be "warm" from prior local steps. Resetting it cold on every GLOBAL→LOCAL switch makes it oversensitive.

## Proposed Fix

### Option A — Minimum local step gate (simple, recommended)

Add a `min_local_steps_before_dead_end` parameter (suggest **5–10 steps**). In `_run_local_planning`, skip `is_dead_end()` until the step counter reaches the minimum.

**State variable** (`_init_state_variables`):
```python
self.local_steps_since_mode_switch = 0
```

**Reset in every LOCAL entry point:**
- `_handle_settling_complete` → add `self.local_steps_since_mode_switch = 0`
- `_handle_dead_end_transition` (LOCAL fallback) → same
- `trigger_recovery` (LOCAL fallback) → same

**Gate in `_run_local_planning`:**
```python
self.local_steps_since_mode_switch += 1
if self.params['enable_global_planner']:
    if self.local_steps_since_mode_switch >= self.params['min_local_steps_before_dead_end']:
        dead_end_detected = self.dead_end_detector.is_dead_end(bi_optimal)
```

### Option B — Don't reset detector on GLOBAL→LOCAL (more principled)

Only reset the `DeadEndDetector` when switching **LOCAL→GLOBAL**, not the other direction. When re-entering LOCAL, the threshold is already warmed up from prior exploration. Remove the `dead_end_detector.reset()` call from `_handle_settling_complete`.

> Both options can be combined: keep the warm-threshold approach (Option B) AND add a short minimum step count (e.g., 3) as a safety floor.

## Files to Modify

- [`igdm.py`](file:///home/efe/ros2_ws/src/base/efe_igdm/efe_igdm/igdm.py) — only file that needs changes
  - `_init_state_variables`: add counter
  - `_init_parameters`: add `min_local_steps_before_dead_end`
  - `_handle_settling_complete`: reset counter (+ optionally remove detector reset)
  - `_handle_dead_end_transition`: reset counter in LOCAL fallback branch
  - `trigger_recovery`: reset counter in LOCAL fallback branch
  - `_run_local_planning`: increment counter, gate dead-end check

## Verification

After the fix, logs should show at least `min_local_steps_before_dead_end` LOCAL steps before any `[DEAD END]` transition:

```bash
ros2 launch efe_igdm <launch_file>.py 2>&1 | grep -E "\[GLOBAL MODE\]|\[MODE SWITCH\]|\[DEAD END\]|\[SWITCH\]"
```
