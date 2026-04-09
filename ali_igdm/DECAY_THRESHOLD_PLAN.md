# Decay Threshold Mode Implementation Plan

## Overview

Add an optional `threshold_mode` parameter that enables exponential decay (factor 0.97) when sensor readings stay below the current threshold. The existing behavior remains the default.

---

## Changes

### 1. `igdm.py` — New ROS Parameter

- Declare parameter: `self.declare_parameter('threshold_mode', 'default')`
- Cache the value in `self.params['threshold_mode']`
- Accepted values:
  - `'default'` — existing behavior (threshold only increases on high readings)
  - `'decay'` — new behavior (threshold *= 0.97 each step without a high reading)

### 2. `sensor_model_discrete.py` — Modify `DiscreteSensorModel`

**Constructor (`__init__`)**:
- Add new parameter: `threshold_mode='default'`
- Store as `self.threshold_mode`

**New method `update_threshold_decay(self, current_measurement)`**:

```python
def update_threshold_decay(self, current_measurement):
    if self.threshold is None:
        self.initialize_threshold(current_measurement)
    elif current_measurement > self.threshold:
        # Normal weighted update (same as existing)
        old_threshold = self.threshold
        self.threshold = (self.threshold_weight * current_measurement +
                        (1 - self.threshold_weight) * self.threshold)
        scale_factor = self.threshold / old_threshold if old_threshold > 0 else 1.0
        self.level_thresholds = [t * scale_factor for t in self.level_thresholds]
    else:
        # Decay: threshold *= 0.97
        self.threshold *= 0.97
        # Level thresholds decay proportionally
        self.level_thresholds = [t * 0.97 for t in self.level_thresholds]
```

### 3. `igdm.py` — Wire It Into `take_step()`

In the `take_step()` method (around the sensor processing section), update the call:

**Before** (current):
```python
self.sensor_model.update_threshold(self.sensor_raw_value)
```

**After**:
```python
if self.params['threshold_mode'] == 'decay':
    self.sensor_model.update_threshold_decay(self.sensor_raw_value)
else:
    self.sensor_model.update_threshold(self.sensor_raw_value)
```

Also log the mode at initialization for clarity.

---

## Files Modified

| File | Change |
|------|--------|
| `src/base/ali_igdm/ali_igdm/igdm.py` | Add `threshold_mode` parameter, wire decay call in `take_step()` |
| `src/base/ali_igdm/ali_igdm/estimation/sensor_model_discrete.py` | Add `threshold_mode` param + `update_threshold_decay()` method |

---

## Usage

```bash
# Default behavior (unchanged)
ros2 run ali_igdm start --ros-args -p scenario_name:=curved_labrinth_left

# With decay mode
ros2 run ali_igdm start --ros-args -p scenario_name:=curved_labrinth_left -p threshold_mode:=decay
```

---

## Design Decisions

1. **Non-breaking**: Existing behavior unchanged when `threshold_mode` is not specified (defaults to `'default'`).
2. **Same initialization**: First measurement initializes identically in both modes (`max(measurement, 0.1)`, level thresholds at 0.25x/0.50x/0.75x/1.00x).
3. **No minimum floor**: Threshold can decay toward zero (no lower bound clamp).
4. **Proportional decay**: Both the main threshold and all 4 level thresholds decay by the same 0.97 factor to maintain relative spacing.
5. **Decay factor**: 0.97 per step without a high reading.
