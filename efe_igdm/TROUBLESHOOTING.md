# Troubleshooting Guide - Dead End Detection & Visualization

## Issue 1: Not Seeing Color-Coded Branches ❌

### Symptoms
- All RRT paths in RViz appear **gray**
- No red/yellow/green gradient visible
- All paths look the same

### Possible Causes & Solutions

#### Cause 1: Utilities Are Too Similar
**What's happening**: All entropy gains are nearly identical, so after normalization they all map to the same color.

**Check console logs**:
```bash
ros2 run efe_igdm start | grep VIZ
```

Look for:
```
[VIZ] Path utilities: min=0.4500, max=0.4503, range=0.0003
[VIZ] All utilities are identical - using uniform coloring
```

**Solution**: This is actually normal in some situations:
- Early in search when all directions are equally informative
- In open spaces with no obstacles
- When particles are very spread out

**Not a bug if**: Search progresses normally and eventually finds the source.

#### Cause 2: No Utilities Being Passed
**Check console logs**:
```
[VIZ] No utilities provided - using gray coloring
```

**Solution**: This means `all_utilities` is not in `debug_info`. Check that RRT's `get_next_move_debug()` returns it.

**Fix**: Make sure line 609 in `igdm.py` is:
```python
self.visualize_all_paths(all_paths, debug_info.get("all_utilities", None))
```

#### Cause 3: RViz Display Settings
**Check**:
1. In RViz, select the "All Paths" display
2. Check "Color" property - should be from topic
3. Alpha should be > 0.5
4. Make sure display is enabled

**Fix**: Delete and re-add the display:
- Remove `/rrt_infotaxis/all_paths` display
- Add → By topic → `/rrt_infotaxis/all_paths` → MarkerArray

### How to Verify Colors Are Working

Run the test script:
```bash
cd ~/ros2_ws/src/base/efe_igdm/efe_igdm
python3 test_color_mapping.py
```

This shows:
- What colors different utilities should produce
- Example scenarios
- Saved image: `color_mapping_test.png`

### Expected Behavior

**Healthy diversity** (you SHOULD see colors):
```
[VIZ] Path utilities: min=0.3800, max=0.6100, range=0.2300
```
→ Should see mix of red, yellow, green paths

**Low diversity** (colors will be similar):
```
[VIZ] Path utilities: min=0.4950, max=0.5050, range=0.0100
```
→ All paths will be yellowish (this is OK!)

**Dead end** (should see mostly red):
```
[VIZ] Path utilities: min=0.0100, max=0.1500, range=0.1400
```
→ Mostly red/orange paths

---

## Issue 2: Early Convergence / "COMPLETE" Too Soon ❌

### Symptoms
- Text shows "Search: ✓ COMPLETE"
- But Std Dev is still high (> threshold)
- Robot stops searching prematurely

### Diagnosis

**Check the console**:
```bash
ros2 run efe_igdm start | grep "Convergence check"
```

Should show:
```
[DEBUG] Convergence check: σ_p = 1.111, σ_t = 0.600, converged = False
```

**If it shows `converged = True` but σ_p > σ_t**: Bug in convergence check!

### Solution 1: Check Threshold Parameter

```bash
ros2 param get /rrt_infotaxis_node sigma_threshold
```

Should return: `0.6` (or your desired value)

**If too high**: Adjust in launch file or command line:
```bash
ros2 run efe_igdm start --ros-args -p sigma_threshold:=0.6
```

### Solution 2: Verify Text Overlay Shows Correct Status

The text visualizer should show real-time status. If it's stuck on "COMPLETE":

**Check**: Is `search_complete` flag being set correctly?

**Debug**:
```python
# In igdm.py, check around line 690
if self.is_estimation_converged():
    self.get_logger().info('SOURCE SEARCH COMPLETED SUCCESSFULLY!')
    # ...
    self.search_complete = True
    return
```

**Key**: The flag `self.search_complete` stays `True` forever once set. This is intentional - search should stop once converged.

### Solution 3: Particle Filter Not Converging

**If particles stay spread out**:

1. **Check sensor readings**:
   ```bash
   ros2 topic echo /fake_pid/Sensor_reading
   ```
   Should show varying values as robot moves

2. **Check particle filter is updating**:
   - Entropy should decrease over time
   - Std Dev should decrease
   - Particles should cluster

3. **Increase particle count** (if needed):
   ```bash
   ros2 run efe_igdm start --ros-args -p number_of_particles:=2000
   ```

---

## Issue 3: Dead End Detection Not Triggering 🟡

### Symptoms
- Robot gets stuck in corners/rooms
- Status always shows "✓ OK"
- Never see "⚠ DEAD END!"

### Diagnosis

**Check BI values in text overlay**:
```
--- Dead End Detect ---
BI*: 0.350
Threshold: 0.760
Margin: -0.410
Status: ✓ OK  ← Should be "⚠ DEAD END!"
```

**If Margin is negative but Status is OK**: Bug!

### Solution 1: Check Epsilon Parameter

```bash
ros2 param get /rrt_infotaxis_node dead_end_epsilon
```

**If too low** (< 0.7): Threshold adapts too quickly
**If too high** (> 0.95): Threshold adapts too slowly

**Recommended**: 0.85

### Solution 2: Verify Detector is Running

Add this to check:
```bash
ros2 run efe_igdm start 2>&1 | grep -E "(DEAD END|BI\*)"
```

Should see:
```
[PLAN] BI*=0.450 >= BI_thresh=0.420
[PLAN] BI*=0.380 >= BI_thresh=0.415
[WARN] [DEAD END DETECTED] BI*=0.250 < BI_thresh=0.390
```

### Solution 3: Not Actually a Dead End

**In open spaces**, BI* might stay high:
- Plenty of informative paths available
- No obstacles blocking RRT
- Robot finding source normally

**This is good!** Dead end detection should only trigger when actually stuck.

---

## Issue 4: RViz Not Showing Markers 🔴

### Symptoms
- No particles visible
- No paths visible
- Text overlay missing

### Solutions

1. **Check topics are publishing**:
   ```bash
   ros2 topic list | grep rrt_infotaxis
   ros2 topic hz /rrt_infotaxis/particles
   ```

2. **Check Fixed Frame**:
   - In RViz, Global Options → Fixed Frame → `map`

3. **Add missing displays**:
   - Panels → Displays → Add
   - By topic → Select `/rrt_infotaxis/*`

4. **Check marker lifetime**:
   - Markers should not expire (lifetime = 0)
   - If blinking, increase update rate

5. **Reset RViz**:
   ```bash
   # Save config first!
   # Then: View → Reset to Default Layout
   ```

---

## Debugging Checklist ✅

Use this to diagnose issues systematically:

### Color-Coded Branches
- [ ] Run `test_color_mapping.py` - colors work in test?
- [ ] Check console for `[VIZ]` messages
- [ ] Verify `all_utilities` has variation (range > 0.01)
- [ ] RViz display shows Alpha > 0.5
- [ ] Fixed Frame is `map`

### Dead End Detection
- [ ] Text overlay shows BI* and Threshold values
- [ ] Margin calculation is correct (BI* - Threshold)
- [ ] Console shows `[DEAD END DETECTED]` when stuck
- [ ] Epsilon parameter is reasonable (0.85)
- [ ] BI* actually drops when in corners/dead ends

### Early Convergence
- [ ] Std Dev shown in text matches actual particle spread
- [ ] Convergence threshold is correct (default: 0.6)
- [ ] Search status changes from SEARCHING → COMPLETE only when converged
- [ ] Console shows convergence message when it happens
- [ ] Particle filter is updating (entropy decreasing)

### General RViz Issues
- [ ] All topics are publishing (`ros2 topic hz`)
- [ ] Fixed Frame is `map`
- [ ] Displays are enabled (checkbox checked)
- [ ] Markers have correct namespace
- [ ] Time is synchronized (not stale)

---

## Quick Fixes 🔧

### 1. Increase Color Contrast
Edit `igdm.py` line 297, change exponent:
```python
norm_util_enhanced = norm_util ** 0.3  # Lower = more contrast
```

### 2. Force Dead End Detection (Testing)
Edit `dead_end_detector.py` line ~45:
```python
self.bi_threshold = 10.0  # Very high - will always trigger
```

### 3. Disable Dead End Detection Temporarily
```bash
ros2 run efe_igdm start --ros-args -p dead_end_epsilon:=0.999
```

### 4. Verbose Logging
Enable DEBUG level:
```bash
ros2 run efe_igdm start --ros-args --log-level DEBUG
```

---

## Getting Help 📖

If still stuck, check:

1. **Logs**: `~/igdm_logs/igdm_log_*.csv`
2. **Analysis**: `python3 analyze_dead_end.py <log_file>`
3. **Documentation**:
   - `DEAD_END_DETECTION.md`
   - `VISUALIZATION_GUIDE.md`
   - `CLAUDE.md`

## Common Misunderstandings 💡

### "All paths are the same color - is this a bug?"
**Maybe not!** If all RRT branches have similar entropy gain, they SHOULD be the same color. This happens:
- In open spaces
- Early in search
- When far from source

### "Dead end detection never triggers"
**Could be good!** If your environment is open and the robot never gets stuck, dead ends won't be detected. Try running in a more complex environment (rooms, corridors).

### "Search completes but Std Dev is still 1.0"
**This IS a bug!** Convergence should require `max(σ_x, σ_y) < 0.6`. File an issue with your log files.
