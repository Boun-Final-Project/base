# EFE IGDM - Improvement Suggestions Report

Analysis of the no-wind GSL implementation (`igdm_basic.py`) based on Kim et al. IEEE RA-L 2025.

---

## 1. Particle Filter Robustness

### 1.1 Weight Arithmetic Should Use Log-Space

**Problem:** Weights are accumulated multiplicatively (`self.weights *= likelihoods`). After many steps, this causes floating-point underflow regardless of normalization. Combined with the minimum likelihood floor of `1e-50`, weights become subnormal floats long before the `weight_sum == 0` guard triggers.

**Suggestion:** Work in log-weight space internally. Store `log_weights`, accumulate via addition (`log_weights += log_likelihoods`), and only exponentiate for resampling and entropy calculations. This is standard practice in sequential Monte Carlo and eliminates underflow entirely.

**Impact:** Prevents silent numerical degeneracy in long-running searches (>50 steps).

**Files:** `estimation/particle_filter.py`

### 1.2 MCMC Rejuvenation Uses Only the Last Measurement

**Problem:** The Metropolis-Hastings accept/reject ratio in `_mcmc_move` is computed using only `self.last_measurement` and `self.last_sensor_position`. In proper sequential Monte Carlo, the MCMC kernel should target the full posterior (product of all likelihoods seen so far), not just the most recent one. Using only the last measurement allows particles to drift away from the accumulated posterior.

**Suggestion:** Store the full measurement history and compute the acceptance ratio using all observations. Alternatively, use the current particle weight as a proxy for the posterior and compare `w_proposed / w_current` as the acceptance ratio.

**Impact:** Improves particle diversity without sacrificing posterior consistency.

**Files:** `estimation/particle_filter.py`

### 1.3 MCMC Proposal Standard Deviation is Too Large

**Problem:** `mcmc_std` defaults to 10% of the search range. For a 20m x 12m room, this gives `std_x = 2.0m`, `std_y = 1.2m`. Such large proposals lead to very low acceptance rates, making the MCMC move effectively a no-op most of the time.

**Suggestion:** Use an adaptive proposal based on the current particle spread (e.g., `mcmc_std = 0.5 * weighted_std`). This automatically tightens as the filter converges and stays broad during exploration.

**Impact:** Better particle diversity without wasted computation on rejected proposals.

**Files:** `estimation/particle_filter.py`

### 1.4 Particles Initialized in Unknown/Occupied Cells

**Problem:** `_initialize_particles` places particles in any cell where `grid != 1`, including unknown cells (`-1`). Particles in unknown cells get infinite Dijkstra distance, producing zero predicted concentration. These particles contribute nothing but dilute the effective sample size.

**Suggestion:** Only initialize particles in free cells (`grid == 0`), or add a post-initialization filter that removes particles with invalid positions. As the SLAM map grows, newly discovered free cells can be populated via the MCMC move step.

**Impact:** More effective use of the particle budget, especially in early exploration when most cells are unknown.

**Files:** `estimation/particle_filter.py`

### 1.5 `compute_hypothetical_entropy` Returns Max Entropy on Weight Collapse

**Problem:** When hypothetical weights sum to zero, `log(N)` (maximum entropy) is returned. This biases mutual information upward for truly uninformative positions, potentially misleading the RRT into choosing paths toward locations where all likelihoods are zero.

**Suggestion:** Return the current entropy (i.e., no information gain) instead of max entropy when weight collapse occurs. This correctly signals that the position provides no useful discrimination.

**Impact:** Prevents MI over-estimation at boundary/wall positions.

**Files:** `estimation/particle_filter.py`

---

## 2. Local Planner (RRT-Infotaxis)

### 2.1 Incremental Node Position Array in `sprawl()`

**Problem:** `get_closest_node` rebuilds the position array from scratch on every iteration:
```python
all_positions = np.array([n.position for n in self.nodes])
```
For `N_tn=50` with up to 5000 iterations, this creates and discards thousands of temporary arrays.

**Suggestion:** Pre-allocate `self.node_positions = np.zeros((N_tn, 2))` and maintain a fill counter. Append new positions in-place: `self.node_positions[self.node_count] = new_pos`.

**Impact:** ~2-3x speedup in `sprawl()`, saving ~30-50ms per planning step.

**Files:** `planning/rrt.py`

### 2.2 Goal-Biased Sampling for RRT Tree

**Problem:** Tree sampling is uniform within a disk around the robot. This wastes vertices in low-information regions far from the particle distribution.

**Suggestion:** Mix uniform sampling (70%) with goal-biased sampling (30%) where goals are drawn from the particle filter's spatial distribution. This concentrates the tree toward likely source locations while maintaining exploration diversity.

**Impact:** Better branch information values with the same `N_tn` budget, leading to faster convergence.

**Files:** `planning/rrt.py`

### 2.3 `get_entropy()` Called Inside Branch Loop

**Problem:** `current_entropy = initial_particle_filter.get_entropy()` is called inside the per-node loop in `calculate_branch_information`, even though entropy is constant for the entire branch evaluation.

**Suggestion:** Hoist the call before the loop.

**Impact:** Minor CPU savings, cleaner code.

**Files:** `planning/rrt.py`

### 2.4 `positive_weight` Parameter is Dead Code

**Problem:** The parameter is stored (`self.positive_weight = positive_weight`) but never referenced anywhere in the RRT implementation. The CLAUDE.md documents it as an alias for `discount_factor`, but the code does not use it.

**Suggestion:** Remove the dead parameter or wire it to replace `discount_factor`.

**Impact:** Code clarity.

**Files:** `planning/rrt.py`

---

## 3. Dead End Detection

### 3.1 Negative BI Permanently Disables Detection

**Problem:** If `bi_optimal = -inf` (returned when no RRT paths are found), the threshold update becomes `epsilon * old + (1-epsilon) * (-inf) = -inf`. After this, `bi_threshold = -inf` and `is_dead_end` always returns `False`, permanently disabling dead end detection.

**Suggestion:** Clamp `bi_optimal` to zero before updating the threshold:
```python
bi_optimal_safe = max(bi_optimal, 0.0)
self.update_threshold(bi_optimal_safe)
```

**Impact:** Prevents the detector from becoming permanently broken after a single bad RRT step.

**Files:** `planning/dead_end_detector.py`

### 3.2 Single-Step Triggering is Fragile

**Problem:** A dead end is declared on a single step where `BI* < threshold`. A single unlucky RRT sample (e.g., tree grew into a corner) can trigger a mode switch, causing unnecessary global planning overhead.

**Suggestion:** Require 2-3 consecutive dead end detections before switching. Add a `consecutive_count` counter that resets when `BI* >= threshold`.

**Impact:** More robust mode switching, fewer unnecessary global planning calls.

**Files:** `planning/dead_end_detector.py`

### 3.3 Unused `BranchInformation` Helper Class

**Problem:** The `BranchInformation` class (lines 209-273 in `dead_end_detector.py`) reimplements BI calculation but is never used anywhere. The actual BI computation happens inline in `rrt.py`.

**Suggestion:** Remove the dead code.

**Impact:** Code clarity, reduced maintenance burden.

**Files:** `planning/dead_end_detector.py`

---

## 4. Global Planner

### 4.1 Fallback When All Frontier MI Values Are Near-Zero

**Problem:** `evaluate_frontier_vertices` drops frontiers with `mi <= 1e-6`. When the particle filter is well-converged (most particles near the estimated source), all distant frontier positions will have negligible MI. This eliminates all candidates, returning `'error': 'No reachable high-info frontiers'`, leaving the robot stuck.

**Suggestion:** When all MI-filtered candidates are empty, fall back to selecting the nearest reachable frontier by path cost alone (pure exploration without information gain).

**Impact:** Prevents the robot from getting stuck in GLOBAL mode with no valid target.

**Files:** `planning/global_planner.py`

### 4.2 PRM Path Collision Check is Not Vectorized

**Problem:** `_is_path_collision_free` uses a Python for-loop to sample points along each PRM edge. With `prm_connection_radius=5.0m` and `resolution=0.1m`, each edge requires up to 50 samples. With hundreds of KD-Tree pairs, this is a significant bottleneck.

**Suggestion:** Vectorize the line sampling (generate all sample points at once with `np.linspace`, convert to grid coordinates, batch-check occupancy). The RRT's `is_collision_free_vectorized` already demonstrates this pattern.

**Impact:** ~3-5x speedup in PRM construction.

**Files:** `planning/global_planner.py`

### 4.3 PRM Random Sampling Rejection Rate

**Problem:** Random samples are drawn uniformly from the world bounding box, then rejected if invalid. In cluttered environments, most of the bounding box is occupied or unknown, leading to high rejection rates and wasted `_is_valid_optimistic` calls.

**Suggestion:** Pre-extract a list of all free (and optionally unknown) grid cells at PRM construction time. Sample directly from this list by choosing random indices, then convert to world coordinates. Rejection rate drops to near zero.

**Impact:** Faster PRM construction, guaranteed sample count.

**Files:** `planning/global_planner.py`

### 4.4 `get_next_best_frontier` Can Stack Overflow

**Problem:** The recursive fallback `return self.get_next_best_frontier()` can overflow the Python stack if many consecutive frontiers have invalid path reconstructions.

**Suggestion:** Convert to a while-loop.

**Impact:** Prevents crash in edge cases.

**Files:** `planning/global_planner.py`

---

## 5. Mapping

### 5.1 LiDAR Ray Tracing is Pure Python

**Problem:** Bresenham ray tracing is implemented as a Python `while True` loop, called once per laser beam per scan. At 360 beams per scan and 10 Hz scan rate, this is ~360,000 Python loop iterations per second. This is the single hottest code path in the mapper.

**Suggestion:** Use Numba JIT compilation (already used for Dijkstra in `igdm_gas_model.py`) or `skimage.draw.line` for vectorized Bresenham. Alternatively, downsample the laser scan (process every 2nd-3rd beam) with negligible map quality loss.

**Impact:** Major CPU reduction in mapping, freeing cycles for planning.

**Files:** `mapping/lidar_mapper.py`

### 5.2 `is_valid` vs `_is_valid_optimistic` Inconsistency

**Problem:** Two validity checking functions exist with different semantics:
- `OccupancyGridMap.is_valid`: unknown cells (-1) are treated as **obstacles** (`!= 0`)
- `_is_valid_optimistic` (in igdm_basic.py and global_planner.py): unknown cells are treated as **free** (`> 0`)

The RRT uses the SLAM map which may call `is_valid` from the occupancy grid, while the global planner uses `_is_valid_optimistic`. This creates inconsistent collision semantics.

**Suggestion:** Consolidate into a single configurable validity check on `OccupancyGridMap` with an `optimistic` flag parameter, and use it consistently.

**Impact:** Eliminates subtle planning inconsistencies.

**Files:** `mapping/occupancy_grid.py`, `planning/global_planner.py`, `planning/rrt.py`

### 5.3 No Minimum Map Coverage Before Planning

**Problem:** Planning begins as soon as `take_step` fires, even if no laser scans have been processed yet. If the SLAM map is completely unknown, `is_valid` rejects all positions and the RRT fails silently.

**Suggestion:** Add a minimum scan count or free-cell count guard before the first planning step. The initial spin already provides some time for mapping, but an explicit check would be more robust.

**Impact:** Prevents early planning failures on slow-starting sensors.

**Files:** `igdm_basic.py`

---

## 6. Sensor Model & Discretization

### 6.1 Dynamic Thresholds Compress Low-Concentration Particles

**Problem:** When particle predictions span a wide range (some near source, some far), `create_discretization_thresholds` sets `d_max = max_conc + 3*sigma`. This stretches the bin edges to cover high concentrations, compressing most low-concentration particles into bin 0 (`-inf` to first threshold). The MI calculation loses discrimination power for moderate-concentration particles.

**Suggestion:** Use logarithmic bin spacing instead of linear `np.linspace`. This gives finer resolution at low concentrations (where most particles cluster in early exploration) and coarser resolution at high concentrations:
```python
thresholds = np.logspace(np.log10(0.1), np.log10(d_max), self.num_levels)[1:]
```

**Impact:** Better MI discrimination, especially during early exploration when most readings are near-background.

**Files:** `estimation/sensor_model.py`

### 6.2 `num_levels=10` May Be Too Coarse

**Problem:** With 10 measurement bins, the entropy computation has only 10 possible outcomes. The paper's comparison methods often use 20-50 levels. Coarser discretization underestimates mutual information, which can cause the dead end detector to trigger prematurely (BI* values are systematically lower than they should be).

**Suggestion:** Increase to `num_levels=15` or `20`. The computational cost scales linearly with `num_levels` in the vectorized `compute_expected_entropy`, so the overhead is minimal (~1.5x for 15 bins).

**Impact:** More accurate MI estimation, better dead-end detection calibration.

**Files:** `estimation/sensor_model.py`, ROS2 parameter `sensor_num_levels`

---

## 7. Navigation & Recovery

### 7.1 Initial Spin is 180 Degrees, Not 360

**Problem:** `perform_initial_spin` sends the robot to `current_theta + 3.14` (pi radians = 180 degrees), but the docstring and logs say "360 degree sensor sweep". Only half the surroundings are scanned.

**Suggestion:** Either perform two sequential 180-degree turns or use Nav2's rotate recovery behavior for a full 360-degree spin.

**Impact:** Complete initial map coverage, better starting conditions for the particle filter.

**Files:** `planning/navigator.py`

### 7.2 `time.sleep(1.0)` Blocks the ROS2 Executor

**Problem:** In `attempt_teleport_recovery`, `time.sleep(1.0)` blocks the entire ROS2 thread. No sensor, pose, or laser callbacks can be processed during this second. In a single-threaded executor, this freezes the entire node.

**Suggestion:** Replace with a ROS2 timer callback or state machine that waits for the next planning cycle without blocking.

**Impact:** Prevents 1-second communication blackout during recovery.

**Files:** `planning/navigator.py`

### 7.3 Teleport via `initialpose` Does Not Move the Simulated Robot

**Problem:** Publishing to `/PioneerP3DX/initialpose` only resets the Nav2 localization estimate, not the Gazebo model position. If the robot is physically stuck against a wall in simulation, the localization and physics diverge, causing further navigation failures.

**Suggestion:** For Gazebo simulation, use the `/gazebo/set_entity_state` service or `gazebo_ros` teleport interface to actually move the robot model. Keep the `initialpose` publish for real-robot scenarios where AMCL is used.

**Impact:** Recovery actually works in simulation.

**Files:** `planning/navigator.py`

---

## 8. Convergence & Termination

### 8.1 Convergence Ignores Release Rate Uncertainty

**Problem:** Convergence is declared when `max(std_x, std_y) < sigma_threshold`. The release rate standard deviation `std_Q` is completely ignored. The spatial estimate could converge while the release rate is wildly uncertain, indicating the filter has not actually localized the source reliably.

**Suggestion:** Add a secondary check: `std_Q / mean_Q < relative_Q_threshold` (e.g., 0.5). This ensures the filter has converged on both location and intensity.

**Impact:** Reduces false convergence declarations.

**Files:** `igdm_basic.py`

### 8.2 No Minimum Step Count Before Convergence

**Problem:** If particles collapse early (e.g., a strong first reading near the source), the filter can converge in 2-3 steps. This is likely a lucky coincidence, not true convergence. There is no minimum exploration requirement.

**Suggestion:** Add a `min_steps_before_convergence` parameter (e.g., 10). Only check convergence after this many steps.

**Impact:** Prevents premature termination from early particle collapse.

**Files:** `igdm_basic.py`

### 8.3 Source Near Walls Causes Artificial Convergence

**Problem:** When the source is near a wall, particles on the wall side are in occupied cells and get zero concentration predictions. They are quickly eliminated, squeezing the distribution against the wall boundary. This artificially lowers `std_x` or `std_y`, triggering convergence even when the estimate may be biased.

**Suggestion:** Monitor the particle distribution's skewness near boundaries. If the distribution is heavily truncated (many particles at a boundary), increase the convergence threshold dynamically or add a boundary-proximity penalty.

**Impact:** More reliable convergence in corner/wall-adjacent scenarios.

**Files:** `igdm_basic.py`, `estimation/particle_filter.py`

---

## 9. Data Logging

### 9.1 CSV Column Mismatch

**Problem:** The CSV header includes `bi_threshold` but the data row writes `dead_end_detected` (as a string) in that column instead. The `bi_threshold` value is never logged. All columns after this point are shifted by one.

**Suggestion:** Add the actual `bi_threshold` value to the data row in the correct position.

**Impact:** Fixes data corruption in log files; enables post-hoc analysis of dead end detection behavior.

**Files:** `utils/experiment_logger.py`

### 9.2 Elapsed Time is Always Zero

**Problem:** The `elapsed_time` column is hardcoded to 0. Actual timestamps are never written per step.

**Suggestion:** Pass `elapsed_time` from the node (computed from `self.start_time`) and log it per row.

**Impact:** Enables time-series analysis and correlation with rosbag data.

**Files:** `utils/experiment_logger.py`, `igdm_basic.py`

### 9.3 Summary Missing Parameters and Success Flag

**Problem:** The summary file does not record which parameter configuration was used or whether the search was successful (error < threshold).

**Suggestion:** Log all parameter values and a `success: True/False` field based on the estimation error and step count.

**Impact:** Reproducibility and batch experiment analysis.

**Files:** `utils/experiment_logger.py`

---

## 10. IGDM Distance Cache

### 10.1 FIFO Eviction Should Be LRU

**Problem:** The distance map cache uses FIFO eviction (`pop(next(iter(...)))`). The first-inserted entry is evicted regardless of how recently it was used. In practice, the RRT evaluates the same leaf positions across planning cycles, but FIFO may evict these frequently-reused entries.

**Suggestion:** Use `collections.OrderedDict` with `move_to_end()` on access for LRU behavior. This retains hot entries from the current planning area.

**Impact:** Higher cache hit rate, fewer redundant Dijkstra computations (~15ms savings per cache hit).

**Files:** `estimation/igdm_gas_model.py`

---

## Priority Summary

| # | Improvement | Impact | Effort | Priority |
|---|-----------|--------|--------|----------|
| 3.1 | Fix -inf BI disabling dead end detection | Critical bug fix | Low | **P0** |
| 9.1 | Fix CSV column mismatch | Data integrity | Low | **P0** |
| 7.1 | Fix initial spin (180 -> 360 degrees) | Correctness | Low | **P1** |
| 1.1 | Log-space weight arithmetic | Numerical stability | Medium | **P1** |
| 1.4 | Don't initialize particles in unknown cells | Estimation quality | Low | **P1** |
| 5.1 | Numba/vectorize LiDAR ray tracing | Performance | Medium | **P1** |
| 3.2 | Consecutive confirmation for dead end | Robustness | Low | **P1** |
| 4.1 | Fallback to nearest frontier on zero MI | Robustness | Low | **P1** |
| 2.1 | Incremental node positions in RRT sprawl | Performance | Low | **P2** |
| 4.2 | Vectorize PRM collision checks | Performance | Medium | **P2** |
| 1.2 | Fix MCMC to use full posterior | Estimation quality | Medium | **P2** |
| 1.3 | Adaptive MCMC proposal std | Estimation quality | Low | **P2** |
| 8.1 | Check Q convergence | Convergence quality | Low | **P2** |
| 8.2 | Minimum step count before convergence | Robustness | Low | **P2** |
| 6.1 | Logarithmic discretization bins | MI quality | Low | **P2** |
| 10.1 | LRU cache eviction for distance maps | Performance | Low | **P2** |
| 7.2 | Replace time.sleep with ROS2 timer | ROS2 best practice | Medium | **P2** |
| 2.2 | Goal-biased RRT sampling | Planning quality | Medium | **P3** |
| 4.3 | Sample PRM from free cell list | Performance | Medium | **P3** |
| 5.2 | Consolidate validity check semantics | Code quality | Medium | **P3** |
| 9.2 | Log elapsed time per step | Logging | Low | **P3** |
| 9.3 | Log parameters in summary | Logging | Low | **P3** |
| 1.5 | Return current entropy on weight collapse | MI quality | Low | **P3** |
| 7.3 | Gazebo teleport for recovery | Simulation | Medium | **P3** |
| 8.3 | Boundary-aware convergence | Robustness | High | **P3** |
| 6.2 | Increase num_levels to 15-20 | MI quality | Low | **P3** |

---

*Report generated: February 2026*
*Based on: Kim et al. "Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search", IEEE RA-L, Vol. 10, No. 1, 2025*
