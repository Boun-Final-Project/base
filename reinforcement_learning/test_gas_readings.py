"""
Diagnostic: gas concentration readings across all 6 map templates.

For each template, walks the robot toward the source and logs:
- distance to source
- true concentration
- binary sensor reading
- sigma_m at that timestep

Also samples random positions to check concentration falloff makes sense.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.gas_source_env import GasSourceEnv

TEMPLATE_NAMES = [
    "empty", "single_wall", "u_shape", "three_walls", "complex_maze", "multi_room"
]


def test_walk_toward_source(template_id, seed=42):
    """Walk straight toward the source, logging gas readings at each step."""
    env = GasSourceEnv(template_id=template_id)
    obs, info = env.reset(seed=seed)

    source = env._source_pos.copy()
    robot = env._robot_pos.copy()
    init_dist = np.linalg.norm(source - robot)

    print(f"\n{'='*70}")
    print(f"Template {template_id}: {TEMPLATE_NAMES[template_id]}")
    print(f"  Map size: {env._map_width:.1f} x {env._map_height:.1f} m")
    print(f"  Source: ({source[0]:.1f}, {source[1]:.1f})")
    print(f"  Robot:  ({robot[0]:.1f}, {robot[1]:.1f})")
    print(f"  Initial distance: {init_dist:.2f} m")
    print(f"  Wind offset: ({env._wind_offset[0]:.2f}, {env._wind_offset[1]:.2f})")
    print(f"  sigma_m(t=0): {cfg.SIGMA_M_BASE:.2f}")
    print(f"{'='*70}")
    print(f"  {'Step':>5}  {'Dist':>7}  {'Conc':>10}  {'Binary':>6}  {'sigma_m':>8}  {'Threshold':>10}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*10}")

    detections = 0
    zero_conc_near_source = False

    for step in range(min(cfg.MAX_STEPS, 300)):
        # Compute angle toward source
        direction = source - env._robot_pos
        dist = np.linalg.norm(direction)
        angle = np.arctan2(direction[1], direction[0])
        action = np.array([(angle / (2 * np.pi)) % 1.0], dtype=np.float32)

        # Get concentration before stepping (at current position)
        conc = env._get_concentration()
        sigma_m = env._igdm.get_sigma_m(env._current_step)
        threshold = env._sensor.threshold if env._sensor.threshold is not None else 0.0

        obs, reward, terminated, truncated, info = env.step(action)

        # Binary reading from the last step
        binary = int(env._gas_history[-1][2])
        if binary == 1:
            detections += 1

        if dist < 2.0 and conc == 0.0:
            zero_conc_near_source = True

        # Print every 10 steps or when close, or on detection
        if step % 20 == 0 or dist < 3.0 or binary == 1:
            print(f"  {step:>5}  {dist:>7.2f}  {conc:>10.6f}  {binary:>6}  {sigma_m:>8.3f}  {threshold:>10.6f}")

        if terminated:
            print(f"  >>> SOURCE FOUND at step {step+1}, dist={info['distance_to_source']:.3f}")
            break
        if truncated:
            final_dist = np.linalg.norm(source - env._robot_pos)
            print(f"  >>> TIMEOUT at step {step+1}, final dist={final_dist:.2f}")
            break

        # If robot is stuck (collision), try a slightly different angle
        if info.get("collision", False):
            # Perturb angle
            angle += np.pi / 6
            action = np.array([(angle / (2 * np.pi)) % 1.0], dtype=np.float32)

    print(f"\n  Summary: {detections} detections out of {step+1} steps")

    # Flag potential issues
    if zero_conc_near_source:
        print(f"  *** WARNING: Zero concentration observed near source (<2m) — "
              f"possible obstacle blocking Dijkstra path")
    if detections == 0 and init_dist < 10.0:
        print(f"  *** WARNING: No detections despite starting <10m from source")

    return {
        "template": template_id,
        "init_dist": init_dist,
        "detections": detections,
        "found": terminated,
        "zero_near_source": zero_conc_near_source,
    }


def test_concentration_grid(template_id, seed=42):
    """Sample concentration at a grid of positions to check spatial pattern."""
    env = GasSourceEnv(template_id=template_id)
    env.reset(seed=seed)

    source = env._source_pos.copy()
    print(f"\n  Concentration samples around source ({source[0]:.1f}, {source[1]:.1f}):")
    print(f"  {'Dist':>7}  {'Conc':>10}  {'Expected trend'}")

    distances = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    prev_conc = float('inf')
    monotonic = True

    for d in distances:
        # Sample in 4 directions, take the max (some may be blocked by walls)
        concs = []
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            test_pos = source + d * np.array([np.cos(angle), np.sin(angle)])
            # Clamp to map
            test_pos[0] = np.clip(test_pos[0], 0, env._map_width)
            test_pos[1] = np.clip(test_pos[1], 0, env._map_height)

            # Compute concentration at this position
            r, c = env._igdm._world_to_coarse_idx(test_pos[0], test_pos[1])
            dijk_d = env._dijkstra_from_source[r, c]
            if not np.isinf(dijk_d):
                dijk_d = max(dijk_d, 0.1)
                sigma_m = env._igdm.get_sigma_m(0)
                conc = cfg.SOURCE_RELEASE_RATE * np.exp(-(dijk_d ** 2) / (2 * sigma_m ** 2))
                concs.append(conc)

        if concs:
            max_conc = max(concs)
            avg_conc = np.mean(concs)
            trend = "OK" if max_conc <= prev_conc else "*** NON-MONOTONIC"
            if max_conc > prev_conc and d > 0.5:
                monotonic = False
            print(f"  {d:>7.1f}  {max_conc:>10.6f}  (avg={avg_conc:.6f})  {trend}")
            prev_conc = max_conc
        else:
            print(f"  {d:>7.1f}  {'blocked':>10}  (all directions blocked)")

    if not monotonic:
        print(f"  *** WARNING: Non-monotonic concentration — Dijkstra detour may cause this in walled maps")


if __name__ == "__main__":
    print("Gas Reading Diagnostics")
    print(f"Config: SIGMA_M_BASE={cfg.SIGMA_M_BASE}, DISPERSION_RATE={cfg.DISPERSION_RATE}")
    print(f"        SOURCE_RELEASE_RATE={cfg.SOURCE_RELEASE_RATE}")
    print(f"        SENSOR: alpha={cfg.SENSOR_ALPHA}, sigma_env={cfg.SENSOR_SIGMA_ENV}, "
          f"threshold_weight={cfg.SENSOR_THRESHOLD_WEIGHT}")

    results = []
    for tid in range(6):
        r = test_walk_toward_source(tid, seed=42)
        test_concentration_grid(tid, seed=42)
        results.append(r)

    print(f"\n{'='*70}")
    print("Overall Summary")
    print(f"{'='*70}")
    for r in results:
        status = "FOUND" if r["found"] else "TIMEOUT"
        warn = " *** ZERO NEAR SOURCE" if r["zero_near_source"] else ""
        print(f"  Template {r['template']} ({TEMPLATE_NAMES[r['template']]:>14}): "
              f"init_dist={r['init_dist']:>6.2f}m, detections={r['detections']:>3}, "
              f"{status}{warn}")
