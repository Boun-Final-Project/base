"""
Generate 80 fixed test environments for NavigationEnv evaluation.

Run once to produce nav_test_envs.json. The same seeds are always produced
from MASTER_SEED, ensuring reproducible maps across machines.

Distribution:
  20 × small       (area < 120 m²)        interpolation — small training rooms
  20 × medium      (120 ≤ area < 190 m²)  interpolation — mid training rooms
  20 × large       (area ≥ 190 m²)        interpolation — large training rooms
  20 × extra-large (width 22–25 m,        extrapolation — beyond training range
                    height 16–20 m)

Usage:
    python3 generate_nav_test_envs.py
"""

import json
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from reinforcement_learning.envs.nav_env import NavigationEnv

OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "nav_test_envs.json")
MASTER_SEED = 2 ** 17 - 1   # 131071
ENVS_PER_CATEGORY = 20

# Interpolation categories — thresholds based on actual MapGenerator output
# (training: width 8–20 m, height 6–15 m): p25 ≈ 115 m², p75 ≈ 190 m²
_INTERP_CATEGORIES = ("small", "medium", "large")
_AREA_SMALL  = 120.0   # area < 120
_AREA_MEDIUM = 190.0   # 120 ≤ area < 190

# Extrapolation category — rooms outside the training range
_EXTRAP_WIDTH_RANGE  = (22.0, 25.0)
_EXTRAP_HEIGHT_RANGE = (16.0, 20.0)


def _size_category(w, h):
    area = w * h
    if area < _AREA_SMALL:
        return "small"
    if area < _AREA_MEDIUM:
        return "medium"
    return "large"


def main():
    rng = np.random.default_rng(MASTER_SEED)

    # --- interpolation envs (standard training range) ---
    buckets = {cat: [] for cat in _INTERP_CATEGORIES}
    env = NavigationEnv()
    env_id = 0

    while any(len(buckets[c]) < ENVS_PER_CATEGORY for c in _INTERP_CATEGORIES):
        seed = int(rng.integers(0, 1_000_000))
        env.reset(seed=seed)
        cat = _size_category(env._room_width, env._room_height)
        if len(buckets[cat]) >= ENVS_PER_CATEGORY:
            continue
        buckets[cat].append({
            "env_id":      env_id,
            "seed":        seed,
            "room_width":  round(float(env._room_width),  3),
            "room_height": round(float(env._room_height), 3),
            "start_pos":   [round(float(env.robot_pos[0]), 3),
                            round(float(env.robot_pos[1]), 3)],
            "goal_pos":    [round(float(env.goal_pos[0]),  3),
                            round(float(env.goal_pos[1]),  3)],
            "size_cat":    cat,
        })
        env_id += 1

    # --- extrapolation envs (beyond training range) ---
    xl_env = NavigationEnv(width_range=_EXTRAP_WIDTH_RANGE,
                           height_range=_EXTRAP_HEIGHT_RANGE)
    xl_bucket = []
    while len(xl_bucket) < ENVS_PER_CATEGORY:
        seed = int(rng.integers(0, 1_000_000))
        xl_env.reset(seed=seed)
        xl_bucket.append({
            "env_id":      env_id,
            "seed":        seed,
            "room_width":  round(float(xl_env._room_width),  3),
            "room_height": round(float(xl_env._room_height), 3),
            "start_pos":   [round(float(xl_env.robot_pos[0]), 3),
                            round(float(xl_env.robot_pos[1]), 3)],
            "goal_pos":    [round(float(xl_env.goal_pos[0]),  3),
                            round(float(xl_env.goal_pos[1]),  3)],
            "size_cat":    "extra-large",
        })
        env_id += 1

    all_envs = []
    for cat in _INTERP_CATEGORIES:
        all_envs.extend(buckets[cat])
    all_envs.extend(xl_bucket)
    for idx, rec in enumerate(all_envs):
        rec["env_id"] = idx

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "master_seed":       MASTER_SEED,
            "total":             len(all_envs),
            "size_distribution": {
                **{c: len(buckets[c]) for c in _INTERP_CATEGORIES},
                "extra-large": len(xl_bucket),
            },
            "envs":              all_envs,
        }, f, indent=2)

    print(f"  small       ( area < 120 m²):          {len(buckets['small'])} environments  [interpolation]")
    print(f"  medium      (120 ≤ area < 190 m²):     {len(buckets['medium'])} environments  [interpolation]")
    print(f"  large       ( area ≥ 190 m²):          {len(buckets['large'])} environments  [interpolation]")
    print(f"  extra-large (22–25 m × 16–20 m):       {len(xl_bucket)} environments  [extrapolation]")
    print()
    print(f"Saved {len(all_envs)} test environments → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
