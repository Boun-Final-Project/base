"""
Generate 100 fixed test environments for agent evaluation.

Run once to produce test_envs.json. The same (template_id, seed) pairs
are always produced from MASTER_SEED, ensuring reproducible maps across
machines.

Distribution: 10×template0, 10×template1, 15×template2,
              15×template3, 25×template4, 25×template5

Usage:
    python3 generate_test_envs.py
"""

import json
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from rl_5_channel.envs.gas_source_env import GasSourceEnv

OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "test_envs.json")

TEMPLATE_COUNTS = [(0, 10), (1, 10), (2, 15), (3, 15), (4, 25), (5, 25)]
TEMPLATE_NAMES  = ["empty", "single_wall", "u_shape", "three_walls", "complex_maze", "multi_room"]
MASTER_SEED = 2 ** 17 - 1


def main():
    rng = np.random.default_rng(MASTER_SEED)
    envs = []
    env_id = 0

    for template_id, count in TEMPLATE_COUNTS:
        seeds = rng.integers(0, 100_000, size=count)
        for seed in seeds:
            env = GasSourceEnv(template_id=template_id)
            obs, _ = env.reset(seed=int(seed))
            envs.append({
                "env_id":      env_id,
                "template_id": int(template_id),
                "seed":        int(seed),
                "source_pos":  [round(float(env._source_pos[0]), 3),
                                round(float(env._source_pos[1]), 3)],
                "robot_pos":   [round(float(env._robot_pos[0]),  3),
                                round(float(env._robot_pos[1]),  3)],
            })
            env_id += 1

        print(f"  Template {template_id} ({TEMPLATE_NAMES[template_id]:>14}): "
              f"{count} environments")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"master_seed": MASTER_SEED, "total": len(envs), "envs": envs}, f, indent=2)

    print(f"\nSaved {len(envs)} test environments → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
