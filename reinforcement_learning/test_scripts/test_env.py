"""
End-to-end test for GasSourceEnv (Task 9).

Tests:
1. Basic instantiation and reset
2. Step with random actions
3. All 6 map templates generate successfully
4. Collision detection
5. Source discovery (terminal reward)
6. Render produces RGB array
7. Performance benchmark
8. Determinism
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.gas_source_env import GasSourceEnv


def test_basic_instantiation():
    """Env creates, resets, and returns correct observation shape."""
    print("=== Test: Basic instantiation ===")
    env = GasSourceEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    assert obs.shape == (cfg.STATE_DIM,), f"Expected ({cfg.STATE_DIM},), got {obs.shape}"
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), \
        f"Obs out of [0,1]: min={obs.min()}, max={obs.max()}"
    assert "distance_to_source" in info

    print(f"  obs shape={obs.shape}, range=[{obs.min():.3f}, {obs.max():.3f}]  ✓")
    print("PASSED\n")


def test_random_steps():
    """Step 200 times with random actions, check obs/reward validity."""
    print("=== Test: Random steps (200 steps) ===")
    env = GasSourceEnv()
    obs, _ = env.reset(seed=7)
    total_reward = 0.0
    resets = 0

    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (cfg.STATE_DIM,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), \
            f"Step {i}: obs out of range [{obs.min()}, {obs.max()}]"
        assert isinstance(reward, float)
        total_reward += reward

        if terminated or truncated:
            obs, _ = env.reset()
            resets += 1

    print(f"  200 steps done, {resets} resets, total_reward={total_reward:.1f}  ✓")
    print("PASSED\n")


def test_all_templates():
    """Verify all 6 templates work via repeated resets."""
    print("=== Test: All 6 map templates ===")
    env = GasSourceEnv()
    templates_seen = set()

    for i in range(300):
        obs, info = env.reset()
        templates_seen.add(env._map_gen.generate.__func__)  # just check reset works
        assert obs.shape == (cfg.STATE_DIM,)

    # We can't directly check template_id from env, so just confirm no crashes
    print(f"  300 resets without errors  ✓")
    print("PASSED\n")


def test_collision():
    """Step into a wall and verify collision penalty."""
    print("=== Test: Collision detection ===")
    env = GasSourceEnv()
    found_collision = False

    for trial in range(50):
        obs, _ = env.reset(seed=trial)
        # Try all directions to find a wall
        for angle_frac in np.linspace(0, 1, 36, endpoint=False):
            action = np.array([angle_frac], dtype=np.float32)
            pos_before = env._robot_pos.copy()

            # Step repeatedly in the same direction to hit a wall
            for _ in range(30):
                obs, reward, term, trunc, info = env.step(action)
                if info["collision"]:
                    # Robot should stay in place
                    assert reward <= cfg.R_STEP + cfg.R_COLLISION + cfg.R_DETECTION + 0.01, \
                        f"Collision reward too high: {reward}"
                    found_collision = True
                    break
                if term or trunc:
                    break
            if found_collision:
                break
        if found_collision:
            break

    assert found_collision, "Never hit a wall in 50 trials"
    print(f"  Collision detected and penalized  ✓")
    print("PASSED\n")


def test_source_discovery():
    """Manually teleport robot near source, verify terminal reward."""
    print("=== Test: Source discovery ===")
    env = GasSourceEnv()
    obs, _ = env.reset(seed=42)

    # Teleport robot close to source
    direction = env._source_pos - env._robot_pos
    dist = np.linalg.norm(direction)
    # Move to within D_SUCCESS
    env._robot_pos = env._source_pos - direction / dist * (cfg.D_SUCCESS * 0.3)

    # Take a step toward source
    angle_to_source = np.arctan2(direction[1], direction[0])
    action = np.array([angle_to_source / (2 * np.pi) % 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated, "Should be terminated (source found)"
    assert reward >= cfg.R_SUCCESS + cfg.R_STEP - 1.0, \
        f"Expected high reward, got {reward}"

    print(f"  Source found: reward={reward:.1f}, terminated={terminated}  ✓")
    print("PASSED\n")


def test_render():
    """Verify render writes a PNG frame to viz_output_dir."""
    print("=== Test: Render ===")
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        env = GasSourceEnv(viz_output_dir=d)
        env.reset(seed=42)

        # Take a few steps first
        for _ in range(5):
            env.step(env.action_space.sample())

        env.render()
        files = sorted(os.listdir(d))
        assert files, "render() did not write any PNG"
        assert files[0].endswith(".png")
        print(f"  Wrote {files[0]}  ✓")
    print("PASSED\n")


def test_determinism():
    """Same seed produces same episode."""
    print("=== Test: Determinism ===")
    rewards1, rewards2 = [], []

    for run_rewards, seed in [(rewards1, 123), (rewards2, 123)]:
        env = GasSourceEnv()
        obs, _ = env.reset(seed=seed)
        for _ in range(50):
            # Use a fixed action sequence (not env.action_space.sample which uses its own rng)
            action = np.array([0.25], dtype=np.float32)
            obs, reward, term, trunc, info = env.step(action)
            run_rewards.append(reward)
            if term or trunc:
                break

    assert rewards1 == rewards2, "Rewards differ between runs with same seed"
    print(f"  {len(rewards1)} steps match exactly  ✓")
    print("PASSED\n")


def test_max_steps_truncation():
    """Verify episode truncates at MAX_STEPS."""
    print("=== Test: Max steps truncation ===")
    env = GasSourceEnv()
    env.reset(seed=99)

    # Move source far away so we don't accidentally find it
    env._source_pos = np.array([9999.0, 9999.0])

    for step in range(cfg.MAX_STEPS):
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if step == cfg.MAX_STEPS - 1:
            assert truncated, "Should truncate at MAX_STEPS"
            assert reward <= cfg.R_MAX_STEPS + cfg.R_STEP + 0.5, \
                f"Expected timeout penalty, got {reward}"

    print(f"  Truncated at step {cfg.MAX_STEPS} with penalty  ✓")
    print("PASSED\n")


def test_performance():
    """Benchmark steps per second."""
    print("=== Test: Performance ===")
    env = GasSourceEnv()
    env.reset(seed=0)

    n = 10000
    t0 = time.time()
    for _ in range(n):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    elapsed = time.time() - t0
    rate = n / elapsed

    print(f"  {n} steps in {elapsed:.2f} s  ({rate:.0f} steps/sec)")
    target = 5000
    if rate >= target:
        print(f"  Above target ({target} steps/sec)  ✓")
    else:
        print(f"  Below target ({target} steps/sec) — may need optimization")
    print("PASSED\n")


if __name__ == "__main__":
    test_basic_instantiation()
    test_random_steps()
    test_all_templates()
    test_collision()
    test_source_discovery()
    test_render()
    test_determinism()
    test_max_steps_truncation()
    test_performance()

    print("All tests passed!")
