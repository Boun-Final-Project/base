"""
Visual test for GasSourceEnv.
Runs one episode with random actions and saves key frames + a final trajectory image.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement_learning.envs.gas_source_env import GasSourceEnv

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_test")
os.makedirs(out_dir, exist_ok=True)

env = GasSourceEnv(render_mode="rgb_array")
obs, info = env.reset()

print(f"Map: {env._map_width:.1f} x {env._map_height:.1f} m")
print(f"Source: ({env._source_pos[0]:.1f}, {env._source_pos[1]:.1f})")
print(f"Robot:  ({env._robot_pos[0]:.1f}, {env._robot_pos[1]:.1f})")
print(f"Wind:   speed={env._wind.speed:.2f} m/s, dir={np.degrees(env._wind.direction):.0f}°")
print()

# Save frames at these steps
save_at = {0, 10, 30, 60, 100, 150, 200, 299}
total_reward = 0.0
gas_detections = 0

# Save initial frame
img = env.render()
plt.imsave(os.path.join(out_dir, "step_000.png"), img)

for step in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if info["gas_reading"] == 1:
        gas_detections += 1

    if step + 1 in save_at:
        img = env.render()
        plt.imsave(os.path.join(out_dir, f"step_{step+1:03d}.png"), img)
        print(f"  Step {step+1:3d}: dist={info['distance_to_source']:.1f}m, "
              f"reward={reward:+.1f}, gas={info['gas_reading']}, "
              f"collision={info['collision']}")

    if terminated:
        print(f"\n  SOURCE FOUND at step {step+1}!")
        img = env.render()
        plt.imsave(os.path.join(out_dir, f"step_{step+1:03d}_found.png"), img)
        break
    if truncated:
        print(f"\n  Timeout at step {step+1}")
        img = env.render()
        plt.imsave(os.path.join(out_dir, f"step_{step+1:03d}_timeout.png"), img)
        break

print(f"\nTotal reward: {total_reward:.1f}")
print(f"Gas detections: {gas_detections}/{step+1} steps")
print(f"Final distance to source: {info['distance_to_source']:.1f} m")
print(f"\nFrames saved to: {out_dir}/")
