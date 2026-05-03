# reinforcement_learning

Shared training-side code for the PPO gas-source-localization agents:

- `config.py` — hyperparameters mirrored from the training trees (`rl_old/`,
  `rl_new/`, `rl_osl/`). Keep in sync with whichever training tree you use.
- `envs/sensor_model.py` — `BinarySensorModel` and other small env pieces the
  deployment also reuses.
- `models/actor_critic*.py` — actor-critic networks. The deployment loads
  `.pt` checkpoints into one of these classes.

GADEN-side ROS deployment of trained checkpoints lives in the sibling
**`gaden_transfer/`** package — see its README for the lidar / image_5ch /
image_6ch variants, motion modes, and run/batch instructions.

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select reinforcement_learning --symlink-install
source install/setup.bash
```

Dependencies (apart from ROS 2 Humble): `torch`, `numpy`.
