"""gpulaika — the flagship baseline: a PPO policy (dual backbone, heading-frame
lidar, local point-wind) trained on GADEN and deployed under Nav2 drive.

This is an ADAPTER, not a reimplementation: the observation builder, the checkpoint
loader and the action->target conversion are imported from gaden_transfer, so there
is exactly one source of truth for the 107-dim observation. Reimplementing any of it
here is how train/deploy drift gets reintroduced.

Its validated recipe (run_escape_7x5.sh): Nav2 drive, escape ON, drive_timeout 30,
nav_goal_tolerance 0.10, max_steps 600.
"""
import os
import sys

import numpy as np
import torch

_SRC_BASE = '/home/efe/ros2_ws/src/base'
if _SRC_BASE not in sys.path:
    sys.path.insert(0, _SRC_BASE)

from gsl_bench.agent import GSLAgent, Observation, ScenarioInfo, Waypoint


class GpulaikaAgent(GSLAgent):

    def __init__(self, config=None):
        cfg = config or {}
        self.checkpoint = cfg.get('checkpoint', '')
        self.arch = cfg.get('arch', 'dual')
        self.device = torch.device(cfg.get('device', 'cpu'))
        self.wind_csv = cfg.get('wind_file', '')
        self.lidar_frame = cfg.get('lidar_frame', 'heading')  # gpulaika is heading-frame
        # The observation builder reads this at construction time, and the policy was
        # trained on live local wind, not the episode-mean polar wind.
        os.environ['OSL_LOCAL_WIND_OBS'] = str(cfg.get('local_wind_obs', 1))
        if not self.checkpoint or not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f'gpulaika checkpoint not found: {self.checkpoint!r}')
        self._builder = None
        self._latest = None
        self._step = 0

    def initialize(self) -> None:
        from gaden_transfer.gaden_transfer_lidar.gaden_rl_node import (
            _load_agent, _load_run_config,
        )
        run_cfg = _load_run_config(self.checkpoint)
        self.model = _load_agent(self.checkpoint, self.arch, self.device, run_cfg)
        n = sum(p.numel() for p in self.model.parameters())
        print(f'[gpulaika] loaded {os.path.basename(self.checkpoint)} '
              f'({n:,} parameters, arch={self.arch})', flush=True)

    def reset(self, scenario: ScenarioInfo) -> None:
        from gaden_transfer.gaden_transfer_lidar.obs_builder import ObservationBuilder
        self._builder = ObservationBuilder(
            scenario.map.width_m, scenario.map.height_m,
            lidar_frame=self.lidar_frame,
            drive_mode=True,          # the harness always drives physically
        )
        if not self.wind_csv:
            raise ValueError(
                'gpulaika needs wind_file (the scenario wind CSV) — the observation '
                'builder returns None until the wind is locked. episode_runner injects '
                'it automatically; pass it in the agent config when running by hand.')
        self._builder.load_wind_from_file(self.wind_csv)
        self._step = 0
        # Training normalizes position/gas-history in a corner-origin WORLD frame
        # (world_to_grid = floor(x/res), no offset → robot_pos in [0, map_size]),
        # verified against the training env's _build_observation. Ground-truth
        # deploy already matches this. Under SLAM/TF, obs.x/y are in the SLAM map
        # frame (anchored near start, goes negative) — the position feature clips
        # to ~0 and the gas-history deltas scale wrong. Capture the SLAM→world
        # offset on the first observe() from the KNOWN world start pose. In
        # ground-truth mode obs≈start at step 0, so the offset is ~0 (a no-op).
        self._true_start_x = scenario.start_x
        self._true_start_y = scenario.start_y
        self._world_offset = None

    def observe(self, obs: Observation) -> None:
        b = self._builder
        if self._world_offset is None:
            self._world_offset = (self._true_start_x - obs.x,
                                  self._true_start_y - obs.y)
        b.robot_x = obs.x + self._world_offset[0]
        b.robot_y = obs.y + self._world_offset[1]
        b.robot_theta = obs.theta
        b.update_gas(obs.gas_ppm)
        b.update_live_wind(obs.wind_speed, obs.wind_direction)
        if obs.lidar_msg is not None:
            b.update_lidar(obs.lidar_msg)
        else:
            # Off-ROS fallback: the builder wants normalised sensor-frame rays.
            b.lidar_norm = np.clip(
                obs.lidar / max(obs.lidar_max_range, 1e-6), 0.0, 1.0).astype(np.float32)
        # Training appends one gas-history entry per env.step(); step 0 is already
        # seeded by the first gas callback, so recording it again would double it.
        if self._step > 0:
            b.record_step()
        self._latest = obs

    def act(self) -> Waypoint:
        from reinforcement_learning import config as cfg
        from gaden_transfer.gaden_transfer_lidar.gaden_rl_node import _action_to_target

        vec = self._builder.build()
        if vec is None:
            # Sensors not all in yet — hold position; the harness will step again.
            return Waypoint(self._latest.x, self._latest.y)
        vec = np.clip(vec, 0.0, 1.0)
        # Verify the position feature (dims 102,103) is a sane in-frame [0,1]
        # value, not clipped to a boundary — printed to node.log at a low rate.
        if self._step % 100 == 0:
            print(f'[gpulaika] step {self._step}: world_pos=('
                  f'{self._builder.robot_x:.2f},{self._builder.robot_y:.2f}) '
                  f'pos_feat=({vec[102]:.3f},{vec[103]:.3f})', flush=True)

        t = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.arch == 'dual':
                enc = self.model._encode_shared(t)
                action = self.model._actor_dist(enc).mean      # deterministic
            else:
                action, _, _, _ = self.model.get_action_and_value(t)
        a = action.cpu().numpy().flatten()

        tx, ty, theta = _action_to_target(
            self._latest.x, self._latest.y, a, self.arch, cfg.STEP_SIZE)

        # Load-bearing for heading-frame checkpoints: ray 0 of the NEXT observation
        # must align with the direction just commanded.
        self._builder.last_action_theta = float(theta)
        self._step += 1
        return Waypoint(tx, ty, theta=theta)

    def name(self) -> str:
        return 'gpulaika'
