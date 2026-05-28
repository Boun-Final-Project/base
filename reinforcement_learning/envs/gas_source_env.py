"""
Gymnasium environment for RL gas source localization.

Wraps map generation, IGDM gas dispersion, LiDAR, wind, and binary sensor
into a standard reset() / step() / render() interface.
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from collections import deque

from .. import config as cfg
from .map_generator import MapGenerator
from .occupancy_grid import OccupancyGrid
from .igdm_model import IGDMModel
from .filament_plume import FilamentPlume
from .lidar_sim import LidarSim
from .wind_model import WindModel, make_training_wind_field
from .sensor_model import BinarySensorModel
from .visualizer import StepVisualizer


class GasSourceEnv(gymnasium.Env):
    """Gas source localization environment.

    Observation (107-dim):
        [gas_history (30)] + [lidar (24)] + [pos_x, pos_y] + [wind_local_ux, wind_local_uy] + [time]
        Gas history: 10 timesteps x 3 features (rel_x, rel_y, binary).
        Positions are relative to current robot position, normalized to [0, 1].
        All values normalized to [0, 1].

    Action:
        1-dim scalar in [0, 1] (Beta) → angle = action * 2π, OR
        2-dim (cos θ, sin θ) (Gaussian) → angle = atan2(sin, cos).
        Robot moves STEP_SIZE meters in that direction.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, seed=None, template_id=None,
                 viz_output_dir=None):
        super().__init__()
        self.render_mode = render_mode
        self._template_id = template_id  # None = random, 0-9 = fixed template
        self._max_template_id = None     # None = all templates, set by curriculum
        self._template_weights = None    # None = uniform; else list/array len >= max_id+1
        self._viz_output_dir = viz_output_dir
        self._visualizer = None

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(cfg.STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._map_gen = MapGenerator(rng=self._rng)
        self._wind = WindModel(
            speed_range=cfg.WIND_SPEED_RANGE,
            max_speed=cfg.WIND_MAX_SPEED,
        )

        # Populated on reset
        self._grid = None
        self._igdm = None
        self._plume = None
        self._lidar = None
        self._sensor = None
        self._source_pos = None
        self._robot_pos = None
        self._robot_heading = None
        self._map_width = 0.0
        self._map_height = 0.0
        self._current_step = 0
        self._gas_history = None
        self._deploy_motion = False  # set per-episode in reset()
        self._sensor_noise = True     # set per-episode in reset(); see reset()
        self._trajectory = []
        self._wind_offset = None
        self._dijkstra_from_source = None

    def set_room_size_range(self, width_range, height_range):
        """Update room size range (for curriculum learning)."""
        self._map_gen.width_range = width_range
        self._map_gen.height_range = height_range

    def set_max_template(self, max_template_id, weights=None):
        """Set maximum template index for curriculum (0-9). None = use all.

        weights : sequence of float, optional
            Per-template sampling weights. Only the first (max_template_id+1)
            entries are used. None = uniform sampling.
        """
        self._max_template_id = max_template_id
        self._template_weights = weights

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._map_gen = MapGenerator(rng=self._rng)

        # Map: either injected by caller (GADEN eval) or generated procedurally.
        if options is not None and "map_data" in options:
            map_data   = options["map_data"]
            wind_field = options.get("wind_field")        # may be None
            gas_field  = options.get("gas_field")         # may be None (ReplayGasSource)
            self._last_template_id = None  # GADEN eval: template ID not applicable
            # Deployment walk-back collision motion is opt-in via
            # options["deploy_motion"] (the GADEN eval harness sets it True).
            # Default False so injected-map TRAINING paths (e.g. the CFD wind
            # library, which injects map_data+wind_field every reset) keep the
            # all-or-nothing training motion the policy is trained against.
            self._deploy_motion = bool(options.get("deploy_motion", False))
            # Sensor noise: default True (training behavior). The GADEN eval
            # harness passes sensor_noise=False to match the real ROS2 deploy
            # pipeline, which feeds the raw (noiseless) PID concentration — so
            # the Python eval predicts real deployment rather than injecting a
            # sigma_env floor the deployed sensor doesn't have.
            self._sensor_noise = bool(options.get("sensor_noise", True))
        else:
            tid = self._template_id
            if tid is None and self._max_template_id is not None:
                if self._template_weights is not None:
                    w = np.asarray(
                        self._template_weights[: self._max_template_id + 1],
                        dtype=float,
                    )
                    total = w.sum()
                    if total > 0:
                        w = w / total
                    else:
                        w = np.ones_like(w) / len(w)
                    tid = int(self._rng.choice(self._max_template_id + 1, p=w))
                else:
                    tid = int(self._rng.integers(0, self._max_template_id + 1))
            self._last_template_id = tid  # exposed for tests and logging
            map_data   = self._map_gen.generate(template_id=tid)
            wind_field = None
            gas_field  = None
            self._deploy_motion = False  # training keeps all-or-nothing motion
            self._sensor_noise = True     # training keeps the sensor noise floor

        self._grid = map_data["grid"]
        self._source_pos = np.array(map_data["source_pos"], dtype=np.float64)
        self._robot_pos = np.array(map_data["robot_pos"], dtype=np.float64)
        self._robot_heading = 0.0
        self._map_width = map_data["width"]
        self._map_height = map_data["height"]

        # Wind: spatial mean of the field for the policy ctx vector when a
        # wind_field is provided; otherwise random per-episode uniform wind.
        # If map_data has a "wind_bias" hint (radians, source->robot direction
        # used by wall-trap-like templates), sample direction within ±30° of
        # that hint so the plume saturates the robot's area.
        if wind_field is not None:
            speed, direction = wind_field.spatial_mean()
            if cfg.LOCAL_WIND_OBS and getattr(wind_field, "field", None) is not None:
                # Policy observes LOCAL wind at the robot (point anemometer),
                # matching real GADEN deployment. Wrap the spatial field in a
                # WindModel so get_local_wind(robot_pos) queries it. set_uniform
                # still sets (speed, direction) for the dispersion-offset calc
                # and as fallback, but does NOT clear _field.
                self._wind = WindModel(
                    field=np.asarray(wind_field.field, dtype=np.float64),
                    resolution=wind_field.resolution,
                    occupancy=~wind_field._free_mask,
                    max_speed=cfg.WIND_MAX_SPEED,
                )
            self._wind.set_uniform(speed, direction)
        elif map_data.get("wind_bias") is not None:
            self._wind.randomize_biased(self._rng, float(map_data["wind_bias"]))
        else:
            self._wind = make_training_wind_field(
                self._grid, self._rng, cfg.WIND_SPEED_RANGE, cfg.WIND_MAX_SPEED
            )

        # Gas model selection
        if gas_field is not None:
            # GADEN eval: replay the REAL stored concentration field (drop-in
            # gas source) instead of the surrogate plume. Removes the largest
            # eval fidelity gap — the surrogate's gas field differs in structure
            # from real GADEN's (esp. near-homogeneous in tight curved mazes).
            self._plume = gas_field
            self._igdm = None
            self._wind_offset = None
            self._dijkstra_from_source = None
        elif cfg.GAS_MODEL == "filament":
            self._plume = FilamentPlume(
                source_pos=self._source_pos,
                wind_speed=self._wind.speed,
                wind_angle=self._wind.direction,
                occupancy_grid=self._grid,
                dt=cfg.FILAMENT_DT,
                K=cfg.FILAMENT_K,
                turbulence_scale=cfg.FILAMENT_TURBULENCE_SCALE,
                max_age=cfg.FILAMENT_MAX_AGE,
                filaments_per_step=cfg.FILAMENTS_PER_STEP,
                initial_sigma=cfg.FILAMENT_INITIAL_SIGMA,
                mass=cfg.FILAMENT_MASS,
                min_sigma=cfg.FILAMENT_MIN_SIGMA,
                reflection_energy=cfg.FILAMENT_REFLECTION_ENERGY,
                rng=self._rng,
                # Training: self._wind (spatial WindModel) drives both obs and advection.
                # Eval: GadenWindField drives advection; self._wind gives uniform obs fallback.
                wind_field=wind_field if wind_field is not None else self._wind,
            )
            # Warm up the plume so step 0 has some initial filaments. For GADEN
            # eval maps, warm to the scenario's playback start_time (real GADEN
            # begins playback at start_time s, by which point the plume has
            # dispersed for minutes — saturating large maps). Procedural
            # training keeps the short fresh-dispersion warmup. start_time is in
            # seconds; convert to filament steps via FILAMENT_DT.
            start_time = float(map_data.get("start_time", 0) or 0)
            if start_time > 0:
                warmup_steps = int(round(start_time / cfg.FILAMENT_DT))
            else:
                warmup_steps = cfg.FILAMENT_WARMUP_STEPS
            for _ in range(warmup_steps):
                self._plume.update()
            self._igdm = None
            self._wind_offset = None
            self._dijkstra_from_source = None
        else:
            # IGDM model
            self._igdm = IGDMModel(
                sigma_m=cfg.SIGMA_M_BASE,
                occupancy_grid=self._grid,
                dispersion_rate=cfg.DISPERSION_RATE,
                coarse_resolution=cfg.COARSE_RESOLUTION,
            )
            self._plume = None
            self._wind_offset = self._wind.get_dispersion_offset(
                cfg.WIND_DISPERSION_FACTOR
            )
            # Pre-compute Dijkstra once from the effective source (fixed per episode).
            # Dijkstra distance is symmetric, so we look up the robot's cell each step.
            eff_x = max(
                0,
                min(
                    self._source_pos[0] + self._wind_offset[0],
                    self._map_width - cfg.COARSE_RESOLUTION,
                ),
            )
            eff_y = max(
                0,
                min(
                    self._source_pos[1] + self._wind_offset[1],
                    self._map_height - cfg.COARSE_RESOLUTION,
                ),
            )
            # Snap to nearest free cell if wind pushed source into a wall
            eff_source = self._igdm.snap_to_free_cell(eff_x, eff_y)
            self._dijkstra_from_source = self._igdm.get_dijkstra_distances_from(
                eff_source
            )

        # LiDAR
        self._lidar = LidarSim(
            num_rays=cfg.LIDAR_NUM_RAYS,
            max_range=cfg.LIDAR_MAX_RANGE,
            occupancy_grid=self._grid,
            noise_std=cfg.LIDAR_NOISE_STD,
        )

        # Sensor
        self._sensor = BinarySensorModel(
            alpha=cfg.SENSOR_ALPHA,
            sigma_env=cfg.SENSOR_SIGMA_ENV,
            threshold_weight=cfg.SENSOR_THRESHOLD_WEIGHT,
        )

        # State
        self._current_step = 0
        self._gas_history = deque(
            [(None, None, 0)] * cfg.GAS_HISTORY_LENGTH,
            maxlen=cfg.GAS_HISTORY_LENGTH,
        )
        self._trajectory = [self._robot_pos.copy()]

        # Initialize sensor with first reading at start position
        if cfg.GAS_MODEL == "filament":
            conc = self._plume.concentration_at(self._robot_pos)
        else:
            conc = self._get_concentration_igdm()
        noisy = conc + (self._rng.normal(0, self._sensor.get_std(conc))
                        if self._sensor_noise else 0.0)
        self._last_noisy = noisy
        self._sensor.initialize_threshold(noisy)
        # First reading is always 0 by definition (at threshold)
        self._gas_history.append((self._robot_pos[0], self._robot_pos[1], 0))

        # Visualizer (PNG-to-disk, igdm_improved style)
        if self._viz_output_dir is not None:
            # For filament model, pass None for IGDM (visualizer falls back
            # to simple rendering). For IGDM, pass the actual model.
            igdm_for_viz = self._igdm if cfg.GAS_MODEL != "filament" else None
            self._visualizer = StepVisualizer(
                output_dir=self._viz_output_dir,
                igdm_model=igdm_for_viz,
            )
        else:
            self._visualizer = None

        # _robot_heading must be set before this call (scan() needs it)
        obs = self._build_observation()
        info = self._build_info(0.0, False, 0)
        return obs, info

    def _clamp_to_free(self, rx, ry, tx, ty, theta):
        """Validate target; if blocked, walk back along the ray until free.

        Mirrors gaden_transfer_lidar/gaden_rl_node._clamp_to_free so eval
        motion matches deployment. Returns (x, y, collided).
        """
        if self._grid.is_valid(position=(tx, ty), radius=cfg.ROBOT_RADIUS):
            return tx, ty, False
        step = cfg.STEP_SIZE - 0.05
        while step >= 0.1:
            cx = rx + step * np.cos(theta)
            cy = ry + step * np.sin(theta)
            if self._grid.is_valid(position=(cx, cy), radius=cfg.ROBOT_RADIUS):
                return cx, cy, True
            step -= 0.05
        return rx, ry, True

    def step(self, action):
        action = np.asarray(action).flatten()
        if len(action) == 1:
            # Beta-style: scalar in [0, 1] -> angle
            theta = float(np.clip(action[0], 0.0, 1.0)) * 2 * np.pi
        else:
            # Gaussian-style: (cos θ, sin θ) -> angle
            theta = float(np.arctan2(action[1], action[0]))

        self._robot_heading = theta

        # Compute target position
        dx = cfg.STEP_SIZE * np.cos(theta)
        dy = cfg.STEP_SIZE * np.sin(theta)
        new_pos = self._robot_pos + np.array([dx, dy])

        if self._deploy_motion:
            # Match the deployment node's _clamp_to_free: if the full-step
            # target is blocked, walk back along the ray in 0.05 m decrements
            # until a free cell is found (robot creeps up to the wall and stops
            # short) instead of the all-or-nothing "stay put" used in training.
            # This removes the frictionless wall-hugging that made the Python
            # eval over-succeed on tight-corridor maps vs real GADEN.
            cx, cy, collision = self._clamp_to_free(
                self._robot_pos[0], self._robot_pos[1],
                new_pos[0], new_pos[1], theta,
            )
            self._robot_pos = np.array([cx, cy], dtype=np.float64)
        else:
            # Collision check (training: all-or-nothing full step)
            collision = not self._grid.is_valid(
                position=(new_pos[0], new_pos[1]),
                radius=cfg.ROBOT_RADIUS,
            )
            if not collision:
                self._robot_pos = new_pos

        self._trajectory.append(self._robot_pos.copy())

        # Gas measurement
        if cfg.GAS_MODEL == "filament":
            self._plume.update()
            conc = self._plume.concentration_at(self._robot_pos)
        else:
            conc = self._get_concentration_igdm()

        noisy = conc + (self._rng.normal(0, self._sensor.get_std(conc))
                        if self._sensor_noise else 0.0)
        self._last_noisy = noisy
        self._sensor.update_threshold(noisy)
        binary = self._sensor.get_binary_measurement(noisy)
        self._gas_history.append((self._robot_pos[0], self._robot_pos[1], binary))

        # LiDAR (part of observation, computed in _build_observation)

        # Rewards
        reward = cfg.R_STEP

        if collision:
            reward += cfg.R_COLLISION

        if binary == 1:
            reward += cfg.R_DETECTION

        # Trajectory-loop proximity penalty: discourage tight revisits of
        # recent positions. trajectory[-1] is the just-appended current pos;
        # walk back through the previous LOOP_HISTORY points with age-decay.
        n_traj = len(self._trajectory)
        for age in range(1, cfg.LOOP_HISTORY + 1):
            idx = -1 - age
            if -idx > n_traj:
                break
            past = self._trajectory[idx]
            if np.linalg.norm(self._robot_pos - past) < cfg.D_LOOP:
                reward += cfg.R_LOOP_BASE * (cfg.LOOP_DECAY ** (age - 1))


        dist = np.linalg.norm(self._robot_pos - self._source_pos)
        terminated = False
        truncated = False

        if dist < cfg.D_SUCCESS:
            reward += cfg.R_SUCCESS
            terminated = True

        self._current_step += 1
        if self._current_step >= cfg.MAX_STEPS:
            truncated = True

        obs = self._build_observation()
        info = self._build_info(dist, collision, binary)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Save a 2-panel PNG frame via StepVisualizer (igdm_improved style).

        Requires the env to have been constructed with ``viz_output_dir``.
        Returns ``None``.
        """
        if self._visualizer is None:
            raise RuntimeError(
                "render() requires viz_output_dir to be set on the environment."
            )

        last_entry = self._gas_history[-1]
        digital_value = int(last_entry[2]) if last_entry[0] is not None else None
        dist = float(np.linalg.norm(self._robot_pos - self._source_pos))

        if cfg.GAS_MODEL == "filament":
            # Filament model: no effective source, no IGDM concentration field
            eff_source = None
        else:
            # IGDM model: compute effective source for rendering
            eff_x = max(
                0,
                min(
                    self._source_pos[0] + self._wind_offset[0],
                    self._map_width - cfg.COARSE_RESOLUTION,
                ),
            )
            eff_y = max(
                0,
                min(
                    self._source_pos[1] + self._wind_offset[1],
                    self._map_height - cfg.COARSE_RESOLUTION,
                ),
            )
            eff_source = self._igdm.snap_to_free_cell(eff_x, eff_y)

        self._visualizer.save_step(
            robot_pos=tuple(self._robot_pos),
            trajectory=self._trajectory,
            true_source=tuple(self._source_pos),
            step_num=self._current_step,
            current_step=self._current_step,
            occupancy_grid=self._grid,
            distance_to_true=dist,
            d_success_thr=cfg.D_SUCCESS,
            sensor_reading=float(self._last_noisy),
            sensor_threshold=float(self._sensor.threshold),
            digital_value=digital_value,
            wind_offset=self._wind_offset,
            eff_source=eff_source,
            filaments=self._plume.get_all_filaments()
            if cfg.GAS_MODEL == "filament" else None,
            wind_speed=self._wind.speed,
            wind_dir=self._wind.direction,
        )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_concentration_igdm(self):
        """Compute true gas concentration at robot position using IGDM.

        Uses the precomputed Dijkstra grid from the effective source.
        Distance is symmetric so dist(source→robot) == dist(robot→source).
        """
        r, c = self._igdm._world_to_coarse_idx(
            self._robot_pos[0], self._robot_pos[1]
        )
        d = self._dijkstra_from_source[r, c]
        if np.isinf(d):
            return 0.0
        d = max(d, 0.1)
        sigma_m = self._igdm.get_sigma_m(self._current_step)
        return cfg.SOURCE_RELEASE_RATE * np.exp(-(d ** 2) / (2 * sigma_m ** 2))

    def _build_observation(self):
        """Build the 107-dim normalized observation vector.

        Wind slice (dims 104-105): local wind (Ux_norm, Uy_norm) at robot
        position, encoded as (component / max_speed_uniform + 1) / 2 in [0, 1].
        """
        gas_entries = []
        for ax, ay, b in self._gas_history:
            if ax is None:
                gas_entries.extend([0.5, 0.5, 0.0])
            else:
                rel_x = 0.5 + (ax - self._robot_pos[0]) / (2.0 * self._map_width)
                rel_y = 0.5 + (ay - self._robot_pos[1]) / (2.0 * self._map_height)
                gas_entries.extend([
                    np.clip(rel_x, 0.0, 1.0),
                    np.clip(rel_y, 0.0, 1.0),
                    float(b),
                ])
        gas = np.array(gas_entries, dtype=np.float32)
        lidar = self._lidar.scan(tuple(self._robot_pos), self._robot_heading).astype(np.float32)
        pos = np.array([
            self._robot_pos[0] / self._map_width,
            self._robot_pos[1] / self._map_height,
        ], dtype=np.float32)
        wind = self._wind.get_local_wind(self._robot_pos)  # local wind (Ux_norm, Uy_norm)
        time_frac = np.array([self._current_step / cfg.MAX_STEPS], dtype=np.float32)

        obs = np.concatenate([gas, lidar, pos, wind, time_frac])
        return np.clip(obs, 0.0, 1.0)

    def _build_info(self, dist, collision, binary):
        return {
            "distance_to_source": dist,
            "gas_reading": binary,
            "collision": collision,
            "step": self._current_step,
        }
