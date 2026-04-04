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
from .lidar_sim import LidarSim
from .wind_model import WindModel
from .sensor_model import BinarySensorModel


class GasSourceEnv(gymnasium.Env):
    """Gas source localization environment.

    Observation (39-dim):
        [gas_history (10)] + [lidar (24)] + [pos_x, pos_y] + [wind_speed, wind_dir] + [time]
        All values normalized to [0, 1].

    Action (1-dim):
        Scalar in [0, 1], scaled internally to angle [0, 2*pi).
        Robot moves STEP_SIZE meters in that direction.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, seed=None, template_id=None):
        super().__init__()
        self.render_mode = render_mode
        self._template_id = template_id  # None = random, 0-5 = fixed template

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
        self._lidar = None
        self._sensor = None
        self._source_pos = None
        self._robot_pos = None
        self._map_width = 0.0
        self._map_height = 0.0
        self._current_step = 0
        self._gas_history = None
        self._visited_cells = None
        self._trajectory = []
        self._wind_offset = None
        self._dijkstra_from_source = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._map_gen = MapGenerator(rng=self._rng)

        # Generate map
        map_data = self._map_gen.generate(template_id=self._template_id)
        self._grid = map_data["grid"]
        self._source_pos = np.array(map_data["source_pos"], dtype=np.float64)
        self._robot_pos = np.array(map_data["robot_pos"], dtype=np.float64)
        self._map_width = map_data["width"]
        self._map_height = map_data["height"]

        # IGDM
        self._igdm = IGDMModel(
            sigma_m=cfg.SIGMA_M_BASE,
            occupancy_grid=self._grid,
            dispersion_rate=cfg.DISPERSION_RATE,
            coarse_resolution=cfg.COARSE_RESOLUTION,
        )

        # LiDAR
        self._lidar = LidarSim(
            num_rays=cfg.LIDAR_NUM_RAYS,
            max_range=cfg.LIDAR_MAX_RANGE,
            occupancy_grid=self._grid,
        )

        # Wind
        self._wind.randomize(self._rng)
        self._wind_offset = self._wind.get_dispersion_offset(cfg.WIND_DISPERSION_FACTOR)

        # Pre-compute Dijkstra once from the effective source (fixed per episode).
        # Dijkstra distance is symmetric, so we look up the robot's cell each step.
        eff_source = (
            self._source_pos[0] + self._wind_offset[0],
            self._source_pos[1] + self._wind_offset[1],
        )
        eff_source = (
            max(0, min(eff_source[0], self._map_width - cfg.COARSE_RESOLUTION)),
            max(0, min(eff_source[1], self._map_height - cfg.COARSE_RESOLUTION)),
        )
        self._dijkstra_from_source = self._igdm.get_dijkstra_distances_from(eff_source)

        # Sensor
        self._sensor = BinarySensorModel(
            alpha=cfg.SENSOR_ALPHA,
            sigma_env=cfg.SENSOR_SIGMA_ENV,
            threshold_weight=cfg.SENSOR_THRESHOLD_WEIGHT,
        )

        # State
        self._current_step = 0
        self._gas_history = deque([0] * cfg.GAS_HISTORY_LENGTH,
                                  maxlen=cfg.GAS_HISTORY_LENGTH)
        self._visited_cells = set()
        self._trajectory = [self._robot_pos.copy()]

        # Mark starting cell as visited
        cell_key = self._cell_key(self._robot_pos)
        self._visited_cells.add(cell_key)

        # Initialize sensor with first reading at start position
        conc = self._get_concentration()
        noisy = conc + self._rng.normal(0, self._sensor.get_std(conc))
        self._sensor.initialize_threshold(noisy)
        # First reading is always 0 by definition (at threshold)
        self._gas_history.append(0)

        obs = self._build_observation()
        info = self._build_info(0.0, False, 0)
        return obs, info

    def step(self, action):
        action_val = float(np.clip(action, 0.0, 1.0).flat[0])
        theta = action_val * 2 * np.pi

        # Compute target position
        dx = cfg.STEP_SIZE * np.cos(theta)
        dy = cfg.STEP_SIZE * np.sin(theta)
        new_pos = self._robot_pos + np.array([dx, dy])

        # Collision check
        collision = not self._grid.is_valid(
            position=(new_pos[0], new_pos[1]),
            radius=cfg.ROBOT_RADIUS,
        )
        if not collision:
            self._robot_pos = new_pos

        self._trajectory.append(self._robot_pos.copy())

        # Gas measurement
        conc = self._get_concentration()
        noisy = conc + self._rng.normal(0, self._sensor.get_std(conc))
        self._sensor.update_threshold(noisy)
        binary = self._sensor.get_binary_measurement(noisy)
        self._gas_history.append(binary)

        # LiDAR (part of observation, computed in _build_observation)

        # Rewards
        reward = cfg.R_STEP

        if collision:
            reward += cfg.R_COLLISION

        if binary == 1:
            reward += cfg.R_DETECTION

        cell_key = self._cell_key(self._robot_pos)
        if cell_key not in self._visited_cells:
            reward += cfg.R_NEW_CELL
            self._visited_cells.add(cell_key)

        dist = np.linalg.norm(self._robot_pos - self._source_pos)
        terminated = False
        truncated = False

        if dist < cfg.D_SUCCESS:
            reward += cfg.R_SUCCESS
            terminated = True

        self._current_step += 1
        if self._current_step >= cfg.MAX_STEPS:
            reward += cfg.R_MAX_STEPS
            truncated = True

        obs = self._build_observation()
        info = self._build_info(dist, collision, binary)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        import matplotlib.pyplot as plt
        import matplotlib
        if self.render_mode == "rgb_array":
            matplotlib.use("Agg")

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Gas concentration heatmap (on the coarse Dijkstra grid)
        dists = self._dijkstra_from_source  # (rows, cols)
        sigma_m = self._igdm.get_sigma_m(self._current_step)
        conc = np.where(
            np.isinf(dists), 0.0,
            cfg.SOURCE_RELEASE_RATE * np.exp(
                -np.maximum(dists, 0.1) ** 2 / (2 * sigma_m ** 2)
            ),
        )
        ax.imshow(
            conc,
            origin="lower",
            extent=[0, self._map_width, 0, self._map_height],
            cmap="YlOrRd",
            alpha=0.5,
            aspect="equal",
            interpolation="bilinear",
        )

        # Occupancy grid (walls on top of heatmap)
        wall_mask = np.ma.masked_where(self._grid.grid == 0, self._grid.grid)
        ax.imshow(
            wall_mask,
            origin="lower",
            extent=[0, self._map_width, 0, self._map_height],
            cmap="Greys",
            vmin=0, vmax=1,
            aspect="equal",
        )

        # Trajectory
        if len(self._trajectory) > 1:
            traj = np.array(self._trajectory)
            ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1, alpha=0.5)

        # Robot
        ax.plot(self._robot_pos[0], self._robot_pos[1], "bo", markersize=8)

        # Source
        ax.plot(self._source_pos[0], self._source_pos[1], "r*", markersize=15)

        # LiDAR rays
        lidar_dists = self._lidar.scan(tuple(self._robot_pos))
        for i, angle in enumerate(self._lidar.ray_angles):
            d = lidar_dists[i] * cfg.LIDAR_MAX_RANGE
            ex = self._robot_pos[0] + d * np.cos(angle)
            ey = self._robot_pos[1] + d * np.sin(angle)
            ax.plot([self._robot_pos[0], ex], [self._robot_pos[1], ey],
                    "g-", linewidth=0.5, alpha=0.3)

        # Wind arrow
        ws, wd = self._wind.speed, self._wind.direction
        arrow_len = 1.0
        ax.annotate(
            "",
            xy=(1.5 + arrow_len * np.cos(wd), self._map_height - 1.5 + arrow_len * np.sin(wd)),
            xytext=(1.5, self._map_height - 1.5),
            arrowprops=dict(arrowstyle="->", color="purple", lw=2),
        )
        ax.text(1.5, self._map_height - 0.5, f"wind {ws:.1f} m/s", fontsize=8, color="purple")

        # Info text
        gas_str = "".join(str(g) for g in self._gas_history)
        dist = np.linalg.norm(self._robot_pos - self._source_pos)
        ax.set_title(
            f"Step {self._current_step}/{cfg.MAX_STEPS}  "
            f"Dist: {dist:.1f}m  Gas: [{gas_str}]"
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            return buf
        else:
            plt.show()
            plt.close(fig)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_concentration(self):
        """Compute true gas concentration at robot position.

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

    def _cell_key(self, pos):
        return (int(pos[0] / cfg.VISITED_CELL_RESOLUTION),
                int(pos[1] / cfg.VISITED_CELL_RESOLUTION))

    def _build_observation(self):
        """Build the 39-dim normalized observation vector."""
        gas = np.array(list(self._gas_history), dtype=np.float32)
        lidar = self._lidar.scan(tuple(self._robot_pos)).astype(np.float32)
        pos = np.array([
            self._robot_pos[0] / self._map_width,
            self._robot_pos[1] / self._map_height,
        ], dtype=np.float32)
        wind = np.array(self._wind.get_observation(), dtype=np.float32)
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
