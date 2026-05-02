"""
NavigationEnv: wall-following navigation for LiDAR validation.

Empty room of random size, start and goal both near a wall.
Uses LiDAR wall-proximity for reward shaping to encourage wall-following.
"""

import numpy as np
import gymnasium

from .map_generator import MapGenerator
from .lidar_sim import LidarSim
from .. import config as cfg

# Near-wall placement band (meters from the boundary wall interior face)
WALL_NEAR_MIN = 0.4
WALL_NEAR_MAX = 1.5

# Wall-following reward: Gaussian centered at this distance from nearest wall
WALL_TARGET_DIST = 0.5   # meters
WALL_SIGMA = 0.3         # meters  ->  denominator = 2 * 0.3^2 = 0.18

# Navigation constants
MIN_NAV_DIST = 3.0       # minimum start-to-goal Euclidean distance (meters)
NAV_MAX_STEPS = 400

NAV_OBS_DIM = cfg.LIDAR_NUM_RAYS + 5  # lidar + pos(2) + goal_dir(2) + goal_dist(1)
NAV_ACTION_DIM = 2


class NavigationEnv(gymnasium.Env):
    """
    Gymnasium environment for wall-following navigation.

    Observation (77,):
        [0:72]   LiDAR -- 72 normalized ray distances in [0, 1]
        [72:74]  pos   -- (x / room_width, y / room_height)
        [74:76]  goal_dir -- (cos theta_goal, sin theta_goal)
        [76]     goal_dist -- dist(robot, goal) / room_diagonal

    Action (2,):
        (cos theta, sin theta) from a 2D Gaussian; decoded via atan2 to robot heading.
    """

    metadata = {"render_modes": []}

    def __init__(self, width_range=None, height_range=None, viz_output_dir=None):
        super().__init__()
        _obs_low  = np.concatenate([
            np.zeros(cfg.LIDAR_NUM_RAYS, dtype=np.float32),   # lidar [0, 1]
            np.zeros(2, dtype=np.float32),                     # pos [0, 1]
            np.full(2, -1.0, dtype=np.float32),               # goal_dir [-1, 1]
            np.zeros(1, dtype=np.float32),                     # goal_dist [0, 1]
        ])
        _obs_high = np.ones(NAV_OBS_DIM, dtype=np.float32)

        self.observation_space = gymnasium.spaces.Box(
            low=_obs_low, high=_obs_high, dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(NAV_ACTION_DIM,), dtype=np.float32
        )
        self._width_range = width_range
        self._height_range = height_range
        self._viz_output_dir = viz_output_dir
        self._visualizer = None
        self._rng = np.random.default_rng()
        self._map_gen = MapGenerator(rng=self._rng,
                                     width_range=width_range,
                                     height_range=height_range)

        # Set during reset()
        self.grid = None
        self.lidar = None
        self.robot_pos = None
        self.goal_pos = None
        self._step_count = 0
        self._prev_dist = 0.0
        self._room_width = 0.0
        self._room_height = 0.0
        self._trajectory = []
        self._cum_reward = 0.0
        self._last_reward = 0.0
        self._last_heading = 0.0
        self._last_lidar = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._map_gen = MapGenerator(rng=self._rng,
                                         width_range=self._width_range,
                                         height_range=self._height_range)

        result = self._map_gen.generate(template_id=0)
        self.grid = result["grid"]
        self._room_width = result["width"]
        self._room_height = result["height"]

        self.lidar = LidarSim(
            num_rays=cfg.LIDAR_NUM_RAYS,
            max_range=cfg.LIDAR_MAX_RANGE,
            occupancy_grid=self.grid,
            noise_std=cfg.LIDAR_NOISE_STD,
        )

        self.robot_pos, self.goal_pos = self._place_start_and_goal()
        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(
            np.array(self.robot_pos) - np.array(self.goal_pos)
        ))

        self._trajectory = [self.robot_pos]
        self._cum_reward = 0.0
        self._last_reward = 0.0
        self._last_heading = 0.0
        self._last_lidar = self.lidar.scan(self.robot_pos, heading=0.0)

        if self._viz_output_dir is not None:
            from .nav_visualizer import NavStepVisualizer
            self._visualizer = NavStepVisualizer(self._viz_output_dir)
        else:
            self._visualizer = None

        obs = self._build_obs(heading=0.0, lidar=self._last_lidar)
        return obs, {}

    def step(self, action):
        cos_a = float(action[0])
        sin_a = float(action[1])
        heading = float(np.arctan2(sin_a, cos_a))

        new_pos = (
            self.robot_pos[0] + cfg.STEP_SIZE * np.cos(heading),
            self.robot_pos[1] + cfg.STEP_SIZE * np.sin(heading),
        )

        if not self.grid.is_valid(new_pos, cfg.ROBOT_RADIUS):
            reward = cfg.R_COLLISION
        else:
            self.robot_pos = new_pos
            self._trajectory.append(self.robot_pos)
            reward = 0.0

        self._step_count += 1

        lidar = self.lidar.scan(self.robot_pos, heading=heading)
        self._last_lidar = lidar

        # Potential-based progress shaping
        curr_dist = float(np.linalg.norm(
            np.array(self.robot_pos) - np.array(self.goal_pos)
        ))
        entry_dist = self._prev_dist
        reward += 0.5 * (self._prev_dist - curr_dist)
        self._prev_dist = curr_dist

        # Wall-following: Gaussian reward centered at WALL_TARGET_DIST
        min_wall_dist = float(np.min(lidar)) * cfg.LIDAR_MAX_RANGE
        reward += 0.3 * float(np.exp(
            -((min_wall_dist - WALL_TARGET_DIST) ** 2) / (2.0 * WALL_SIGMA ** 2)
        ))

        reward += cfg.R_STEP

        # entry_dist handles the case where the robot starts a step already at the goal
        terminated = curr_dist < cfg.D_SUCCESS or entry_dist < cfg.D_SUCCESS
        if terminated:
            reward += 100.0

        truncated = self._step_count >= NAV_MAX_STEPS
        if truncated:
            reward += cfg.R_MAX_STEPS

        self._cum_reward += reward
        self._last_reward = float(reward)
        self._last_heading = heading

        obs = self._build_obs(heading=heading, lidar=lidar)
        return obs, float(reward), terminated, truncated, {"terminated": terminated}

    def render(self):
        if self._visualizer is None:
            raise RuntimeError(
                "render() requires viz_output_dir to be set at construction time"
            )
        lidar_angles = (
            np.linspace(0, 2 * np.pi, cfg.LIDAR_NUM_RAYS, endpoint=False)
            + self._last_heading
        ) % (2 * np.pi)
        wall_dist = float(np.min(self._last_lidar)) * cfg.LIDAR_MAX_RANGE

        self._visualizer.save_step(
            occupancy_grid = self.grid,
            robot_pos      = self.robot_pos,
            heading        = self._last_heading,
            goal_pos       = self.goal_pos,
            lidar_rays     = self._last_lidar,
            lidar_angles   = lidar_angles,
            trajectory     = list(self._trajectory),
            wall_dist      = wall_dist,
            step_reward    = self._last_reward,
            cum_reward     = self._cum_reward,
            step_num       = self._step_count,
            room_width     = self._room_width,
            room_height    = self._room_height,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_obs(self, heading: float, lidar=None) -> np.ndarray:
        if lidar is None:
            lidar = self.lidar.scan(self.robot_pos, heading=heading)

        pos_norm = np.array([
            self.robot_pos[0] / self._room_width,
            self.robot_pos[1] / self._room_height,
        ], dtype=np.float32)

        dx = self.goal_pos[0] - self.robot_pos[0]
        dy = self.goal_pos[1] - self.robot_pos[1]
        dist = float(np.sqrt(dx ** 2 + dy ** 2))
        diagonal = float(np.sqrt(self._room_width ** 2 + self._room_height ** 2))

        goal_dir = np.array([dx / (dist + 1e-8), dy / (dist + 1e-8)], dtype=np.float32)
        goal_dist = np.array([dist / diagonal], dtype=np.float32)

        return np.concatenate([lidar.astype(np.float32), pos_norm, goal_dir, goal_dist])

    def _place_start_and_goal(self, max_retries: int = 500):
        near_wall = self._get_near_wall_positions()

        if len(near_wall) < 2:
            # Fallback: use any valid cells
            near_wall = self._map_gen._get_free_cells(self.grid)
            if len(near_wall) < 2:
                raise RuntimeError("Not enough valid cells in generated map")

        for _ in range(max_retries):
            idx_s = self._rng.integers(0, len(near_wall))
            start = near_wall[idx_s]
            dists = np.linalg.norm(near_wall - start, axis=1)
            far_idx = np.where(dists >= MIN_NAV_DIST)[0]
            if len(far_idx) == 0:
                continue
            idx_g = self._rng.choice(far_idx)
            return tuple(near_wall[idx_s]), tuple(near_wall[idx_g])

        # Fallback: pick any farthest pair
        idx_s = self._rng.integers(0, len(near_wall))
        start = near_wall[idx_s]
        dists = np.linalg.norm(near_wall - start, axis=1)
        return tuple(start), tuple(near_wall[np.argmax(dists)])

    def _get_near_wall_positions(self) -> np.ndarray:
        """Return valid positions whose distance to the nearest boundary wall
        is in [WALL_NEAR_MIN, WALL_NEAR_MAX].

        For template 0 (empty room), the distance to the nearest wall is:
            min(px - t, width - t - px, py - t, height - t - py)
        where t = WALL_THICKNESS.
        """
        t = cfg.WALL_THICKNESS
        w = self._room_width
        h = self._room_height
        res = cfg.GRID_RESOLUTION
        step = max(1, int(0.5 / res))

        cells = []
        for gy in range(0, self.grid.grid_height, step):
            for gx in range(0, self.grid.grid_width, step):
                if not self.grid.is_valid(gx=gx, gy=gy, radius=cfg.ROBOT_RADIUS):
                    continue
                px = (gx + 0.5) * res
                py = (gy + 0.5) * res
                d_wall = min(px - t, w - t - px, py - t, h - t - py)
                if WALL_NEAR_MIN <= d_wall <= WALL_NEAR_MAX:
                    cells.append((px, py))

        return np.array(cells, dtype=np.float64) if cells else np.empty((0, 2))
