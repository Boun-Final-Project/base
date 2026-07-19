"""Faithful continuous-motion ADSM adapter backed by the C++ decision engine."""
from __future__ import annotations

from typing import Optional

from gsl_bench.agent import GSLAgent, Observation, ScenarioInfo, Waypoint

try:
    import adsm_core_py
except ImportError as exc:  # give a useful error instead of a mysterious agent load failure
    raise ImportError(
        'ADSM requires the adsm C++ package. Build and source the workspace with '
        '`colcon build --packages-select adsm gsl_bench`.') from exc


class AdsmAgent(GSLAgent):
    motion_mode = 'continuous'
    decision_rate_hz = 1.0
    max_goal_distance = 3.0
    required_pose_source = 'tf'
    required_map_source = 'slam'
    required_motion = 'nav2'
    required_nav_profile = 'faithful'

    _CONFIG_KEYS = (
        'k1', 'random_sample_r', 'goal_cluster_num', 'obs_r',
        'goal_reach_th', 'resample_time_th', 'gas_high_th', 'gas_low_th',
        'sensor_window_length', 'frontier_search_th', 'rrt_max_iter',
        'rrt_max_r', 'rrt_min_r', 'rrt_step_size', 'stuck_duration_th',
    )

    def __init__(self, config=None):
        raw = dict(config or {})
        cfg = adsm_core_py.Config()
        for key in self._CONFIG_KEYS:
            if key in raw:
                setattr(cfg, key, raw[key])
        self._requested_seed = raw.get('seed')
        self._engine = adsm_core_py.Engine(cfg)
        self._seed: Optional[int] = None
        self._latest: Optional[Observation] = None
        self._last_decision = None

    def reset(self, scenario: ScenarioInfo) -> None:
        del scenario
        self._seed = int(self._engine.reset(self._requested_seed))
        self._latest = None
        self._last_decision = None

    def observe(self, obs: Observation) -> None:
        self._latest = obs

    def act(self) -> Waypoint:
        if self._latest is None:
            raise RuntimeError('act() called before observe()')
        o = self._latest
        m = o.map
        self._last_decision = self._engine.step(
            o.x, o.y, o.theta, o.gas_ppm, o.wind_speed, o.wind_direction,
            o.sim_time, m.grid, m.resolution, m.origin_x, m.origin_y)
        # The paper's move_base goal used identity orientation (world yaw 0).
        return Waypoint(self._last_decision.x, self._last_decision.y, theta=0.0)

    def metadata(self) -> dict:
        out = {
            'implementation': 'adsm_core_py',
            'upstream': 'mwanggh/An-adaptive-robot-search-algorithm',
            'seed': self._seed,
            'sensor_adaptation': 'PID',
        }
        if self._last_decision is not None:
            out.update({
                'iteration': self._last_decision.iteration,
                'goal_type': self._last_decision.goal_type,
                'gas_hit': self._last_decision.gas_hit,
                'epi_size': self._last_decision.epi_size,
                'epr_size': self._last_decision.epr_size,
            })
        return out
