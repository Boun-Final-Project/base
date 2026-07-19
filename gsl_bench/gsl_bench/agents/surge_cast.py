"""Moth-inspired reactive chemotaxis (Kuwana/Cardé & Willis silkworm-moth model).

Three states, switched purely on the latest gas reading:
  SURGE  - gas_ppm > gas_threshold: drive straight upwind.
  CAST   - lost the hit: crosswind zigzag (shared with ZigzagAgent), amplitude
           growing each leg, trying to re-cross the plume.
  SPIRAL - CAST failed to reacquire for lost_steps_to_spiral steps: widen into
           an outward spiral around the point the plume was lost, until any
           whiff is re-detected.
Re-detecting gas at any point drops straight back to SURGE.
"""
import math

from gsl_bench.agent import GSLAgent, Observation, Waypoint
from gsl_bench.agents._casting import crosswind_step

SURGE, CAST, SPIRAL = 'SURGE', 'CAST', 'SPIRAL'


class SurgeCastAgent(GSLAgent):

    def __init__(self, config=None):
        cfg = config or {}
        self.gas_threshold = float(cfg.get('gas_threshold', 50.0))
        self.step_size = float(cfg.get('step_size', 0.5))
        self.leg_length0 = float(cfg.get('leg_length', 1.0))
        self.amplitude_growth = float(cfg.get('amplitude_growth', 0.3))
        self.lost_steps_to_spiral = int(cfg.get('lost_steps_to_spiral', 40))
        self.spiral_growth = float(cfg.get('spiral_growth', 0.15))
        self.obs = None
        self._reset_state()

    def _reset_state(self):
        self.state = SURGE
        self.side = 1
        self.leg_length = self.leg_length0
        self.leg_dist = 0.0
        self.steps_since_detect = 0
        self.spiral_cx = None
        self.spiral_cy = None
        self.spiral_angle = 0.0
        self.spiral_radius = self.step_size

    def reset(self, scenario) -> None:
        self._reset_state()

    def observe(self, obs: Observation) -> None:
        self.obs = obs

    def act(self) -> Waypoint:
        obs = self.obs
        if obs.gas_ppm > self.gas_threshold and obs.wind_speed > 1e-3:
            self._enter_surge()
            return self._surge()

        self.steps_since_detect += 1

        if self.steps_since_detect > self.lost_steps_to_spiral:
            if self.state != SPIRAL:
                self._enter_spiral()
            return self._spiral()

        if self.state != CAST:
            self._enter_cast()
        return self._cast()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _enter_surge(self):
        self.state = SURGE
        self.steps_since_detect = 0
        self.leg_length = self.leg_length0
        self.leg_dist = 0.0

    def _enter_cast(self):
        self.state = CAST
        self.leg_length = self.leg_length0
        self.leg_dist = 0.0

    def _enter_spiral(self):
        self.state = SPIRAL
        self.spiral_cx, self.spiral_cy = self.obs.x, self.obs.y
        self.spiral_angle = 0.0
        self.spiral_radius = self.step_size

    # ------------------------------------------------------------------
    # Per-state motion
    # ------------------------------------------------------------------

    def _surge(self) -> Waypoint:
        upwind = self.obs.wind_direction + math.pi
        for dth in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
            th = upwind + dth
            x = self.obs.x + self.step_size * math.cos(th)
            y = self.obs.y + self.step_size * math.sin(th)
            if self.obs.map.is_free(x, y):
                return Waypoint(x, y, theta=th)
        return Waypoint(self.obs.x, self.obs.y)

    def _cast(self) -> Waypoint:
        x, y, th = crosswind_step(self.obs, self.side, self.step_size)
        self.leg_dist += self.step_size
        if self.leg_dist >= self.leg_length:
            self.side *= -1
            self.leg_dist = 0.0
            self.leg_length += self.amplitude_growth
        return Waypoint(x, y, theta=th)

    def _spiral(self) -> Waypoint:
        """Archimedean spiral about the point the plume was lost; radius grows
        a fixed amount each full revolution."""
        d_theta = self.step_size / max(self.spiral_radius, 0.1)
        self.spiral_angle += d_theta
        if self.spiral_angle >= 2 * math.pi:
            self.spiral_angle -= 2 * math.pi
            self.spiral_radius += self.spiral_growth

        th = self.spiral_angle
        for r in (self.spiral_radius, self.spiral_radius + self.spiral_growth,
                  self.spiral_radius + 2 * self.spiral_growth):
            x = self.spiral_cx + r * math.cos(th)
            y = self.spiral_cy + r * math.sin(th)
            if self.obs.map.is_free(x, y):
                return Waypoint(x, y, theta=th)
        return Waypoint(self.obs.x, self.obs.y)
