"""Open-loop crosswind sweep with a slow upwind drift.

Gas-blind precursor to surge-cast's CAST state: turns happen on a fixed
crosswind-distance schedule, never on gas readings.
"""
from gsl_bench.agent import GSLAgent, Observation, Waypoint
from gsl_bench.agents._casting import crosswind_step
from gsl_bench.agents.random_walk import RandomWalkAgent


class ZigzagAgent(GSLAgent):

    def __init__(self, config=None):
        cfg = config or {}
        self.step_size = float(cfg.get('step_size', 0.5))
        self.leg_length = float(cfg.get('leg_length', 2.0))
        self.amplitude_growth = float(cfg.get('amplitude_growth', 0.0))
        self.upwind_drift = float(cfg.get('upwind_drift', 0.15))
        self._fallback = RandomWalkAgent(cfg)   # no wind vector -> can't zigzag
        self.side = 1
        self.leg_dist = 0.0
        self.obs = None

    def reset(self, scenario) -> None:
        self.side = 1
        self.leg_dist = 0.0

    def observe(self, obs: Observation) -> None:
        self.obs = obs
        self._fallback.observe(obs)

    def act(self) -> Waypoint:
        if self.obs.wind_speed <= 1e-3:
            return self._fallback.act()

        x, y, th = crosswind_step(
            self.obs, self.side, self.step_size, self.upwind_drift)

        self.leg_dist += self.step_size
        if self.leg_dist >= self.leg_length:
            self.side *= -1
            self.leg_dist = 0.0
            self.leg_length += self.amplitude_growth

        return Waypoint(x, y, theta=th)
