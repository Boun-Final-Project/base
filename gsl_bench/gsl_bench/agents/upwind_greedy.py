"""Classical sanity baseline: surge upwind on a gas hit, random-walk otherwise.

The anemometer's wind_direction points DOWNWIND (the direction the gas travels),
so the source lies at wind_direction + pi.
"""
import math
import random

from gsl_bench.agent import GSLAgent, Observation, Waypoint


class UpwindGreedyAgent(GSLAgent):

    def __init__(self, config=None):
        cfg = config or {}
        self.rng = random.Random(cfg.get('seed', 0))
        self.gas_threshold = float(cfg.get('gas_threshold', 50.0))
        self.step_size = float(cfg.get('step_size', 0.5))
        self.obs = None

    def observe(self, obs: Observation) -> None:
        self.obs = obs

    def act(self) -> Waypoint:
        if self.obs.gas_ppm > self.gas_threshold and self.obs.wind_speed > 1e-3:
            upwind = self.obs.wind_direction + math.pi
            # Fan out around the upwind bearing so a blocked surge still makes progress.
            for dth in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
                th = upwind + dth
                x = self.obs.x + self.step_size * math.cos(th)
                y = self.obs.y + self.step_size * math.sin(th)
                if self.obs.map.is_free(x, y):
                    return Waypoint(x, y, theta=th)

        for _ in range(20):                        # casting: no gas (or boxed in)
            th = self.rng.uniform(-math.pi, math.pi)
            x = self.obs.x + self.step_size * math.cos(th)
            y = self.obs.y + self.step_size * math.sin(th)
            if self.obs.map.is_free(x, y):
                return Waypoint(x, y, theta=th)
        return Waypoint(self.obs.x, self.obs.y)
