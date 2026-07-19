"""The docs example: a complete agent in ~20 lines."""
import math
import random

from gsl_bench.agent import GSLAgent, Observation, Waypoint


class RandomWalkAgent(GSLAgent):
    """Pick a random free direction, step 0.5 m."""

    def __init__(self, config=None):
        self.rng = random.Random((config or {}).get('seed', 0))
        self.step_size = float((config or {}).get('step_size', 0.5))
        self.obs = None

    def observe(self, obs: Observation) -> None:
        self.obs = obs

    def act(self) -> Waypoint:
        for _ in range(20):                       # rejection-sample a free target
            th = self.rng.uniform(-math.pi, math.pi)
            x = self.obs.x + self.step_size * math.cos(th)
            y = self.obs.y + self.step_size * math.sin(th)
            if self.obs.map.is_free(x, y):
                return Waypoint(x, y, theta=th)
        return Waypoint(self.obs.x, self.obs.y)   # boxed in: stay
