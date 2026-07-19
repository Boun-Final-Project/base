"""Shared crosswind-stepping geometry for ZigzagAgent and SurgeCastAgent's CAST state.

The anemometer's wind_direction points DOWNWIND, so the crosswind axis is
perpendicular to the upwind bearing (wind_direction + pi).
"""
import math


def crosswind_step(obs, side: int, step_size: float, upwind_drift: float = 0.0):
    """One crosswind hop, `side` = +1 (left of upwind) or -1 (right).

    `upwind_drift` (0..step_size) blends a small upwind bias INTO the bearing
    before stepping, so the hop still covers exactly `step_size` rather than
    stacking a second vector on top of it (which would blow past max_hop).

    Fans out around the resulting bearing if the map blocks it, same pattern
    as the other baseline agents. Returns (x, y, theta); stays put (returns
    obs.x, obs.y) if boxed in on every fan angle.
    """
    crosswind = obs.wind_direction + side * (math.pi / 2.0)
    if upwind_drift > 0.0 and step_size > 0.0:
        upwind = obs.wind_direction + math.pi
        ratio = min(upwind_drift / step_size, 1.0)
        vx = math.cos(crosswind) + ratio * math.cos(upwind)
        vy = math.sin(crosswind) + ratio * math.sin(upwind)
        crosswind = math.atan2(vy, vx)

    for dth in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
        th = crosswind + dth
        x = obs.x + step_size * math.cos(th)
        y = obs.y + step_size * math.sin(th)
        if obs.map.is_free(x, y):
            return x, y, th
    return obs.x, obs.y, crosswind
