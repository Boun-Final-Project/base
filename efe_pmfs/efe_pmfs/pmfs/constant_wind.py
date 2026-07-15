"""
Constant wind field.

Replaces PMFS's GMRF-wind. Every free cell gets the same (u_mean, v_mean)
provided as a ROS param. Wrapped in a class so a potential-flow solver could
later drop in without touching the rest of the code.
"""
import numpy as np


class ConstantWindField:
    def __init__(self, grid_shape, u_mean, v_mean):
        """
        Parameters
        ----------
        grid_shape : (height, width)
        u_mean, v_mean : float
            World-frame wind components in m/s. Downwind vector.
        """
        h, w = grid_shape
        self.height = h
        self.width = w
        self.u_mean = float(u_mean)
        self.v_mean = float(v_mean)
        self.u = np.full((h, w), self.u_mean, dtype=np.float32)
        self.v = np.full((h, w), self.v_mean, dtype=np.float32)

    @property
    def speed(self):
        return float(np.hypot(self.u_mean, self.v_mean))

    @property
    def downwind_angle(self):
        """Angle (rad) of the downwind vector in world frame. atan2(v, u)."""
        return float(np.arctan2(self.v_mean, self.u_mean))

    def at(self, gx, gy):
        return self.u_mean, self.v_mean
