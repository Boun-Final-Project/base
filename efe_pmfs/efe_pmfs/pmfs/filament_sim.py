"""
2D filament dispersion simulator for PMFS.

Port of Simulations::simulateSourceInPosition + moveAlongPath + moveFilament
(gsl_server/.../PMFS/internal/Simulations.cpp, lines 271-452).

Each call to `simulate_source(source_cell, ...)` runs:
  1. Warmup phase: release filaments from the source, drift them with
     `velocity = wind + Gaussian_noise`. Stop when either min-iterations
     reached and a filament has exited bounds, or max-iterations hit.
  2. Recording phase: release 5 new filaments per step for N steps, accumulate
     a per-cell hit count (incrementing at most once per timestep per cell).
     Return per-cell relative frequency in [0, 1].

Key compromises vs. the full C++:
  - Collision via step-wise DDA along the per-step movement (matches the
    USE_DDA=0 fallback path in the C++).
  - Treats cells marked as UNKNOWN (-1) as traversable (see plan §Decisions):
    filaments can pass through not-yet-explored regions so predictions aren't
    artificially clipped at frontiers.
  - Filaments exit through the grid boundary only; walls stop a step.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numba import njit


CELL_FREE = 0
CELL_UNKNOWN = -1
CELL_OCCUPIED = 1


@dataclass
class FilamentSettings:
    noise_stddev: float = 0.2
    delta_time: float = 0.1
    min_warmup_iterations: int = 20
    max_warmup_iterations: int = 300
    iterations_to_record: int = 100
    filaments_per_step: int = 5


@njit(cache=True)
def _cell_is_traversable(occupancy, gx, gy):
    h, w = occupancy.shape
    if gx < 0 or gx >= w or gy < 0 or gy >= h:
        return False
    # Free OR unknown is traversable. Only wall (occupied) blocks.
    return occupancy[gy, gx] != CELL_OCCUPIED


@njit(cache=True)
def _move_along_path(occupancy, origin_x, origin_y, resolution,
                     x_world, y_world, dx_world, dy_world):
    """
    Advance filament from (x_world,y_world) by (dx,dy) with step-wise collision.
    Returns new (x, y) and whether the filament is still inside bounds.
    """
    travel = np.sqrt(dx_world * dx_world + dy_world * dy_world)
    if travel < 1e-9:
        return x_world, y_world, True

    step_size = min(travel, resolution * 0.5)
    nx = dx_world / travel
    ny = dy_world / travel
    inc_x = nx * step_size
    inc_y = ny * step_size
    steps = int(travel / step_size)

    h, w = occupancy.shape
    cur_x = x_world
    cur_y = y_world

    for _ in range(steps):
        cur_x += inc_x
        cur_y += inc_y
        gx = int(np.floor((cur_x - origin_x) / resolution))
        gy = int(np.floor((cur_y - origin_y) / resolution))
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            # out of bounds → mark as exited
            return cur_x, cur_y, False
        if occupancy[gy, gx] == CELL_OCCUPIED:
            # hit wall → back up and stop moving further this step
            cur_x -= inc_x
            cur_y -= inc_y
            return cur_x, cur_y, True
    return cur_x, cur_y, True


@njit(cache=True)
def _simulate_source_numba(occupancy, wind_u, wind_v,
                           origin_x, origin_y, resolution,
                           source_x, source_y,
                           noise_stddev, delta_time,
                           min_warmup, max_warmup, iterations_record,
                           filaments_per_step):
    """
    Returns: hit_count[h,w] as float32, filled with relative frequency in [0,1].
    """
    h, w = occupancy.shape
    hit_count = np.zeros((h, w), dtype=np.float32)
    updated = np.zeros((h, w), dtype=np.int32)  # last timestep each cell was hit

    # Pre-allocate filament storage. Bound: (max_warmup + iterations_record) * filaments_per_step
    max_filaments = (max_warmup + iterations_record + 2) * filaments_per_step
    fx = np.empty(max_filaments, dtype=np.float32)
    fy = np.empty(max_filaments, dtype=np.float32)
    n_active = 0

    # ---- Warmup ----
    stable = False
    warmup_dt = delta_time * 2.0  # matches PMFS C++ line 324
    iter_count = 0
    while iter_count < min_warmup or (not stable and iter_count < max_warmup):
        # Release new filaments
        for _ in range(filaments_per_step):
            if n_active < max_filaments:
                fx[n_active] = source_x
                fy[n_active] = source_y
                n_active += 1

        # Move all filaments; compact in place.
        write = 0
        for r in range(n_active):
            x = fx[r]
            y = fy[r]
            gx = int(np.floor((x - origin_x) / resolution))
            gy = int(np.floor((y - origin_y) / resolution))
            if gx < 0 or gx >= w or gy < 0 or gy >= h:
                stable = True
                continue
            u = wind_u[gy, gx] + np.random.normal(0.0, noise_stddev)
            v = wind_v[gy, gx] + np.random.normal(0.0, noise_stddev)
            x2, y2, inside = _move_along_path(
                occupancy, origin_x, origin_y, resolution,
                x, y, warmup_dt * u, warmup_dt * v,
            )
            if not inside:
                stable = True
                continue
            fx[write] = x2
            fy[write] = y2
            write += 1
        n_active = write
        iter_count += 1

    # ---- Recording ----
    for t in range(1, iterations_record + 1):
        for _ in range(filaments_per_step):
            if n_active < max_filaments:
                fx[n_active] = source_x
                fy[n_active] = source_y
                n_active += 1

        write = 0
        for r in range(n_active):
            x = fx[r]
            y = fy[r]
            gx = int(np.floor((x - origin_x) / resolution))
            gy = int(np.floor((y - origin_y) / resolution))
            if gx < 0 or gx >= w or gy < 0 or gy >= h:
                continue
            # record hit (once per cell per timestep)
            if updated[gy, gx] < t:
                hit_count[gy, gx] += 1.0
                updated[gy, gx] = t
            # advect
            u = wind_u[gy, gx] + np.random.normal(0.0, noise_stddev)
            v = wind_v[gy, gx] + np.random.normal(0.0, noise_stddev)
            x2, y2, inside = _move_along_path(
                occupancy, origin_x, origin_y, resolution,
                x, y, delta_time * u, delta_time * v,
            )
            if not inside:
                continue
            fx[write] = x2
            fy[write] = y2
            write += 1
        n_active = write

    # Normalize: hit_count[i] / iterations_record
    inv = 1.0 / float(iterations_record)
    for gy in range(h):
        for gx in range(w):
            hit_count[gy, gx] *= inv
    return hit_count


class FilamentSimulator:
    def __init__(self, occupancy_grid, wind_field, settings: FilamentSettings):
        """
        Parameters
        ----------
        occupancy_grid : OccupancyGridMap  (for origin/resolution)
        wind_field : ConstantWindField
        settings : FilamentSettings
        """
        self.grid = occupancy_grid
        self.wind = wind_field
        self.settings = settings

    def simulate_source(self, source_cell, occupancy_array):
        """
        Parameters
        ----------
        source_cell : (gx, gy) candidate source position
        occupancy_array : np.int8[H, W] current SLAM grid snapshot
        """
        gx, gy = int(source_cell[0]), int(source_cell[1])
        sx, sy = self.grid.grid_to_world(gx, gy)
        s = self.settings
        return _simulate_source_numba(
            occupancy_array, self.wind.u, self.wind.v,
            self.grid.origin_x, self.grid.origin_y, self.grid.resolution,
            float(sx), float(sy),
            s.noise_stddev, s.delta_time,
            s.min_warmup_iterations, s.max_warmup_iterations, s.iterations_to_record,
            s.filaments_per_step,
        )
