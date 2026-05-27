"""Heuristic inlet/outlet placement on procedural maps.

Constraints:
- Inlets on one wall, outlets on opposite wall (W↔E or S↔N)
- Each opening 1.0-2.5 m wide
- Openings on the same wall ≥ 1 m apart
- Each opening must connect to the map's main interior via flood-fill
  (filters out "sealed-off room" cases)
- Inlet-to-outlet euclidean distance ≥ 0.5 × map diagonal
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RL_PACKAGE_PATH
sys.path.insert(0, RL_PACKAGE_PATH)
from reinforcement_learning.envs.map_generator import MapGenerator


@dataclass
class OpeningSpec:
    side: str            # 'west', 'east', 'south', 'north'
    lo: float            # m
    hi: float            # m
    role: str            # 'inlet' or 'outlet'

    def center(self, map_w, map_h):
        mid = 0.5 * (self.lo + self.hi)
        if self.side == 'west':  return (0.0, mid)
        if self.side == 'east':  return (map_w, mid)
        if self.side == 'south': return (mid, 0.0)
        if self.side == 'north': return (mid, map_h)


def _clearance_zone_free(grid, side, lo, hi, cell_size, wall_thick_cells,
                          near_depth_m=0.5, far_depth_m=1.5,
                          near_min_free=1.0, far_min_free=0.85):
    """Two-zone check inside the opening:
    - Near zone (depth 0 to near_depth_m, full opening width): MUST be 100% free.
      This kills "wall directly behind the opening" cases.
    - Far zone (near_depth_m to far_depth_m): >= far_min_free free.
      Allows some interior walls deeper in.
    """
    H, W = grid.shape
    near_d = max(1, int(round(near_depth_m / cell_size)))
    far_d = max(near_d + 1, int(round(far_depth_m / cell_size)))
    if side == 'west':
        r_lo, r_hi = int(lo / cell_size), int(np.ceil(hi / cell_size))
        c_near = (wall_thick_cells, wall_thick_cells + near_d)
        c_far = (wall_thick_cells + near_d, wall_thick_cells + far_d)
    elif side == 'east':
        r_lo, r_hi = int(lo / cell_size), int(np.ceil(hi / cell_size))
        c_near = (W - wall_thick_cells - near_d, W - wall_thick_cells)
        c_far = (W - wall_thick_cells - far_d, W - wall_thick_cells - near_d)
    elif side == 'south':
        c_lo, c_hi = int(lo / cell_size), int(np.ceil(hi / cell_size))
        r_near = (wall_thick_cells, wall_thick_cells + near_d)
        r_far = (wall_thick_cells + near_d, wall_thick_cells + far_d)
    else:  # north
        c_lo, c_hi = int(lo / cell_size), int(np.ceil(hi / cell_size))
        r_near = (H - wall_thick_cells - near_d, H - wall_thick_cells)
        r_far = (H - wall_thick_cells - far_d, H - wall_thick_cells - near_d)
    if side in ('west', 'east'):
        near = grid[r_lo:r_hi, c_near[0]:c_near[1]]
        far = grid[r_lo:r_hi, c_far[0]:c_far[1]]
    else:
        near = grid[r_near[0]:r_near[1], c_lo:c_hi]
        far = grid[r_far[0]:r_far[1], c_lo:c_hi]
    if near.size == 0 or far.size == 0: return False
    if (near == 0).mean() < near_min_free: return False
    if (far == 0).mean() < far_min_free: return False
    return True


def _flood_fill_from_edge(grid, side, lo, hi, cell_size, wall_thick_cells):
    """Try to fill from the opening cells; return # of reached free cells."""
    H, W = grid.shape
    work = grid.copy()
    # Punch the opening
    if side == 'west':
        r_lo, r_hi = int(lo / cell_size), int(np.ceil(hi / cell_size))
        work[r_lo:r_hi, :wall_thick_cells] = 0
    elif side == 'east':
        r_lo, r_hi = int(lo / cell_size), int(np.ceil(hi / cell_size))
        work[r_lo:r_hi, -wall_thick_cells:] = 0
    elif side == 'south':
        c_lo, c_hi = int(lo / cell_size), int(np.ceil(hi / cell_size))
        work[:wall_thick_cells, c_lo:c_hi] = 0
    elif side == 'north':
        c_lo, c_hi = int(lo / cell_size), int(np.ceil(hi / cell_size))
        work[-wall_thick_cells:, c_lo:c_hi] = 0
    free = (work == 0)
    # BFS from the opening cells
    if side == 'west':   seeds = [(r, 0) for r in range(r_lo, r_hi) if free[r, 0]]
    elif side == 'east': seeds = [(r, W-1) for r in range(r_lo, r_hi) if free[r, W-1]]
    elif side == 'south':seeds = [(0, c) for c in range(c_lo, c_hi) if free[0, c]]
    else:                seeds = [(H-1, c) for c in range(c_lo, c_hi) if free[H-1, c]]
    if not seeds:
        return 0
    visited = np.zeros_like(free, dtype=bool)
    from collections import deque
    q = deque(seeds)
    for r, c in seeds: visited[r, c] = True
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and free[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    return int(visited.sum())


def _try_place_opening(grid, side, map_dim, cell_size, wall_thick_cells,
                       width, existing, rng, min_separation=1.0,
                       min_reach_cells=200, max_attempts=30):
    """Try to find one valid opening on a given side. Returns OpeningSpec or None."""
    edge = map_dim  # length of this wall in meters (W for N/S, H for W/E)
    for _ in range(max_attempts):
        lo = float(rng.uniform(0.2, edge - width - 0.2))
        hi = lo + width
        # Check spacing from existing openings on same side
        overlap = False
        for o in existing:
            if o.side == side:
                # gap test
                if not (hi + min_separation < o.lo or lo > o.hi + min_separation):
                    overlap = True; break
        if overlap: continue
        # Check clearance zone (no walls right behind opening)
        if not _clearance_zone_free(grid, side, lo, hi, cell_size, wall_thick_cells):
            continue
        # Check connectivity
        reach = _flood_fill_from_edge(grid, side, lo, hi, cell_size, wall_thick_cells)
        if reach < min_reach_cells: continue
        return (lo, hi)
    return None


def sample_openings(grid, cell_size, map_w, map_h, rng,
                    wall_thick_m=0.6,
                    n_inlets=None, n_outlets=None,
                    inlet_side=None,
                    min_width=1.0, max_width=2.5):
    """Sample inlet/outlet openings for one map. Returns list of OpeningSpec
    or None if no valid config found."""
    wall_thick_cells = max(1, int(round(wall_thick_m / cell_size)))
    if n_inlets is None:
        n_inlets = int(rng.choice([1, 2, 3], p=[0.2, 0.6, 0.2]))
    if n_outlets is None:
        n_outlets = int(rng.choice([1, 2, 3], p=[0.2, 0.6, 0.2]))
    if inlet_side is None:
        inlet_side = rng.choice(['west', 'south'], p=[0.7, 0.3])
    opposites = {'west': 'east', 'south': 'north'}
    outlet_side = opposites[inlet_side]

    openings = []
    # Place inlets
    inlet_wall_dim = map_h if inlet_side in ('west', 'east') else map_w
    for _ in range(n_inlets):
        w = float(rng.uniform(min_width, max_width))
        res = _try_place_opening(grid, inlet_side, inlet_wall_dim, cell_size,
                                  wall_thick_cells, w, openings, rng)
        if res is None: continue
        openings.append(OpeningSpec(inlet_side, res[0], res[1], 'inlet'))
    if not openings: return None
    # Place outlets
    outlet_wall_dim = map_h if outlet_side in ('west', 'east') else map_w
    for _ in range(n_outlets):
        w = float(rng.uniform(min_width, max_width))
        res = _try_place_opening(grid, outlet_side, outlet_wall_dim, cell_size,
                                  wall_thick_cells, w, openings, rng)
        if res is None: continue
        openings.append(OpeningSpec(outlet_side, res[0], res[1], 'outlet'))
    if not any(o.role == 'outlet' for o in openings): return None
    # Min separation between any inlet and outlet
    diag = (map_w**2 + map_h**2) ** 0.5
    min_sep = 0.5 * diag
    for i in [o for o in openings if o.role == 'inlet']:
        for j in [o for o in openings if o.role == 'outlet']:
            ic = i.center(map_w, map_h); jc = j.center(map_w, map_h)
            d = ((ic[0]-jc[0])**2 + (ic[1]-jc[1])**2) ** 0.5
            if d < min_sep:
                return None
    return openings


def sample_map_with_openings(template_id, seed, max_retries=10):
    """Generate one map + valid openings. Returns (map_data, openings) or (None, None)."""
    rng = np.random.default_rng(seed)
    gen_rng = np.random.default_rng(seed)
    gen = MapGenerator(rng=gen_rng)
    map_data = gen.generate(template_id=template_id)
    grid = np.array(map_data['grid'].grid, dtype=np.int8)
    cell_size = map_data['grid'].resolution
    map_w = map_data['width']; map_h = map_data['height']
    for attempt in range(max_retries):
        ops = sample_openings(grid, cell_size, map_w, map_h, rng)
        if ops is not None:
            return map_data, ops
    return None, None
