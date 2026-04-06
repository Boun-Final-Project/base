"""
Map generator for RL gas source localization.

6 map templates with per-episode randomization of dimensions, wall positions,
gap widths, source/robot placement, and connectivity validation.
"""

import numpy as np
from collections import deque

from .occupancy_grid import OccupancyGrid
from .. import config as cfg


class MapGenerator:
    """Generates randomized maps from 6 templates."""

    TEMPLATES = [
        "_generate_empty",
        "_generate_single_wall",
        "_generate_u_shape",
        "_generate_three_walls",
        "_generate_complex_maze",
        "_generate_multi_room",
    ]

    def __init__(self, rng=None, width_range=None, height_range=None):
        self.rng = rng or np.random.default_rng()
        self.width_range = width_range or cfg.ROOM_WIDTH_RANGE
        self.height_range = height_range or cfg.ROOM_HEIGHT_RANGE

    def generate(self, template_id=None):
        """Generate a random map.

        Parameters
        ----------
        template_id : int, optional
            Force a specific template (0-5). If None, pick uniformly at random.

        Returns
        -------
        dict with keys: 'grid', 'source_pos', 'robot_pos', 'width', 'height',
                        'template_id'
        """
        if template_id is None:
            template_id = self.rng.integers(0, len(self.TEMPLATES))

        gen_fn = getattr(self, self.TEMPLATES[template_id])
        grid = gen_fn()

        source_pos, robot_pos = self._place_source_and_robot(grid)

        return {
            "grid": grid,
            "source_pos": source_pos,
            "robot_pos": robot_pos,
            "width": grid.width,
            "height": grid.height,
            "template_id": int(template_id),
        }

    # ------------------------------------------------------------------
    # Template generators
    # ------------------------------------------------------------------

    def _generate_empty(self):
        w = self.rng.uniform(*self.width_range)
        h = self.rng.uniform(*self.height_range)
        grid = self._make_grid_with_walls(w, h)
        return grid

    def _generate_single_wall(self):
        w = self.rng.uniform(max(10.0, self.width_range[0]), self.width_range[1])
        h = self.rng.uniform(*self.height_range)
        grid = self._make_grid_with_walls(w, h)

        t = cfg.WALL_THICKNESS
        wall_x = self.rng.uniform(0.3 * w, 0.7 * w)
        wall_len = self.rng.uniform(0.4 * h, 0.8 * h)
        from_top = self.rng.integers(0, 2) == 0

        if from_top:
            # wall from top boundary downward, gap at bottom
            grid.add_rectangular_obstacle(
                wall_x - t / 2, wall_x + t / 2,
                t, t + wall_len,  # start after top boundary wall
            )
        else:
            # wall from bottom boundary upward, gap at top
            grid.add_rectangular_obstacle(
                wall_x - t / 2, wall_x + t / 2,
                h - t - wall_len, h - t,
            )
        return grid

    def _generate_u_shape(self):
        w = self.rng.uniform(max(10.0, self.width_range[0]), self.width_range[1])
        h = self.rng.uniform(max(8.0, self.height_range[0]), self.height_range[1])
        grid = self._make_grid_with_walls(w, h)

        t = cfg.WALL_THICKNESS
        arm_spacing = self.rng.uniform(1.0, 2.5)
        opening = self.rng.choice(["up", "down", "left", "right"])

        # Limit arm length so U fits in the room with margin
        max_arm_h = 0.3 * h - t
        max_arm_w = 0.3 * w - t
        max_arm = min(max_arm_h, max_arm_w, 3.5)
        arm_len = self.rng.uniform(1.5, max(1.5, max_arm))

        # Center of U — ensure it fits with arms inside the room
        margin_x = arm_len + arm_spacing / 2 + t
        margin_y = arm_len + arm_spacing / 2 + t
        cx = self.rng.uniform(margin_x + t, max(margin_x + t + 0.1, w - margin_x - t))
        cy = self.rng.uniform(margin_y + t, max(margin_y + t + 0.1, h - margin_y - t))

        half_s = arm_spacing / 2

        if opening == "up":
            # base (bottom horizontal)
            grid.add_rectangular_obstacle(
                cx - half_s - t / 2, cx + half_s + t / 2,
                cy - t / 2, cy + t / 2,
            )
            # left arm (goes up)
            grid.add_rectangular_obstacle(
                cx - half_s - t / 2, cx - half_s + t / 2,
                cy, cy + arm_len,
            )
            # right arm (goes up)
            grid.add_rectangular_obstacle(
                cx + half_s - t / 2, cx + half_s + t / 2,
                cy, cy + arm_len,
            )
        elif opening == "down":
            grid.add_rectangular_obstacle(
                cx - half_s - t / 2, cx + half_s + t / 2,
                cy - t / 2, cy + t / 2,
            )
            grid.add_rectangular_obstacle(
                cx - half_s - t / 2, cx - half_s + t / 2,
                cy - arm_len, cy,
            )
            grid.add_rectangular_obstacle(
                cx + half_s - t / 2, cx + half_s + t / 2,
                cy - arm_len, cy,
            )
        elif opening == "right":
            grid.add_rectangular_obstacle(
                cx - t / 2, cx + t / 2,
                cy - half_s - t / 2, cy + half_s + t / 2,
            )
            grid.add_rectangular_obstacle(
                cx, cx + arm_len,
                cy - half_s - t / 2, cy - half_s + t / 2,
            )
            grid.add_rectangular_obstacle(
                cx, cx + arm_len,
                cy + half_s - t / 2, cy + half_s + t / 2,
            )
        else:  # left
            grid.add_rectangular_obstacle(
                cx - t / 2, cx + t / 2,
                cy - half_s - t / 2, cy + half_s + t / 2,
            )
            grid.add_rectangular_obstacle(
                cx - arm_len, cx,
                cy - half_s - t / 2, cy - half_s + t / 2,
            )
            grid.add_rectangular_obstacle(
                cx - arm_len, cx,
                cy + half_s - t / 2, cy + half_s + t / 2,
            )
        return grid

    def _generate_three_walls(self):
        w = self.rng.uniform(max(12.0, self.width_range[0]), max(12.0, self.width_range[1]))
        h = self.rng.uniform(max(8.0, self.height_range[0]), max(8.0, self.height_range[1]))
        grid = self._make_grid_with_walls(w, h)

        t = cfg.WALL_THICKNESS
        gap = cfg.MIN_GAP_SIZE

        # Two vertical walls from top, gaps at bottom
        wall1_x = self.rng.uniform(0.2 * w, 0.4 * w)
        wall2_x = self.rng.uniform(0.6 * w, 0.8 * w)
        wall1_len = self.rng.uniform(0.5 * h, 0.8 * h)
        wall2_len = self.rng.uniform(0.5 * h, 0.8 * h)

        wall1_top = h - t
        wall1_bot = wall1_top - wall1_len
        wall2_top = h - t
        wall2_bot = wall2_top - wall2_len

        grid.add_rectangular_obstacle(
            wall1_x - t / 2, wall1_x + t / 2,
            wall1_bot, wall1_top,
        )
        grid.add_rectangular_obstacle(
            wall2_x - t / 2, wall2_x + t / 2,
            wall2_bot, wall2_top,
        )

        # Central block between the two vertical walls, below their bottoms
        # so it doesn't merge with them
        block_half = self.rng.uniform(0.5, 1.0)
        block_x_lo = wall1_x + t / 2 + gap + block_half
        block_x_hi = wall2_x - t / 2 - gap - block_half
        block_y_lo = t + gap + block_half
        block_y_hi = min(wall1_bot, wall2_bot) - gap - block_half

        if block_x_hi > block_x_lo and block_y_hi > block_y_lo:
            block_cx = self.rng.uniform(block_x_lo, block_x_hi)
            block_cy = self.rng.uniform(block_y_lo, block_y_hi)
            grid.add_rectangular_obstacle(
                block_cx - block_half, block_cx + block_half,
                block_cy - block_half, block_cy + block_half,
            )
        return grid

    def _generate_complex_maze(self):
        w = self.rng.uniform(max(12.0, self.width_range[0]), max(12.0, self.width_range[1]))
        h = self.rng.uniform(max(8.0, self.height_range[0]), max(8.0, self.height_range[1]))
        grid = self._make_grid_with_walls(w, h)

        t = cfg.WALL_THICKNESS
        gap = cfg.MIN_GAP_SIZE

        # Two vertical walls from top, gaps at bottom
        wall1_x = self.rng.uniform(0.2 * w, 0.4 * w)
        wall2_x = self.rng.uniform(0.6 * w, 0.8 * w)
        wall1_len = self.rng.uniform(0.5 * h, 0.75 * h)
        wall2_len = self.rng.uniform(0.5 * h, 0.75 * h)

        # Vertical wall tops (they start from the top boundary)
        wall1_top = h - t
        wall1_bot = wall1_top - wall1_len
        wall2_top = h - t
        wall2_bot = wall2_top - wall2_len

        grid.add_rectangular_obstacle(
            wall1_x - t / 2, wall1_x + t / 2,
            wall1_bot, wall1_top,
        )
        grid.add_rectangular_obstacle(
            wall2_x - t / 2, wall2_x + t / 2,
            wall2_bot, wall2_top,
        )

        # 1-2 horizontal walls in the lower half
        # Each horizontal wall runs from a side boundary but stops before
        # reaching any vertical wall, leaving a gap for passage
        n_horiz = self.rng.integers(1, 3)
        for _ in range(n_horiz):
            from_left = self.rng.integers(0, 2) == 0
            hw_y = self.rng.uniform(0.5 * h, 0.85 * h)

            if from_left:
                # Runs from left boundary, stops before wall1 (the first obstacle)
                max_len = wall1_x - t / 2 - t - gap
                if max_len < 1.0:
                    continue
                hw_len = self.rng.uniform(1.0, max_len)
                grid.add_rectangular_obstacle(
                    t, t + hw_len,
                    hw_y - t / 2, hw_y + t / 2,
                )
            else:
                # Runs from right boundary, stops before wall2
                max_len = w - t - (wall2_x + t / 2) - gap
                if max_len < 1.0:
                    continue
                hw_len = self.rng.uniform(1.0, max_len)
                grid.add_rectangular_obstacle(
                    w - t - hw_len, w - t,
                    hw_y - t / 2, hw_y + t / 2,
                )

        # Central block between the two vertical walls, below the wall bottoms
        # so it doesn't merge with them
        block_half = self.rng.uniform(0.5, 1.0)
        block_x_lo = wall1_x + t / 2 + gap + block_half
        block_x_hi = wall2_x - t / 2 - gap - block_half
        block_y_lo = t + gap + block_half
        block_y_hi = min(wall1_bot, wall2_bot) - gap - block_half

        if block_x_hi > block_x_lo and block_y_hi > block_y_lo:
            block_cx = self.rng.uniform(block_x_lo, block_x_hi)
            block_cy = self.rng.uniform(block_y_lo, block_y_hi)
            grid.add_rectangular_obstacle(
                block_cx - block_half, block_cx + block_half,
                block_cy - block_half, block_cy + block_half,
            )
        return grid

    def _generate_multi_room(self):
        # Near-square footprint
        base = self.rng.uniform(max(12.0, self.width_range[0]), max(12.0, self.width_range[1]))
        aspect = self.rng.uniform(0.8, 1.2)
        w = base
        h = base * aspect
        grid = self._make_grid_with_walls(w, h)

        t = cfg.WALL_THICKNESS
        cw = self.rng.uniform(1.5, 3.0)  # corridor width

        cx = w / 2
        cy = h / 2

        # Key coordinates for the corridor edges
        vl = cx - cw / 2          # vertical corridor left edge
        vr = cx + cw / 2          # vertical corridor right edge
        hb = cy - cw / 2          # horizontal corridor bottom edge
        ht = cy + cw / 2          # horizontal corridor top edge

        # 8 wall segments — each stops at the corridor intersection
        # Left vertical wall: bottom segment (bottom-left room right side)
        grid.add_rectangular_obstacle(vl - t, vl, t, hb)
        # Left vertical wall: top segment (top-left room right side)
        grid.add_rectangular_obstacle(vl - t, vl, ht, h - t)
        # Right vertical wall: bottom segment (bottom-right room left side)
        grid.add_rectangular_obstacle(vr, vr + t, t, hb)
        # Right vertical wall: top segment (top-right room left side)
        grid.add_rectangular_obstacle(vr, vr + t, ht, h - t)
        # Bottom horizontal wall: left segment (bottom-left room top side)
        grid.add_rectangular_obstacle(t, vl, hb - t, hb)
        # Bottom horizontal wall: right segment (bottom-right room top side)
        grid.add_rectangular_obstacle(vr, w - t, hb - t, hb)
        # Top horizontal wall: left segment (top-left room bottom side)
        grid.add_rectangular_obstacle(t, vl, ht, ht + t)
        # Top horizontal wall: right segment (top-right room bottom side)
        grid.add_rectangular_obstacle(vr, w - t, ht, ht + t)

        # Cut one doorway per wall segment (8 doorways total)
        # Each room gets 2 doorways (one into vertical corridor, one into horizontal)

        # Helper: cut a doorway in a vertical wall segment
        def _vdoor(wall_x_min, wall_x_max, seg_y_min, seg_y_max):
            dw = self.rng.uniform(cfg.MIN_GAP_SIZE, min(2.0, seg_y_max - seg_y_min - 0.2))
            dy = self.rng.uniform(seg_y_min + 0.1, seg_y_max - dw - 0.1)
            self._clear_rect(grid, wall_x_min, wall_x_max, dy, dy + dw)

        # Helper: cut a doorway in a horizontal wall segment
        def _hdoor(seg_x_min, seg_x_max, wall_y_min, wall_y_max):
            dw = self.rng.uniform(cfg.MIN_GAP_SIZE, min(2.0, seg_x_max - seg_x_min - 0.2))
            dx = self.rng.uniform(seg_x_min + 0.1, seg_x_max - dw - 0.1)
            self._clear_rect(grid, dx, dx + dw, wall_y_min, wall_y_max)

        # Bottom-left room doorways
        _vdoor(vl - t, vl, t, hb)            # into vertical corridor
        _hdoor(t, vl, hb - t, hb)            # into horizontal corridor

        # Bottom-right room doorways
        _vdoor(vr, vr + t, t, hb)            # into vertical corridor
        _hdoor(vr, w - t, hb - t, hb)        # into horizontal corridor

        # Top-left room doorways
        _vdoor(vl - t, vl, ht, h - t)        # into vertical corridor
        _hdoor(t, vl, ht, ht + t)            # into horizontal corridor

        # Top-right room doorways
        _vdoor(vr, vr + t, ht, h - t)        # into vertical corridor
        _hdoor(vr, w - t, ht, ht + t)        # into horizontal corridor

        return grid

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_grid_with_walls(self, width, height):
        """Create a grid with boundary walls."""
        # Snap to resolution
        width = round(width / cfg.GRID_RESOLUTION) * cfg.GRID_RESOLUTION
        height = round(height / cfg.GRID_RESOLUTION) * cfg.GRID_RESOLUTION
        grid = OccupancyGrid(width, height, cfg.GRID_RESOLUTION)

        t = cfg.WALL_THICKNESS
        # Top, bottom, left, right boundary walls
        grid.add_rectangular_obstacle(0, width, 0, t)          # bottom
        grid.add_rectangular_obstacle(0, width, height - t, height)  # top
        grid.add_rectangular_obstacle(0, t, 0, height)          # left
        grid.add_rectangular_obstacle(width - t, width, 0, height)   # right
        return grid

    def _clear_rect(self, grid, x_min, x_max, y_min, y_max):
        """Clear a rectangular region (set to 0) — used for doorways."""
        gx_min, gy_min = grid.world_to_grid(x_min, y_min)
        gx_max, gy_max = grid.world_to_grid(x_max, y_max)
        gx_min = max(0, gx_min)
        gx_max = min(grid.grid_width - 1, gx_max)
        gy_min = max(0, gy_min)
        gy_max = min(grid.grid_height - 1, gy_max)
        grid.grid[gy_min:gy_max + 1, gx_min:gx_max + 1] = 0

    def _place_source_and_robot(self, grid, max_retries=200):
        """Place source and robot in valid, connected positions."""
        free_cells = self._get_free_cells(grid)
        if len(free_cells) < 2:
            raise RuntimeError("Map has fewer than 2 free cells")

        for _ in range(max_retries):
            idx_s = self.rng.integers(0, len(free_cells))
            source = free_cells[idx_s]

            # Filter candidates far enough from source
            dists = np.linalg.norm(free_cells - source, axis=1)
            far_enough = np.where(dists >= cfg.MIN_SOURCE_ROBOT_DIST)[0]
            if len(far_enough) == 0:
                continue

            idx_r = self.rng.choice(far_enough)
            robot = free_cells[idx_r]

            if self._are_connected(grid, source, robot):
                return tuple(source), tuple(robot)

        # Fallback: pick any connected pair
        idx_s = self.rng.integers(0, len(free_cells))
        source = free_cells[idx_s]
        reachable = self._bfs_reachable(grid, source)
        reachable_arr = np.array(list(reachable))
        if len(reachable_arr) == 0:
            raise RuntimeError("No reachable cells from source")
        dists = np.linalg.norm(reachable_arr - source, axis=1)
        best = np.argmax(dists)
        return tuple(source), tuple(reachable_arr[best])

    def _get_free_cells(self, grid):
        """Get world-coordinate centers of all cells valid for robot placement."""
        res = cfg.GRID_RESOLUTION
        radius = cfg.ROBOT_RADIUS
        cells = []
        # Sample on a coarser grid for speed (every 5 cells = 0.5m)
        step = max(1, int(0.5 / res))
        for gy in range(0, grid.grid_height, step):
            for gx in range(0, grid.grid_width, step):
                if grid.is_valid(gx=gx, gy=gy, radius=radius):
                    x = (gx + 0.5) * res
                    y = (gy + 0.5) * res
                    cells.append((x, y))
        return np.array(cells)

    def _are_connected(self, grid, pos_a, pos_b):
        """Check if two world positions are connected via BFS on the grid."""
        res = cfg.GRID_RESOLUTION
        step = max(1, int(0.5 / res))  # same coarse step as _get_free_cells

        start = (int(pos_a[0] / res) // step * step,
                 int(pos_a[1] / res) // step * step)
        goal = (int(pos_b[0] / res) // step * step,
                int(pos_b[1] / res) // step * step)

        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            gx, gy = queue.popleft()
            if (gx, gy) == goal:
                return True
            for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                nx, ny = gx + dx, gy + dy
                if (nx, ny) not in visited and grid.is_valid(gx=nx, gy=ny, radius=cfg.ROBOT_RADIUS):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def _bfs_reachable(self, grid, pos):
        """Return all world positions reachable from pos."""
        res = cfg.GRID_RESOLUTION
        step = max(1, int(0.5 / res))

        start = (int(pos[0] / res) // step * step,
                 int(pos[1] / res) // step * step)

        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            gx, gy = queue.popleft()
            for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                nx, ny = gx + dx, gy + dy
                if (nx, ny) not in visited and grid.is_valid(gx=nx, gy=ny, radius=cfg.ROBOT_RADIUS):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        reachable = []
        for gx, gy in visited:
            x = (gx + 0.5) * res
            y = (gy + 0.5) * res
            reachable.append((x, y))
        return reachable
