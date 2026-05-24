"""
Map generator for RL gas source localization.

10 map templates with per-episode randomization of dimensions, wall positions,
gap widths, source/robot placement, and connectivity validation.
"""

import numpy as np
from collections import deque

from .occupancy_grid import OccupancyGrid
from .. import config as cfg


class MapGenerator:
    """Generates randomized maps from 10 templates."""

    TEMPLATES = [
        "_generate_empty",
        "_generate_single_wall",
        "_generate_u_shape",
        "_generate_three_walls",
        "_generate_complex_maze",
        "_generate_multi_room",
        "_generate_dead_end_corridor",
        "_generate_serpentine_corridor",
        "_generate_dense_multi_room",
        "_generate_hybrid",
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
            Force a specific template (0-9). If None, pick uniformly at random.

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
                h - t - wall_len, h - t,
            )
        else:
            # wall from bottom boundary upward, gap at top
            grid.add_rectangular_obstacle(
                wall_x - t / 2, wall_x + t / 2,
                t, t + wall_len,
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

        # Two vertical walls — each independently anchored at top OR bottom
        wall1_x = self.rng.uniform(0.2 * w, 0.4 * w)
        wall2_x = self.rng.uniform(0.6 * w, 0.8 * w)
        wall1_len = self.rng.uniform(0.5 * h, 0.8 * h)
        wall2_len = self.rng.uniform(0.5 * h, 0.8 * h)
        wall1_from_top = self.rng.integers(0, 2) == 0
        wall2_from_top = self.rng.integers(0, 2) == 0

        if wall1_from_top:
            wall1_top, wall1_bot = h - t, h - t - wall1_len
        else:
            wall1_bot, wall1_top = t, t + wall1_len
        if wall2_from_top:
            wall2_top, wall2_bot = h - t, h - t - wall2_len
        else:
            wall2_bot, wall2_top = t, t + wall2_len

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
        # Open vertical band = intersection of the open sides of both walls
        open1_lo = wall1_top if not wall1_from_top else t
        open1_hi = wall1_bot if wall1_from_top else h - t
        open2_lo = wall2_top if not wall2_from_top else t
        open2_hi = wall2_bot if wall2_from_top else h - t
        block_y_lo = max(open1_lo, open2_lo) + gap + block_half
        block_y_hi = min(open1_hi, open2_hi) - gap - block_half

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

        wall1_x = self.rng.uniform(0.2 * w, 0.4 * w)
        wall2_x = self.rng.uniform(0.6 * w, 0.8 * w)
        wall1_len = self.rng.uniform(0.5 * h, 0.75 * h)
        wall2_len = self.rng.uniform(0.5 * h, 0.75 * h)
        wall1_from_top = self.rng.integers(0, 2) == 0
        wall2_from_top = self.rng.integers(0, 2) == 0

        if wall1_from_top:
            wall1_top, wall1_bot = h - t, h - t - wall1_len
        else:
            wall1_bot, wall1_top = t, t + wall1_len
        if wall2_from_top:
            wall2_top, wall2_bot = h - t, h - t - wall2_len
        else:
            wall2_bot, wall2_top = t, t + wall2_len

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
            hw_y = self.rng.uniform(0.15 * h, 0.85 * h)

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
        open1_lo = wall1_top if not wall1_from_top else t
        open1_hi = wall1_bot if wall1_from_top else h - t
        open2_lo = wall2_top if not wall2_from_top else t
        open2_hi = wall2_bot if wall2_from_top else h - t
        block_y_lo = max(open1_lo, open2_lo) + gap + block_half
        block_y_hi = min(open1_hi, open2_hi) - gap - block_half

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
            max_dw = min(2.0, seg_y_max - seg_y_min - 0.2)
            if max_dw < cfg.MIN_GAP_SIZE:
                return  # segment too short to fit a doorway
            dw = self.rng.uniform(cfg.MIN_GAP_SIZE, max_dw)
            max_dy = seg_y_max - dw - 0.1
            if max_dy < seg_y_min + 0.1:
                return
            dy = self.rng.uniform(seg_y_min + 0.1, max_dy)
            self._clear_rect(grid, wall_x_min, wall_x_max, dy, dy + dw)

        # Helper: cut a doorway in a horizontal wall segment
        def _hdoor(seg_x_min, seg_x_max, wall_y_min, wall_y_max):
            max_dw = min(2.0, seg_x_max - seg_x_min - 0.2)
            if max_dw < cfg.MIN_GAP_SIZE:
                return  # segment too short to fit a doorway
            dw = self.rng.uniform(cfg.MIN_GAP_SIZE, max_dw)
            max_dx = seg_x_max - dw - 0.1
            if max_dx < seg_x_min + 0.1:
                return
            dx = self.rng.uniform(seg_x_min + 0.1, max_dx)
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

    def _generate_dead_end_corridor(self):
        """Open room with 2-3 dead-end corridor branches.

        Each branch is drawn as a "C" — three walls forming a closed-end
        corridor with one open mouth. Source and robot are placed by
        BFS-connected sampling, so sometimes the source is inside a
        cul-de-sac and sometimes the agent must reject empty cul-de-sacs.
        Targets uleft/uright "dead-end re-entry" failure mode.
        """
        w = self.rng.uniform(max(14.0, self.width_range[0]),
                             max(14.0, self.width_range[1]))
        h = self.rng.uniform(max(8.0, self.height_range[0]),
                             max(8.0, self.height_range[1]))
        grid = self._make_grid_with_walls(w, h)

        t = cfg.WALL_THICKNESS
        n_branches = int(self.rng.integers(2, 4))
        placed = []  # list of outer-bbox tuples (x_min, x_max, y_min, y_max)

        for _ in range(n_branches):
            for _attempt in range(15):
                cl = self.rng.uniform(2.5, 5.0)        # corridor length (interior)
                cw = self.rng.uniform(1.4, 2.2)        # corridor width (interior)
                opening = self.rng.choice(["up", "down", "left", "right"])

                # Outer bbox dimensions (including wall thickness on closed sides)
                if opening in ("up", "down"):
                    ow, oh = cw + 2 * t, cl + t
                else:
                    ow, oh = cl + t, cw + 2 * t

                margin = 0.4
                if w - 2 * t - 2 * margin < ow + 0.1 or h - 2 * t - 2 * margin < oh + 0.1:
                    continue
                x_min = self.rng.uniform(t + margin, w - t - ow - margin)
                y_min = self.rng.uniform(t + margin, h - t - oh - margin)
                x_max = x_min + ow
                y_max = y_min + oh

                # Reject overlap with previously placed branches (with clearance)
                clearance = cfg.MIN_GAP_SIZE
                bbox = (x_min - clearance, x_max + clearance,
                        y_min - clearance, y_max + clearance)
                if any(not (bbox[1] < b[0] or bbox[0] > b[1]
                            or bbox[3] < b[2] or bbox[2] > b[3])
                       for b in placed):
                    continue
                placed.append((x_min, x_max, y_min, y_max))

                # Draw 3-wall "C" with opening on side `opening`.
                if opening == "up":
                    # mouth at top → cap at bottom, sides on left/right
                    grid.add_rectangular_obstacle(x_min, x_max, y_min, y_min + t)  # cap
                    grid.add_rectangular_obstacle(x_min, x_min + t, y_min, y_max)  # left
                    grid.add_rectangular_obstacle(x_max - t, x_max, y_min, y_max)  # right
                elif opening == "down":
                    grid.add_rectangular_obstacle(x_min, x_max, y_max - t, y_max)  # cap
                    grid.add_rectangular_obstacle(x_min, x_min + t, y_min, y_max)  # left
                    grid.add_rectangular_obstacle(x_max - t, x_max, y_min, y_max)  # right
                elif opening == "left":
                    grid.add_rectangular_obstacle(x_max - t, x_max, y_min, y_max)  # cap
                    grid.add_rectangular_obstacle(x_min, x_max, y_min, y_min + t)  # bottom
                    grid.add_rectangular_obstacle(x_min, x_max, y_max - t, y_max)  # top
                else:  # right
                    grid.add_rectangular_obstacle(x_min, x_min + t, y_min, y_max)  # cap
                    grid.add_rectangular_obstacle(x_min, x_max, y_min, y_min + t)  # bottom
                    grid.add_rectangular_obstacle(x_min, x_max, y_max - t, y_max)  # top
                break  # successfully placed this branch
        return grid

    def _generate_serpentine_corridor(self):
        """S-shaped corridor carved through a wall-filled room.

        Builds 3-5 horizontal corridors at evenly spaced y-levels,
        connected by vertical jogs at alternating left/right ends.
        Carving (via _clear_rect) into a fully-walled interior, rather
        than wall-painting, mirrors the labyrinth_left/right topology.
        """
        w = self.rng.uniform(max(16.0, self.width_range[0]),
                             max(16.0, self.width_range[1]))
        h = self.rng.uniform(max(10.0, self.height_range[0]),
                             max(10.0, self.height_range[1]))
        grid = self._make_grid_with_walls(w, h)

        t = cfg.WALL_THICKNESS

        # Fill interior with walls (will be carved)
        grid.add_rectangular_obstacle(t, w - t, t, h - t)

        cw = self.rng.uniform(1.3, 1.8)
        n_h = int(self.rng.integers(3, 6))  # 3-5 horizontal corridors

        # Randomize start corner to remove the "always-bottom-left" bias.
        start_bottom = self.rng.integers(0, 2) == 0
        start_left = self.rng.integers(0, 2) == 0

        # y-positions of horizontal corridors (centerlines)
        y_lo = t + cw / 2 + 0.3
        y_hi = h - t - cw / 2 - 0.3
        y_levels = np.linspace(y_lo, y_hi, n_h)
        if not start_bottom:
            y_levels = y_levels[::-1]

        x_left = t + cw / 2 + 0.3
        x_right = w - t - cw / 2 - 0.3

        # Build path waypoints: alternate ends at successive y-levels.
        waypoints = []
        for i, y in enumerate(y_levels):
            forward = (i % 2 == 0) == start_left
            if forward:
                waypoints.append((x_left, y))
                waypoints.append((x_right, y))
            else:
                waypoints.append((x_right, y))
                waypoints.append((x_left, y))

        # Carve straight segments between consecutive waypoints
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]
            if abs(x1 - x2) < 1e-6:  # vertical segment
                self._clear_rect(grid,
                                 x1 - cw / 2, x1 + cw / 2,
                                 min(y1, y2) - cw / 2, max(y1, y2) + cw / 2)
            else:  # horizontal segment
                self._clear_rect(grid,
                                 min(x1, x2) - cw / 2, max(x1, x2) + cw / 2,
                                 y1 - cw / 2, y1 + cw / 2)

        # Add 1-2 short dead-end stubs branching off the serpentine
        n_stubs = int(self.rng.integers(1, 3))
        for _ in range(n_stubs):
            i = int(self.rng.integers(0, n_h))
            y = y_levels[i]
            stub_x = self.rng.uniform(x_left + 1.5, x_right - 1.5)
            stub_len = self.rng.uniform(1.0, 2.0)
            go_up = self.rng.integers(0, 2) == 0
            if go_up:
                self._clear_rect(grid,
                                 stub_x - cw / 2, stub_x + cw / 2,
                                 y, min(y + stub_len, h - t - 0.3))
            else:
                self._clear_rect(grid,
                                 stub_x - cw / 2, stub_x + cw / 2,
                                 max(y - stub_len, t + 0.3), y)
        return grid

    def _generate_dense_multi_room(self):
        """3×3 or 3×4 grid of small sub-rooms with random doorways.

        Targets the many_rooms GADEN map: many small rectangular rooms
        connected via narrow doorways. Sub-room sizes are jittered ±20%
        so the grid is irregular. Each interior wall segment has a 70%
        chance of containing a doorway.
        """
        w = self.rng.uniform(max(16.0, self.width_range[0]),
                             max(16.0, self.width_range[1]))
        h = self.rng.uniform(max(12.0, self.height_range[0]),
                             max(12.0, self.height_range[1]))
        grid = self._make_grid_with_walls(w, h)

        t = cfg.WALL_THICKNESS
        n_cols = int(self.rng.choice([3, 4]))
        n_rows = 3

        # Grid lines with ±20% jitter on interior lines
        x_lines = np.linspace(t, w - t, n_cols + 1)
        y_lines = np.linspace(t, h - t, n_rows + 1)
        jitter_x = ((w - 2 * t) / n_cols) * 0.2
        jitter_y = ((h - 2 * t) / n_rows) * 0.2
        for i in range(1, n_cols):
            x_lines[i] += self.rng.uniform(-jitter_x, jitter_x)
        for j in range(1, n_rows):
            y_lines[j] += self.rng.uniform(-jitter_y, jitter_y)

        # Horizontal interior walls (between rows j and j+1) at y = y_lines[j]
        for j in range(1, n_rows):
            y = y_lines[j]
            for i in range(n_cols):
                seg_x_min = x_lines[i]
                seg_x_max = x_lines[i + 1]
                if seg_x_max - seg_x_min < cfg.MIN_GAP_SIZE + 0.4:
                    continue
                grid.add_rectangular_obstacle(seg_x_min, seg_x_max,
                                              y - t / 2, y + t / 2)
                if self.rng.random() < 0.7:
                    dw = self.rng.uniform(cfg.MIN_GAP_SIZE,
                                          min(2.0, seg_x_max - seg_x_min - 0.3))
                    dx = self.rng.uniform(seg_x_min + 0.15,
                                          seg_x_max - dw - 0.15)
                    self._clear_rect(grid, dx, dx + dw,
                                     y - t / 2, y + t / 2)

        # Vertical interior walls (between cols i and i+1) at x = x_lines[i]
        for i in range(1, n_cols):
            x = x_lines[i]
            for j in range(n_rows):
                seg_y_min = y_lines[j]
                seg_y_max = y_lines[j + 1]
                if seg_y_max - seg_y_min < cfg.MIN_GAP_SIZE + 0.4:
                    continue
                grid.add_rectangular_obstacle(x - t / 2, x + t / 2,
                                              seg_y_min, seg_y_max)
                if self.rng.random() < 0.7:
                    dw = self.rng.uniform(cfg.MIN_GAP_SIZE,
                                          min(2.0, seg_y_max - seg_y_min - 0.3))
                    dy = self.rng.uniform(seg_y_min + 0.15,
                                          seg_y_max - dw - 0.15)
                    self._clear_rect(grid, x - t / 2, x + t / 2,
                                     dy, dy + dw)
        return grid

    def _generate_hybrid(self):
        """Two halves with different obstacle types, separated by a wall + doorway.

        Targets ultimate's hybrid topology: forces the agent to traverse a
        narrow choke between two qualitatively different sub-environments.
        """
        w = self.rng.uniform(max(16.0, self.width_range[0]),
                             max(16.0, self.width_range[1]))
        h = self.rng.uniform(max(10.0, self.height_range[0]),
                             max(10.0, self.height_range[1]))
        grid = self._make_grid_with_walls(w, h)

        t = cfg.WALL_THICKNESS
        vertical_split = self.rng.integers(0, 2) == 0

        if vertical_split:
            split_x = self.rng.uniform(0.4 * w, 0.6 * w)
            grid.add_rectangular_obstacle(split_x - t / 2, split_x + t / 2,
                                          t, h - t)
            dw = self.rng.uniform(cfg.MIN_GAP_SIZE, 2.0)
            dy = self.rng.uniform(t + 0.5, h - t - dw - 0.5)
            self._clear_rect(grid, split_x - t / 2, split_x + t / 2,
                             dy, dy + dw)
            doorway_center = (split_x, dy + dw / 2)
            self._paint_random_obstacles(grid,
                                         t + 0.3, split_x - t / 2 - 0.3,
                                         t + 0.3, h - t - 0.3,
                                         doorway_center)
            self._paint_random_obstacles(grid,
                                         split_x + t / 2 + 0.3, w - t - 0.3,
                                         t + 0.3, h - t - 0.3,
                                         doorway_center)
        else:
            split_y = self.rng.uniform(0.4 * h, 0.6 * h)
            grid.add_rectangular_obstacle(t, w - t,
                                          split_y - t / 2, split_y + t / 2)
            dw = self.rng.uniform(cfg.MIN_GAP_SIZE, 2.0)
            dx = self.rng.uniform(t + 0.5, w - t - dw - 0.5)
            self._clear_rect(grid, dx, dx + dw,
                             split_y - t / 2, split_y + t / 2)
            doorway_center = (dx + dw / 2, split_y)
            self._paint_random_obstacles(grid,
                                         t + 0.3, w - t - 0.3,
                                         t + 0.3, split_y - t / 2 - 0.3,
                                         doorway_center)
            self._paint_random_obstacles(grid,
                                         t + 0.3, w - t - 0.3,
                                         split_y + t / 2 + 0.3, h - t - 0.3,
                                         doorway_center)
        return grid

    def _paint_random_obstacles(self, grid, x0, x1, y0, y1,
                                doorway_center=None, clearance=1.5):
        """Paint 1-2 random small obstacles in a sub-rectangle.

        Used by _generate_hybrid. Avoids placing obstacles within
        `clearance` metres of `doorway_center` so the doorway remains
        passable.
        """
        t = cfg.WALL_THICKNESS
        region_w = x1 - x0
        region_h = y1 - y0
        if region_w < 3.0 or region_h < 3.0:
            return

        n_obs = int(self.rng.integers(1, 3))
        for _ in range(n_obs):
            for _attempt in range(8):
                obs_type = self.rng.choice(["block", "wall_h", "wall_v", "L"])
                if obs_type == "block":
                    sz = self.rng.uniform(0.6, 1.2)
                    cx = self.rng.uniform(x0 + sz / 2 + 0.2, x1 - sz / 2 - 0.2)
                    cy = self.rng.uniform(y0 + sz / 2 + 0.2, y1 - sz / 2 - 0.2)
                    rect = (cx - sz / 2, cx + sz / 2, cy - sz / 2, cy + sz / 2)
                elif obs_type == "wall_h":
                    wl = self.rng.uniform(1.2, min(region_w * 0.6, 3.5))
                    wx = self.rng.uniform(x0 + 0.2, x1 - wl - 0.2)
                    wy = self.rng.uniform(y0 + 0.4, y1 - 0.4)
                    rect = (wx, wx + wl, wy - t / 2, wy + t / 2)
                elif obs_type == "wall_v":
                    wl = self.rng.uniform(1.2, min(region_h * 0.6, 3.5))
                    wx = self.rng.uniform(x0 + 0.4, x1 - 0.4)
                    wy = self.rng.uniform(y0 + 0.2, y1 - wl - 0.2)
                    rect = (wx - t / 2, wx + t / 2, wy, wy + wl)
                else:  # L-shape: short horizontal arm + short vertical arm
                    arm = self.rng.uniform(1.0, 1.8)
                    cx = self.rng.uniform(x0 + 0.5, x1 - arm - 0.5)
                    cy = self.rng.uniform(y0 + 0.5, y1 - arm - 0.5)
                    rect = (cx, cx + arm, cy, cy + arm + t)
                # Clearance check around the doorway
                if doorway_center is not None:
                    ddx = max(0.0, max(rect[0] - doorway_center[0],
                                      doorway_center[0] - rect[1]))
                    ddy = max(0.0, max(rect[2] - doorway_center[1],
                                      doorway_center[1] - rect[3]))
                    if (ddx ** 2 + ddy ** 2) ** 0.5 < clearance:
                        continue
                # Place obstacle
                if obs_type == "L":
                    arm = rect[1] - rect[0]
                    grid.add_rectangular_obstacle(rect[0], rect[1],
                                                  rect[2], rect[2] + t)
                    grid.add_rectangular_obstacle(rect[0], rect[0] + t,
                                                  rect[2], rect[2] + arm)
                else:
                    grid.add_rectangular_obstacle(*rect)
                break

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
