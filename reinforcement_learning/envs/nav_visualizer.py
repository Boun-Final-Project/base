"""
Step-by-step visualization for the wall-following navigation RL environment.

2-panel layout:
  Left  — map with occupancy grid walls, trajectory, LiDAR rays, goal, robot
  Right — episode metrics (wall distance + cumulative reward)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from pathlib import Path

from .. import config as cfg


class NavStepVisualizer:
    """Save step visualizations as PNG frames for the navigation environment."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_count = 0
        self._wall_dists: list = []
        self._cum_rewards: list = []

    def save_step(self,
                  occupancy_grid,           # OccupancyGrid object
                  robot_pos: tuple,         # (x, y) metres
                  heading: float,           # radians
                  goal_pos: tuple,          # (x, y) metres
                  lidar_rays: np.ndarray,   # (N,) normalised [0,1]
                  lidar_angles: np.ndarray, # (N,) radians absolute
                  trajectory: list,         # [(x,y), ...] all positions so far
                  wall_dist: float,         # metres
                  step_reward: float,
                  cum_reward: float,
                  step_num: int,
                  room_width: float,
                  room_height: float):
        """Render and save a single step as a PNG."""

        self._wall_dists.append(wall_dist)
        self._cum_rewards.append(cum_reward)

        res = cfg.GRID_RESOLUTION
        fig = plt.figure(figsize=(12, 5))

        # ------------------------------------------------------------------ #
        # Panel 1 — Map
        # ------------------------------------------------------------------ #
        ax = fig.add_subplot(1, 2, 1)

        # 1. Occupancy grid walls
        for gy in range(occupancy_grid.grid_height):
            for gx in range(occupancy_grid.grid_width):
                if occupancy_grid.grid[gy, gx] != 0:
                    ax.add_patch(
                        Rectangle(
                            (gx * res, gy * res), res, res,
                            color="#555555", zorder=1,
                        )
                    )

        # 2. Trajectory
        if len(trajectory) >= 2:
            traj = np.array(trajectory)
            xs, ys = traj[:, 0], traj[:, 1]
            ax.plot(xs, ys, color="steelblue", lw=1.2, alpha=0.7, zorder=2)

        # 3. LiDAR rays
        for i in range(len(lidar_rays)):
            dist_m = lidar_rays[i] * cfg.LIDAR_MAX_RANGE
            ex = robot_pos[0] + dist_m * np.cos(lidar_angles[i])
            ey = robot_pos[1] + dist_m * np.sin(lidar_angles[i])
            colour = plt.cm.RdYlGn(lidar_rays[i])
            ax.plot(
                [robot_pos[0], ex], [robot_pos[1], ey],
                color=colour, lw=0.6, alpha=0.5, zorder=3,
            )
            ax.scatter(ex, ey, s=8, color=colour, zorder=4)

        # 4. Goal
        ax.plot(
            *goal_pos,
            marker="*", color="gold", markersize=14,
            markeredgecolor="black", markeredgewidth=0.8, zorder=5,
        )

        # 5. Robot body
        ax.add_patch(Circle(robot_pos, cfg.ROBOT_RADIUS, color="tomato", zorder=6))

        # 6. Heading arrow
        ax.annotate(
            "",
            xy=(
                robot_pos[0] + 0.4 * np.cos(heading),
                robot_pos[1] + 0.4 * np.sin(heading),
            ),
            xytext=robot_pos,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            zorder=7,
        )

        ax.set_aspect("equal")
        ax.set_xlim(0, room_width)
        ax.set_ylim(0, room_height)
        ax.set_title("Map", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # ------------------------------------------------------------------ #
        # Panel 2 — Metrics
        # ------------------------------------------------------------------ #
        ax2 = fig.add_subplot(1, 2, 2)

        # Primary y-axis: wall distance (steelblue)
        ax2.plot(
            range(len(self._wall_dists)), self._wall_dists,
            color="steelblue", lw=1.5, label="wall dist",
        )
        ax2.axhline(
            0.5, color="steelblue", linestyle="--", lw=1, alpha=0.6,
            label="target 0.5m",
        )
        ax2.set_ylabel("Wall distance (m)", color="steelblue", fontsize=9)
        ax2.set_ylim(0, cfg.LIDAR_MAX_RANGE)
        ax2.tick_params(axis="y", labelcolor="steelblue")

        # Secondary y-axis: cumulative reward (darkorange)
        ax2b = ax2.twinx()
        ax2b.plot(
            range(len(self._cum_rewards)), self._cum_rewards,
            color="darkorange", lw=1.5, label="cum reward",
        )
        ax2b.set_ylabel("Cumulative reward", color="darkorange", fontsize=9)
        ax2b.tick_params(axis="y", labelcolor="darkorange")

        ax2.set_xlabel("Step")
        ax2.set_title("Episode metrics", fontsize=10)

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

        # ------------------------------------------------------------------ #
        # Supertitle
        # ------------------------------------------------------------------ #
        dist_to_goal = float(
            np.sqrt(
                (robot_pos[0] - goal_pos[0]) ** 2
                + (robot_pos[1] - goal_pos[1]) ** 2
            )
        )
        fig.suptitle(
            f"step={step_num:3d} | reward={step_reward:+.2f} | "
            f"wall={wall_dist:.2f}m | dist_goal={dist_to_goal:.2f}m",
            fontsize=10,
        )

        # ------------------------------------------------------------------ #
        # Save
        # ------------------------------------------------------------------ #
        fig.tight_layout()
        filename = self.output_dir / f"step_{self.step_count:04d}.png"
        fig.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close(fig)
        self.step_count += 1
