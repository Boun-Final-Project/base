"""
Visual debugger for gas concentration in tough conditions.

Generates images showing:
- Occupancy grid (walls)
- Gas concentration heatmap
- Effective source position (after wind offset + snap)
- True source position
- Robot position
- Dijkstra distance field

Focuses on scenarios that are likely to be problematic:
1. Wind pushing effective source into/near walls
2. Source near boundary walls
3. Source behind obstacles (single_wall, u_shape, multi_room)
4. Large wind offsets
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.gas_source_env import GasSourceEnv

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_gas_images")

TEMPLATE_NAMES = [
    "empty", "single_wall", "u_shape", "three_walls", "complex_maze", "multi_room"
]


def render_debug_image(env, title_extra="", filename="debug.png"):
    """Render a 2-panel debug image: concentration heatmap + Dijkstra distances."""
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])

    source = env._source_pos
    robot = env._robot_pos
    eff_x = max(0, min(source[0] + env._wind_offset[0],
                       env._map_width - cfg.COARSE_RESOLUTION))
    eff_y = max(0, min(source[1] + env._wind_offset[1],
                       env._map_height - cfg.COARSE_RESOLUTION))
    eff_source_snapped = env._igdm.snap_to_free_cell(eff_x, eff_y)
    eff_source_raw = (eff_x, eff_y)

    dists = env._dijkstra_from_source
    sigma_m = env._igdm.get_sigma_m(0)

    # --- Panel 1: Gas concentration heatmap ---
    x_res = max(100, int(env._map_width * 10))
    y_res = max(60, int(env._map_height * 10))
    x_grid = np.linspace(0, env._map_width, x_res)
    y_grid = np.linspace(0, env._map_height, y_res)
    X, Y = np.meshgrid(x_grid, y_grid)

    Z = np.zeros_like(X)
    coarse_res = env._igdm.coarse_res
    d_rows, d_cols = dists.shape
    for i in range(len(y_grid)):
        for j in range(len(x_grid)):
            r = int(Y[i, j] / coarse_res)
            c = int(X[i, j] / coarse_res)
            r = max(0, min(r, d_rows - 1))
            c = max(0, min(c, d_cols - 1))
            d = dists[r, c]
            if not np.isinf(d):
                d = max(d, 0.1)
                Z[i, j] = cfg.SOURCE_RELEASE_RATE * np.exp(-(d ** 2) / (2 * sigma_m ** 2))

    im1 = ax1.contourf(X, Y, Z, levels=25, cmap='hot_r', zorder=1)

    # Walls (zorder=2 so they appear on top of contourf)
    wall_mask = np.ma.masked_where(env._grid.grid == 0, env._grid.grid)
    ax1.imshow(
        wall_mask, origin="lower",
        extent=[0, env._map_width, 0, env._map_height],
        cmap="Greys", vmin=0, vmax=1, aspect="equal",
        zorder=2,
    )

    # Source markers
    ax1.plot(source[0], source[1], "r*", markersize=18, label="True source", zorder=5)
    ax1.plot(eff_source_snapped[0], eff_source_snapped[1], "m^", markersize=12,
             label="Eff source (snapped)", zorder=5)
    if abs(eff_source_raw[0] - eff_source_snapped[0]) > 0.01 or \
       abs(eff_source_raw[1] - eff_source_snapped[1]) > 0.01:
        ax1.plot(eff_source_raw[0], eff_source_raw[1], "mx", markersize=12,
                 label="Eff source (raw, in wall)", zorder=5)

    # Robot
    ax1.plot(robot[0], robot[1], "bo", markersize=10, label="Robot", zorder=5)

    # Wind arrow
    ws, wd = env._wind.speed, env._wind.direction
    arrow_len = 1.5
    ax1.annotate(
        "", xy=(2 + arrow_len * np.cos(wd), env._map_height - 2 + arrow_len * np.sin(wd)),
        xytext=(2, env._map_height - 2),
        arrowprops=dict(arrowstyle="->", color="purple", lw=2.5),
    )
    ax1.text(2, env._map_height - 0.8, f"wind {ws:.2f} m/s", fontsize=9, color="purple")

    # Concentration at robot and source
    conc_at_robot = env._get_concentration()
    dist_to_source = np.linalg.norm(robot - source)
    ax1.set_title(
        f"Gas Concentration (sigma_m={sigma_m:.2f})\n"
        f"Conc@robot={conc_at_robot:.6f}, dist={dist_to_source:.2f}m\n"
        f"Wind offset=({env._wind_offset[0]:.2f}, {env._wind_offset[1]:.2f})",
        fontsize=10,
    )
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    fig.colorbar(im1, cax=cbar_ax, label="Concentration")

    # --- Panel 2: Dijkstra distance field ---
    dists_display = dists.copy()
    dists_display[np.isinf(dists_display)] = np.nan
    im2 = ax2.imshow(
        dists_display, origin="lower",
        extent=[0, env._map_width, 0, env._map_height],
        cmap="viridis_r", aspect="equal", interpolation="nearest",
    )

    # Walls
    ax2.imshow(
        wall_mask, origin="lower",
        extent=[0, env._map_width, 0, env._map_height],
        cmap="Greys", vmin=0, vmax=1, aspect="equal",
    )

    ax2.plot(source[0], source[1], "r*", markersize=18, zorder=5)
    ax2.plot(eff_source_snapped[0], eff_source_snapped[1], "m^", markersize=12, zorder=5)
    ax2.plot(robot[0], robot[1], "bo", markersize=10, zorder=5)

    r_robot, c_robot = env._igdm._world_to_coarse_idx(robot[0], robot[1])
    dijkstra_at_robot = dists[r_robot, c_robot]
    ax2.set_title(
        f"Dijkstra Distance from Eff Source\n"
        f"Dijkstra@robot={dijkstra_at_robot:.2f}m",
        fontsize=10,
    )
    ax2.set_xlabel("x (m)")

    fig.suptitle(title_extra, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def scenario_all_templates_default(seed=42):
    """Each template with default seed — baseline reference."""
    print("\n=== Scenario: All templates (default seed) ===")
    for tid in range(6):
        env = GasSourceEnv(render_mode="rgb_array", template_id=tid)
        env.reset(seed=seed)
        render_debug_image(
            env,
            title_extra=f"Template {tid}: {TEMPLATE_NAMES[tid]} (seed={seed})",
            filename=f"template_{tid}_{TEMPLATE_NAMES[tid]}.png",
        )


def scenario_strong_wind(seeds=None):
    """Find cases where wind offset is large, pushing eff source far from true source."""
    print("\n=== Scenario: Strong wind (large offset) ===")
    if seeds is None:
        seeds = range(200)

    cases_found = 0
    for tid in range(6):
        for seed in seeds:
            env = GasSourceEnv(render_mode="rgb_array", template_id=tid)
            env.reset(seed=seed)

            offset_mag = np.linalg.norm(env._wind_offset)
            if offset_mag > 2.0:
                render_debug_image(
                    env,
                    title_extra=(f"STRONG WIND — Template {tid}: {TEMPLATE_NAMES[tid]} "
                                 f"(seed={seed}, |offset|={offset_mag:.2f}m)"),
                    filename=f"strong_wind_t{tid}_{TEMPLATE_NAMES[tid]}_s{seed}.png",
                )
                cases_found += 1
                break  # one per template
    print(f"  Found {cases_found} strong-wind cases")


def scenario_source_near_wall(seeds=None):
    """Find cases where source is placed very close to a wall."""
    print("\n=== Scenario: Source near wall ===")
    if seeds is None:
        seeds = range(300)

    cases_found = 0
    for tid in [1, 2, 3, 4, 5]:  # skip empty
        for seed in seeds:
            env = GasSourceEnv(render_mode="rgb_array", template_id=tid)
            env.reset(seed=seed)

            # Check if source is near any wall by checking surrounding cells
            sx, sy = env._source_pos
            near_wall = False
            for dx, dy in [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]:
                if not env._grid.is_valid(position=(sx + dx, sy + dy), radius=0.1):
                    near_wall = True
                    break

            if near_wall:
                render_debug_image(
                    env,
                    title_extra=(f"SOURCE NEAR WALL — Template {tid}: {TEMPLATE_NAMES[tid]} "
                                 f"(seed={seed})"),
                    filename=f"near_wall_t{tid}_{TEMPLATE_NAMES[tid]}_s{seed}.png",
                )
                cases_found += 1
                break  # one per template
    print(f"  Found {cases_found} near-wall cases")


def scenario_wind_into_wall():
    """Specifically test the bug case: wind pushes eff source into occupied cell."""
    print("\n=== Scenario: Wind pushes eff source into wall (pre-fix verification) ===")
    cases_found = 0

    for tid in range(6):
        for seed in range(500):
            env = GasSourceEnv(render_mode="rgb_array", template_id=tid)
            env.reset(seed=seed)

            # Check raw (pre-snap) effective source position
            eff_x = max(0, min(env._source_pos[0] + env._wind_offset[0],
                               env._map_width - cfg.COARSE_RESOLUTION))
            eff_y = max(0, min(env._source_pos[1] + env._wind_offset[1],
                               env._map_height - cfg.COARSE_RESOLUTION))
            r, c = env._igdm._world_to_coarse_idx(eff_x, eff_y)

            if env._igdm.coarse_grid[r, c] == 1:
                # This would have been the bug — eff source in a wall
                snapped = env._igdm.snap_to_free_cell(eff_x, eff_y)
                render_debug_image(
                    env,
                    title_extra=(f"WIND→WALL (snapped) — Template {tid}: {TEMPLATE_NAMES[tid]} "
                                 f"(seed={seed})\n"
                                 f"Raw eff=({eff_x:.2f},{eff_y:.2f}) → "
                                 f"Snapped=({snapped[0]:.2f},{snapped[1]:.2f})"),
                    filename=f"wind_into_wall_t{tid}_{TEMPLATE_NAMES[tid]}_s{seed}.png",
                )
                cases_found += 1
                break  # one per template

    print(f"  Found {cases_found} wind-into-wall cases (now fixed by snap)")


def scenario_zero_concentration_walk():
    """Walk toward source on each template, flag if zero conc within 2m."""
    print("\n=== Scenario: Zero concentration near source check ===")

    for tid in range(6):
        env = GasSourceEnv(template_id=tid)
        env.reset(seed=42)
        source = env._source_pos.copy()

        found_zero_near = False
        for step in range(min(cfg.MAX_STEPS, 300)):
            direction = source - env._robot_pos
            dist = np.linalg.norm(direction)
            angle = np.arctan2(direction[1], direction[0])
            action = np.array([(angle / (2 * np.pi)) % 1.0], dtype=np.float32)

            conc = env._get_concentration()
            if dist < 2.0 and conc == 0.0:
                found_zero_near = True

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        status = "ZERO CONC <2m" if found_zero_near else "OK"
        print(f"  Template {tid} ({TEMPLATE_NAMES[tid]:>14}): {status}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    scenario_all_templates_default()
    scenario_strong_wind()
    scenario_source_near_wall()
    scenario_wind_into_wall()
    scenario_zero_concentration_walk()

    print(f"\nAll images saved to: {OUTPUT_DIR}")
