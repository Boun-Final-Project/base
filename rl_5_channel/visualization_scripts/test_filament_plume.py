"""
Visual test for the filament-based gas dispersion model.

Generates a map with obstacles, runs the filament plume for N steps,
and renders:
  1. Filament positions colored by age
  2. Concentration heatmap along a cross-wind transect
  3. Intermittency plot (concentration at fixed point over time)
  4. Obstacle reflection verification
  5. Animation of plume evolution (--animate)

Usage:
    python3 -m rl_5_channel.test_filament_plume
    python3 -m rl_5_channel.test_filament_plume --seed 42 --steps 200
    python3 -m rl_5_channel.test_filament_plume --animate --steps 300
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_5_channel import config as cfg
from rl_5_channel.envs.map_generator import MapGenerator
from rl_5_channel.envs.filament_plume import FilamentPlume

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "visual_test", "filament")


def run_visual_test(seed=42, total_steps=200, output_dir=None):
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    map_gen = MapGenerator(rng=rng)

    # Generate a map with obstacles (template 2 = U-shape is good for testing reflection)
    map_data = map_gen.generate(template_id=2)
    grid = map_data["grid"]
    source_pos = map_data["source_pos"]

    # Sample wind
    wind_speed = rng.uniform(*cfg.WIND_SPEED_RANGE)
    wind_angle = rng.uniform(0, 2 * np.pi)
    wind_vx = wind_speed * np.cos(wind_angle)
    wind_vy = wind_speed * np.sin(wind_angle)

    print(f"Seed:        {seed}")
    print(f"Map:         template 2 (U-shape)")
    print(f"Room size:   {map_data['width']:.1f} x {map_data['height']:.1f} m")
    print(f"Source:      ({source_pos[0]:.1f}, {source_pos[1]:.1f})")
    print(f"Wind:        speed={wind_speed:.2f} m/s, angle={np.degrees(wind_angle):.1f}°")
    print(f"Filaments:   {cfg.FILAMENTS_PER_STEP}/step, max_age={cfg.FILAMENT_MAX_AGE}")
    print()

    # Create plume
    plume = FilamentPlume(
        source_pos=source_pos,
        wind_speed=wind_speed,
        wind_angle=wind_angle,
        occupancy_grid=grid,
        rng=rng,
    )

    # --- Plot 1: Filament positions colored by age ---
    for step in range(total_steps):
        plume.update()

    filaments = plume.get_all_filaments()
    positions = filaments["positions"]
    sigmas = filaments["sigmas"]
    ages = filaments["ages"]

    fig1, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect("equal")
    ax.set_title(
        f"Filament Plume after {total_steps} steps\n"
        f"N={plume.n_active} active filaments | "
        f"wind {wind_speed:.2f} m/s @ {np.degrees(wind_angle):.0f}°",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Draw obstacles
    for gy in range(grid.grid_height):
        for gx in range(grid.grid_width):
            if grid.grid[gy, gx] != 0:
                x = gx * grid.resolution
                y = gy * grid.resolution
                rect = plt.Rectangle(
                    (x, y), grid.resolution, grid.resolution,
                    facecolor="gray", edgecolor="black", linewidth=0.5, alpha=0.7,
                )
                ax.add_patch(rect)

    # Draw filaments
    max_age = max(1, ages.max()) if len(ages) > 0 else 1
    for i in range(len(positions)):
        circle = plt.Circle(
            (positions[i, 0], positions[i, 1]),
            radius=2.0 * sigmas[i],
            facecolor=plt.cm.viridis(ages[i] / max_age),
            alpha=0.35,
            edgecolor="none",
        )
        ax.add_patch(circle)

    ax.plot(source_pos[0], source_pos[1], "r*", markersize=15, label="Source")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig1.savefig(output_path / "filament_positions.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[1/4] Filament positions → {output_path / 'filament_positions.png'}")

    # --- Plot 2: Cross-wind concentration profile ---
    # Query concentration along a line perpendicular to wind direction
    n_points = 200
    downwind_dist = 5.0  # 5 m downwind from source
    crosswind_span = 4.0  # ±4 m cross-wind

    # Unit vectors
    wind_unit = np.array([np.cos(wind_angle), np.sin(wind_angle)])
    cross_unit = np.array([-wind_unit[1], wind_unit[0]])  # perpendicular

    # Center of transect: source + downwind_dist * wind_unit
    transect_center = source_pos + downwind_dist * wind_unit
    transect_points = np.array([
        transect_center + t * cross_unit
        for t in np.linspace(-crosswind_span, crosswind_span, n_points)
    ])

    concentrations = np.array([
        plume.concentration_at(p) for p in transect_points
    ])

    crosswind_offsets = np.array([
        np.dot(p - transect_center, cross_unit) for p in transect_points
    ])

    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(crosswind_offsets, concentrations, "b-", linewidth=1.5)
    ax.set_xlabel("Cross-wind offset (m)", fontsize=11)
    ax.set_ylabel("Concentration", fontsize=11)
    ax.set_title(
        f"Concentration profile {downwind_dist:.0f} m downwind\n"
        f"peak={concentrations.max():.4f} at offset={crosswind_offsets[np.argmax(concentrations)]:.2f} m",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Plume centerline")
    ax.legend()

    fig2.savefig(output_path / "crosswind_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[2/4] Cross-wind profile → {output_path / 'crosswind_profile.png'}")

    # --- Plot 3: Intermittency at a fixed point ---
    # Place a sensor 3 m downwind from source
    sensor_pos = source_pos + 3.0 * wind_unit

    # Reset and re-run, recording concentration at sensor each step
    plume2 = FilamentPlume(
        source_pos=source_pos,
        wind_speed=wind_speed,
        wind_angle=wind_angle,
        occupancy_grid=grid,
        rng=np.random.default_rng(seed),  # Same seed for determinism
    )

    sensor_readings = []
    n_active_over_time = []
    for step in range(total_steps):
        plume2.update()
        sensor_readings.append(plume2.concentration_at(sensor_pos))
        n_active_over_time.append(plume2.n_active)

    sensor_readings = np.array(sensor_readings)

    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(range(total_steps), sensor_readings, "b-", linewidth=0.8)
    ax1.set_ylabel("Concentration", fontsize=10)
    ax1.set_title(
        f"Intermittency at sensor (3 m downwind)\n"
        f"mean={sensor_readings.mean():.4f}, std={sensor_readings.std():.4f}, "
        f"zero_fraction={(sensor_readings < 1e-10).mean():.0%}",
        fontsize=11, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(total_steps), n_active_over_time, "g-", linewidth=0.8)
    ax2.set_xlabel("Step", fontsize=10)
    ax2.set_ylabel("Active filaments", fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig3.savefig(output_path / "intermittency.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[3/4] Intermittency time series → {output_path / 'intermittency.png'}")

    # --- Plot 4: Obstacle reflection verification ---
    # Check that no filaments are inside obstacles
    n_inside = 0
    for i in range(len(positions)):
        if not grid.is_valid(position=(positions[i, 0], positions[i, 1]), radius=0.0):
            n_inside += 1

    fig4, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect("equal")

    status = f"PASS (0 inside)" if n_inside == 0 else f"FAIL ({n_inside} inside)"
    ax.set_title(
        f"Obstacle Reflection Check — {status}\n"
        f"Total filaments: {len(positions)}, inside walls: {n_inside}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Draw obstacles
    for gy in range(grid.grid_height):
        for gx in range(grid.grid_width):
            if grid.grid[gy, gx] != 0:
                x = gx * grid.resolution
                y = gy * grid.resolution
                rect = plt.Rectangle(
                    (x, y), grid.resolution, grid.resolution,
                    facecolor="gray", edgecolor="red", linewidth=1.0, alpha=0.7,
                )
                ax.add_patch(rect)

    # Draw filaments
    for i in range(len(positions)):
        inside = not grid.is_valid(position=(positions[i, 0], positions[i, 1]), radius=0.0)
        color = "red" if inside else plt.cm.viridis(ages[i] / max_age)
        circle = plt.Circle(
            (positions[i, 0], positions[i, 1]),
            radius=2.0 * sigmas[i],
            facecolor=color,
            alpha=0.4,
            edgecolor="none",
        )
        ax.add_patch(circle)

    ax.plot(source_pos[0], source_pos[1], "r*", markersize=15, label="Source")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig4.savefig(output_path / "reflection_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"[4/4] Reflection check → {output_path / 'reflection_check.png'}")

    # --- Summary ---
    print()
    print("=" * 60)
    print("FILAMENT PLUME VISUAL TEST SUMMARY")
    print("=" * 60)
    print(f"Active filaments:    {plume.n_active}")
    print(f"Peak concentration:  {concentrations.max():.6f}")
    print(f"Sensor mean:         {sensor_readings.mean():.6f}")
    print(f"Sensor zero fraction:{(sensor_readings < 1e-10).mean():.0%}")
    print(f"Filaments in walls:  {n_inside} {'✓' if n_inside == 0 else '✗'}")
    print(f"Output directory:    {output_path.absolute()}")
    print("=" * 60)


def run_animation(seed=42, total_steps=300, output_dir=None,
                  fps=15, every_n=1):
    """Animate the plume evolving step by step and save as a GIF.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    total_steps : int
        Total simulation steps to animate.
    output_dir : str
        Directory to write ``plume_animation.gif``.
    fps : int
        Frames per second in the output GIF.
    every_n : int
        Capture one frame every *every_n* simulation steps (thins the GIF).
    """
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    map_gen = MapGenerator(rng=rng)
    map_data = map_gen.generate(template_id=2)
    grid = map_data["grid"]
    source_pos = map_data["source_pos"]

    wind_speed = rng.uniform(*cfg.WIND_SPEED_RANGE)
    wind_angle = rng.uniform(0, 2 * np.pi)
    wind_vx = wind_speed * np.cos(wind_angle)
    wind_vy = wind_speed * np.sin(wind_angle)

    # Fixed sensor 3 m downwind of source
    wind_unit = np.array([np.cos(wind_angle), np.sin(wind_angle)])
    sensor_pos = source_pos + 3.0 * wind_unit

    print(f"Seed:      {seed}")
    print(f"Room:      {map_data['width']:.1f} x {map_data['height']:.1f} m")
    print(f"Source:    ({source_pos[0]:.1f}, {source_pos[1]:.1f})")
    print(f"Wind:      {wind_speed:.2f} m/s @ {np.degrees(wind_angle):.1f}°")
    print(f"Steps:     {total_steps}  (every_n={every_n}, fps={fps})")

    plume = FilamentPlume(
        source_pos=source_pos,
        wind_speed=wind_speed,
        wind_angle=wind_angle,
        occupancy_grid=grid,
        rng=rng,
    )

    # Build obstacle patch list once (static background)
    obstacle_patches = []
    for gy in range(grid.grid_height):
        for gx in range(grid.grid_width):
            if grid.grid[gy, gx] != 0:
                obstacle_patches.append(
                    mpatches.Rectangle(
                        (gx * grid.resolution, gy * grid.resolution),
                        grid.resolution, grid.resolution,
                    )
                )

    # --- Figure layout: plume map (left) + sensor time-series (right) ---
    fig, (ax_map, ax_ts) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    fig.tight_layout(pad=3.0)

    # Map axes
    ax_map.set_xlim(0, grid.width)
    ax_map.set_ylim(0, grid.height)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")

    obs_col = PatchCollection(
        obstacle_patches, facecolor="dimgray", edgecolor="black",
        linewidth=0.4, alpha=0.8, zorder=1,
    )
    ax_map.add_collection(obs_col)
    ax_map.plot(*source_pos, "r*", markersize=14, zorder=5, label="Source")
    ax_map.plot(*sensor_pos, "c^", markersize=9, zorder=5, label="Sensor")

    # Wind arrow at top-right corner of map
    arrow_origin = np.array([grid.width * 0.85, grid.height * 0.9])
    arrow_scale = min(grid.width, grid.height) * 0.08
    ax_map.annotate(
        "", xy=arrow_origin + wind_unit * arrow_scale, xytext=arrow_origin,
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=2),
        zorder=6,
    )
    ax_map.text(
        arrow_origin[0], arrow_origin[1] - grid.height * 0.05,
        f"wind\n{wind_speed:.2f} m/s", ha="center", fontsize=8, color="steelblue",
    )
    ax_map.legend(loc="upper left", fontsize=8)

    title = ax_map.set_title("", fontsize=11, fontweight="bold")

    # Time-series axes
    ax_ts.set_xlabel("Step", fontsize=10)
    ax_ts.set_ylabel("Concentration", fontsize=10)
    ax_ts.set_title("Sensor (3 m downwind)", fontsize=10, fontweight="bold")
    ax_ts.set_xlim(0, total_steps)
    ax_ts.grid(True, alpha=0.3)
    ts_line, = ax_ts.plot([], [], "b-", linewidth=0.9)
    ts_dot,  = ax_ts.plot([], [], "ro", markersize=5)

    # Colour map for filament age
    cmap = plt.cm.plasma

    # Mutable state captured by the closure
    state = {
        "step": 0,
        "sensor_steps": [],
        "sensor_vals": [],
        "filament_circles": [],   # list of Circle artists added to ax_map
    }

    def _remove_filament_artists():
        for c in state["filament_circles"]:
            c.remove()
        state["filament_circles"].clear()

    def init():
        ts_line.set_data([], [])
        ts_dot.set_data([], [])
        title.set_text("")
        return [ts_line, ts_dot, title]

    def animate_frame(frame_idx):
        # Advance simulation by every_n steps
        for _ in range(every_n):
            plume.update()
            state["step"] += 1
            conc = plume.concentration_at(sensor_pos)
            state["sensor_steps"].append(state["step"])
            state["sensor_vals"].append(conc)

        step = state["step"]

        # Update filament circles
        _remove_filament_artists()
        data = plume.get_all_filaments()
        pos   = data["positions"]
        sigs  = data["sigmas"]
        ages  = data["ages"]
        max_age = max(1, ages.max()) if len(ages) > 0 else 1

        for i in range(len(pos)):
            age_norm = ages[i] / max_age
            c = plt.Circle(
                (pos[i, 0], pos[i, 1]),
                radius=max(sigs[i], 0.05),
                facecolor=cmap(age_norm),
                alpha=0.35,
                edgecolor="none",
                zorder=2,
            )
            ax_map.add_patch(c)
            state["filament_circles"].append(c)

        title.set_text(
            f"Step {step}/{total_steps} — "
            f"{plume.n_active} filaments | "
            f"wind {wind_speed:.2f} m/s @ {np.degrees(wind_angle):.0f}°"
        )

        # Update time-series
        sv = state["sensor_vals"]
        ss = state["sensor_steps"]
        ts_line.set_data(ss, sv)
        ts_dot.set_data([ss[-1]], [sv[-1]])
        # Auto-scale y axis with a little headroom
        if max(sv) > 0:
            ax_ts.set_ylim(0, max(sv) * 1.15)
        ax_ts.set_xlim(0, total_steps)

        return state["filament_circles"] + [ts_line, ts_dot, title]

    n_frames = total_steps // every_n
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=n_frames,
        init_func=init, blit=False, interval=1000 // fps,
    )

    out_path = output_path / "plume_animation.gif"
    print(f"Saving {n_frames} frames → {out_path}  (this may take a moment…)")
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"Done → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--animate", action="store_true",
        help="Generate a GIF animation instead of static plots.",
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Frames per second for the animation GIF (default: 15).",
    )
    parser.add_argument(
        "--every-n", type=int, default=1,
        help="Capture one frame every N simulation steps (default: 1).",
    )
    args = parser.parse_args()

    if args.animate:
        run_animation(
            seed=args.seed,
            total_steps=args.steps,
            output_dir=args.output_dir,
            fps=args.fps,
            every_n=args.every_n,
        )
    else:
        run_visual_test(seed=args.seed, total_steps=args.steps, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
