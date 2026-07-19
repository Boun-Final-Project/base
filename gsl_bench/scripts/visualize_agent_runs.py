#!/usr/bin/env python3
"""Visual sanity check for zigzag / surge_cast — no ROS, no GADEN.

Runs each agent's real observe()/act() loop against a synthetic Gaussian-plume
world (open room, no walls: this is about the chemotaxis pattern itself, not
obstacle navigation) and saves the trajectory + gas field as a PNG.

    python3 visualize_agent_runs.py [--out-dir DIR] [--max-steps N]
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from gsl_bench.agent import MapInfo, Observation, ScenarioInfo, Waypoint
from gsl_bench.agents.surge_cast import SurgeCastAgent
from gsl_bench.agents.zigzag import ZigzagAgent

# --- synthetic world ---------------------------------------------------

WIDTH_M, HEIGHT_M = 12.0, 8.0
RESOLUTION = 0.1
SOURCE = (3.0, 4.0)
WIND_DIR = 0.0                    # downwind bearing: gas blows toward +x
WIND_SPEED = 0.6
START = (11.5, 7.0)               # downwind, off-centerline: needs a real search
GAS_THRESHOLD = 5.0
SUCCESS_RADIUS = 0.5

# same palette as the surge-cast state-machine diagram
STATE_COLORS = {'SURGE': '#b5651d', 'CAST': '#1e7f72', 'SPIRAL': '#6a4c93'}


def make_map() -> MapInfo:
    h, w = int(HEIGHT_M / RESOLUTION), int(WIDTH_M / RESOLUTION)
    grid = np.zeros((h, w), np.uint8)
    return MapInfo(grid=grid, resolution=RESOLUTION, origin_x=0.0, origin_y=0.0,
                    width_m=WIDTH_M, height_m=HEIGHT_M)


def gas_field(x, y):
    """Gaussian-plume centerline model: narrow near the source, widening and
    diluting downwind. Tuned so the chosen START is just below gas_threshold —
    the agent has to search before it gets a hit."""
    ux, uy = math.cos(WIND_DIR), math.sin(WIND_DIR)
    vx, vy = -math.sin(WIND_DIR), math.cos(WIND_DIR)
    dx, dy = x - SOURCE[0], y - SOURCE[1]
    along = dx * ux + dy * uy
    across = dx * vx + dy * vy
    if along <= 0.0:
        return 80.0 * math.exp(2.5 * along) * math.exp(-(across ** 2) / (2 * 0.3 ** 2))
    sigma = 0.3 + 0.12 * along
    return 80.0 * math.exp(-0.08 * along) * math.exp(-(across ** 2) / (2 * sigma ** 2))


def make_obs(m, x, y, step, rng):
    gas = max(0.0, gas_field(x, y) + rng.normal(0, 0.5))
    wind_dir = WIND_DIR + rng.normal(0, 0.05)
    return Observation(
        x=x, y=y, theta=0.0, gas_ppm=gas, wind_speed=WIND_SPEED,
        wind_direction=wind_dir, lidar=np.full(72, 3.0, np.float32),
        lidar_max_range=3.0, step=step, sim_time=float(step) * 0.5, map=m)


def run_episode(agent, m, max_steps, rng, track_state=False, start=START):
    agent.initialize()
    agent.reset(ScenarioInfo('viz', m, start[0], start[1], max_steps, 0.5))
    x, y = start
    xs, ys, states = [x], [y], []
    success_step = None
    for step in range(max_steps):
        agent.observe(make_obs(m, x, y, step, rng))
        wp = agent.act()

        # mirror the harness's hop cap (max_hop=1.0); no walls here to clamp against
        d = math.hypot(wp.x - x, wp.y - y)
        if d > 1.0:
            s = 1.0 / d
            wp = Waypoint(x + (wp.x - x) * s, y + (wp.y - y) * s)
        x, y = wp.x, wp.y
        xs.append(x)
        ys.append(y)
        if track_state:
            states.append(getattr(agent, 'state', 'CAST'))

        if math.hypot(x - SOURCE[0], y - SOURCE[1]) < SUCCESS_RADIUS:
            success_step = step
            break
    return xs, ys, states, success_step


def plot_run(xs, ys, states, success_step, title, out_path):
    xx = np.linspace(0, WIDTH_M, 240)
    yy = np.linspace(0, HEIGHT_M, 160)
    field = np.array([[gas_field(x, y) for x in xx] for y in yy])

    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.patch.set_facecolor('#0e1512')
    ax.set_facecolor('#0e1512')

    cf = ax.contourf(xx, yy, field, levels=18, cmap='YlOrBr', alpha=0.55)
    cb = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('gas concentration (ppm)', color='#e7efea')
    cb.ax.yaxis.set_tick_params(color='#e7efea')
    plt.setp(cb.ax.get_yticklabels(), color='#e7efea')

    if states:
        for i in range(len(xs) - 1):
            c = STATE_COLORS.get(states[i], '#4fbfae')
            ax.plot(xs[i:i + 2], ys[i:i + 2], color=c, linewidth=2.2, solid_capstyle='round')
        handles = [plt.Line2D([0], [0], color=c, lw=2.5, label=s)
                   for s, c in STATE_COLORS.items()]
        ax.legend(handles=handles, loc='upper left', facecolor='#131b17',
                  edgecolor='#2a3833', labelcolor='#e7efea', fontsize=9)
    else:
        pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap='BuGn', linewidth=2.4)
        lc.set_array(np.linspace(0.25, 1.0, len(segs)))
        ax.add_collection(lc)

    ax.scatter([xs[0]], [ys[0]], marker='o', s=90, color='#e7efea',
              edgecolor='#0e1512', zorder=5, label='start')
    ax.scatter([xs[-1]], [ys[-1]], marker='X', s=110, color='#e7efea',
              edgecolor='#0e1512', zorder=5, label='end')
    ax.scatter([SOURCE[0]], [SOURCE[1]], marker='*', s=260, color='#ffe08a',
              edgecolor='#0e1512', zorder=6, label='source')

    status = f'FOUND @ step {success_step}' if success_step is not None else 'NOT FOUND (max_steps)'
    ax.set_title(f'{title}  —  {status}', color='#e7efea', fontsize=13, pad=12)
    ax.set_xlabel('x (m)', color='#8fa69c')
    ax.set_ylabel('y (m)', color='#8fa69c')
    ax.tick_params(colors='#8fa69c')
    for spine in ax.spines.values():
        spine.set_color('#2a3833')
    ax.set_xlim(0, WIDTH_M)
    ax.set_ylim(0, HEIGHT_M)
    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=Path('.'))
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    m = make_map()

    rng = np.random.default_rng(args.seed)
    zz = ZigzagAgent({'step_size': 0.5, 'leg_length': 1.0, 'amplitude_growth': 0.15,
                      'upwind_drift': 0.03})
    xs, ys, _, done = run_episode(zz, m, args.max_steps, rng)
    out = args.out_dir / 'zigzag_run.png'
    plot_run(xs, ys, [], done, 'ZigzagAgent', out)
    print(f'zigzag:     {"FOUND" if done is not None else "max_steps"} '
          f'({len(xs)} positions) -> {out}')

    rng = np.random.default_rng(args.seed)
    sc = SurgeCastAgent({'gas_threshold': GAS_THRESHOLD, 'step_size': 0.5,
                        'leg_length': 1.0, 'amplitude_growth': 0.3,
                        'lost_steps_to_spiral': 30, 'spiral_growth': 0.15})
    xs, ys, states, done = run_episode(sc, m, args.max_steps, rng, track_state=True)
    out = args.out_dir / 'surge_cast_run.png'
    plot_run(xs, ys, states, done, 'SurgeCastAgent', out)
    print(f'surge_cast: {"FOUND" if done is not None else "max_steps"} '
          f'({len(xs)} positions) -> {out}')

    # A second surge_cast scenario, tuned to actually trigger SPIRAL: start far
    # enough off the plume's crosswind axis that CAST's early (still-narrow)
    # legs can't reach back into it, and lost_steps_to_spiral is lowered so the
    # agent gives up on casting before its legs would have grown wide enough.
    rng = np.random.default_rng(args.seed)
    sc_spiral = SurgeCastAgent({'gas_threshold': GAS_THRESHOLD, 'step_size': 0.5,
                               'leg_length': 1.0, 'amplitude_growth': 0.3,
                               'lost_steps_to_spiral': 10, 'spiral_growth': 0.15})
    spiral_start = (7.0, 7.5)
    xs, ys, states, done = run_episode(
        sc_spiral, m, args.max_steps, rng, track_state=True, start=spiral_start)
    out = args.out_dir / 'surge_cast_spiral_demo.png'
    plot_run(xs, ys, states, done, 'SurgeCastAgent (SPIRAL demo)', out)
    print(f'surge_cast (spiral demo): {"FOUND" if done is not None else "max_steps"} '
          f'({len(xs)} positions), SPIRAL used: {"SPIRAL" in states} -> {out}')


if __name__ == '__main__':
    main()
