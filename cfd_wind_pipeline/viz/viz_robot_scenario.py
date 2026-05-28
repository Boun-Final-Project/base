"""Visualize ONE scenario as the env sets it up: geometry, source/robot,
the spatial wind field that drives the plume, the gas plume after warmup,
AND the single scalar wind vector the policy actually observes.

Makes the 'spatial wind drives advection but policy sees only the mean'
distinction concrete.

Usage:
    python viz/viz_robot_scenario.py --case-dir <case> \\
        --rl-package-path <rl-pkg> [--out scenario.png]
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-dir', required=True)
    p.add_argument('--rl-package-path', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    cfd_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (cfd_pkg, args.rl_package_path):
        if path not in sys.path:
            sys.path.insert(0, path)
    from cfd_library_loader import load_cfd_case
    from reinforcement_learning.envs.gas_source_env import GasSourceEnv

    case = load_cfd_case(args.case_dir, args.rl_package_path)
    md, wf = case['map_data'], case['wind_field']

    env = GasSourceEnv()
    obs, info = env.reset(seed=args.seed,
                          options={'map_data': md, 'wind_field': wf})

    occ = md['grid'].grid
    cell = md['grid'].resolution
    H, W = occ.shape
    src = env._source_pos
    robot = env._robot_pos

    # Spatial wind field (drives advection)
    field = wf.field
    speed = np.linalg.norm(field, axis=-1)

    # What the policy observes: get_local_wind at robot (decoded back to m/s)
    enc = env._wind.get_local_wind(robot)          # in [0,1], (Ux,Uy)
    ms = env._wind._max_speed_uniform
    obs_uv = (np.array(enc) * 2.0 - 1.0) * ms       # decode to m/s
    sm_speed, sm_dir = wf.spatial_mean()

    # Gas plume after warmup
    try:
        fdict = env._plume.get_all_filaments()
        filaments = np.asarray(fdict['positions']) if isinstance(fdict, dict) else None
    except Exception:
        filaments = None

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # --- Left: spatial wind field (what drives the plume) ---
    ax = axes[0]
    ax.imshow(occ, origin='lower', extent=(0, W*cell, 0, H*cell),
              cmap='Greys', alpha=0.7, vmin=0, vmax=2)
    im = ax.imshow(speed, origin='lower', extent=(0, W*cell, 0, H*cell),
                   cmap='viridis', alpha=0.45, vmin=0, vmax=max(0.1, speed.max()))
    plt.colorbar(im, ax=ax, label='|U| [m/s]')
    skip = max(1, H // 25)
    xs = (np.arange(W) + 0.5) * cell
    ys = (np.arange(H) + 0.5) * cell
    Xs, Ys = np.meshgrid(xs[::skip], ys[::skip], indexing='xy')
    ax.quiver(Xs, Ys, field[::skip, ::skip, 0], field[::skip, ::skip, 1],
              color='white', alpha=0.8, scale=20, width=0.004)
    ax.plot(*src, 'r*', ms=22, label='source')
    ax.plot(*robot, 'co', ms=14, mec='k', label='robot')
    ax.set_title(f"SPATIAL wind field (drives plume advection)\n"
                 f"mean|U|={speed[occ==0].mean():.2f}  max={speed.max():.2f} m/s",
                 fontsize=12)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    ax.legend(loc='upper right')

    # --- Right: what the POLICY observes ---
    ax = axes[1]
    ax.imshow(occ, origin='lower', extent=(0, W*cell, 0, H*cell),
              cmap='Greys', alpha=0.7, vmin=0, vmax=2)
    if filaments is not None and len(filaments) > 0:
        fil = np.asarray(filaments)
        ax.scatter(fil[:, 0], fil[:, 1], s=6, c='limegreen', alpha=0.5,
                   label=f'gas filaments (n={len(fil)})')
    ax.plot(*src, 'r*', ms=22, label='source')
    ax.plot(*robot, 'co', ms=14, mec='k', label='robot')
    # The single wind vector the policy sees, drawn at robot
    obs_speed = np.linalg.norm(obs_uv)
    arrow_scale = max(W, H) * 0.12
    ax.annotate('', xy=(robot[0] + obs_uv[0]*arrow_scale,
                        robot[1] + obs_uv[1]*arrow_scale),
                xytext=(robot[0], robot[1]),
                arrowprops=dict(facecolor='orange', edgecolor='k', width=4,
                                headwidth=16))
    ax.set_title(f"What the POLICY observes\n"
                 f"single wind vector = spatial MEAN: "
                 f"speed={sm_speed:.2f} m/s @ {np.rad2deg(sm_dir):+.0f}°  "
                 f"(constant everywhere, all episode)", fontsize=12)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    ax.legend(loc='upper right')

    fig.suptitle(f"{Path(args.case_dir).name}  |  template={case['meta']['template_id']}  "
                 f"inlet={case['meta']['inlet_speed']:.2f} m/s  "
                 f"map={md['width']:.1f}×{md['height']:.1f} m", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out or str(Path(args.case_dir) / 'robot_scenario.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    print(f"Saved {out}")
    print(f"Spatial field: mean|U|={speed[occ==0].mean():.3f}, max={speed.max():.3f}")
    print(f"Policy observes: speed={sm_speed:.3f} m/s @ {np.rad2deg(sm_dir):+.1f}° (the mean)")
    print(f"  decoded obs wind vector at robot: ({obs_uv[0]:+.3f}, {obs_uv[1]:+.3f}) m/s")


if __name__ == '__main__':
    main()
