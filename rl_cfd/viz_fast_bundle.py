"""Visualise the fast-bundle wind field + plume on the training templates.

Produces a 6-panel comparison per template:
  Top row:    [walls + uniform wind quiver] [walls + spatial wind quiver] [streamlines]
  Bottom row: [uniform plume after N steps] [spatial plume after N steps] [gas concentration heatmap]

Output: /tmp/fast_bundle_viz/T{template_id}.png
"""
import sys, os, importlib.util, types, time
sys.path.insert(0, '/home/efe/ros2_ws')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def _load_baseline():
    """Load test_rl as 'test_rl_baseline' so it doesn't clash with test_rl_fast."""
    pkg = types.ModuleType('test_rl_baseline'); pkg.__path__ = ['/home/efe/ros2_ws/test_rl']
    sys.modules['test_rl_baseline'] = pkg
    pkg_envs = types.ModuleType('test_rl_baseline.envs')
    pkg_envs.__path__ = ['/home/efe/ros2_ws/test_rl/envs']
    sys.modules['test_rl_baseline.envs'] = pkg_envs

    for mod in ['config', 'envs.occupancy_grid', 'envs.map_generator',
                'envs.lidar_sim', 'envs.wind_model', 'envs.sensor_model',
                'envs.igdm_model', 'envs.filament_plume', 'envs.visualizer',
                'envs.gas_source_env', 'envs.spatial_obs_wrapper']:
        path = f'/home/efe/ros2_ws/test_rl/{mod.replace(".", "/")}.py'
        spec = importlib.util.spec_from_file_location(f'test_rl_baseline.{mod}', path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[f'test_rl_baseline.{mod}'] = m
        spec.loader.exec_module(m)
    return sys.modules['test_rl_baseline.envs.gas_source_env'].GasSourceEnv


def render_template(template_id: int, seed: int = 0,
                    n_warmup: int = 60, out_dir: str = '/tmp/fast_bundle_viz'):
    os.makedirs(out_dir, exist_ok=True)

    # Build both envs at the same seed so they get identical map + wind sample
    BaselineEnv = _load_baseline()

    from test_rl_fast.envs import GasSourceEnv as FastEnv

    env_b = BaselineEnv(seed=seed, template_id=template_id)
    env_f = FastEnv(seed=seed, template_id=template_id)
    env_b.reset(seed=seed)
    env_f.reset(seed=seed)

    grid = env_f._grid.grid
    res  = env_f._grid.resolution
    H, W = grid.shape
    extent = [0, W * res, 0, H * res]
    walls = grid != 0

    # Run plumes for warmup steps to populate filaments
    for _ in range(n_warmup):
        env_b._plume.update()
        env_f._plume.update()
    fil_b = env_b._plume.get_all_filaments()['positions']
    fil_f = env_f._plume.get_all_filaments()['positions']

    # ---- Build figure ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Template {template_id}  ·  room {env_f._map_width:.1f}×{env_f._map_height:.1f} m  "
        f"·  sampled wind: {env_f._wind.speed:.2f} m/s @ {np.degrees(env_f._wind.direction):.0f}°  "
        f"·  source ✦ at ({env_f._source_pos[0]:.1f}, {env_f._source_pos[1]:.1f})",
        fontsize=13)

    def base_panel(ax, title):
        ax.imshow(walls, origin='lower', extent=extent, cmap='Greys', alpha=0.4)
        ax.set_xlim(0, W * res); ax.set_ylim(0, H * res)
        ax.set_aspect('equal')
        ax.scatter(*env_f._source_pos, marker='*', c='red', s=180, zorder=5,
                   edgecolors='black', linewidths=0.6)
        ax.set_title(title, fontsize=11)

    # Top-left: uniform wind quiver
    ax = axes[0, 0]
    base_panel(ax, "Uniform wind (baseline)")
    s = env_b._wind.speed
    th = env_b._wind.direction
    Ux_unif = np.full_like(grid, s * np.cos(th), dtype=float)
    Uy_unif = np.full_like(grid, s * np.sin(th), dtype=float)
    Ux_unif[walls] = 0; Uy_unif[walls] = 0
    skip = 6
    xs = (np.arange(W) + 0.5) * res
    ys = (np.arange(H) + 0.5) * res
    ax.quiver(xs[::skip], ys[::skip],
              Ux_unif[::skip, ::skip], Uy_unif[::skip, ::skip],
              color='steelblue', scale=20, width=0.0035)

    # Top-middle: potential-flow wind quiver
    ax = axes[0, 1]
    base_panel(ax, "Spatial wind (potential flow)")
    Ux = env_f._wind.Ux; Uy = env_f._wind.Uy
    speed = np.hypot(Ux, Uy)
    speed_max = max(speed[~walls].max(), 1e-6)
    ax.quiver(xs[::skip], ys[::skip],
              Ux[::skip, ::skip], Uy[::skip, ::skip],
              speed[::skip, ::skip] / speed_max,
              cmap='viridis', scale=20, width=0.0035)

    # Top-right: streamlines (only spatial — uniform is trivial)
    ax = axes[0, 2]
    base_panel(ax, "Spatial wind streamlines")
    XS, YS = np.meshgrid(xs, ys)
    Ux_plot = np.where(walls, np.nan, Ux)
    Uy_plot = np.where(walls, np.nan, Uy)
    try:
        ax.streamplot(XS, YS, Ux_plot, Uy_plot, density=1.4,
                      color=speed, cmap='viridis', linewidth=0.9,
                      arrowsize=0.8)
    except Exception:
        pass

    # Bottom-left: uniform-wind plume filaments
    ax = axes[1, 0]
    base_panel(ax, f"Baseline plume — {len(fil_b)} filaments")
    if len(fil_b) > 0:
        ax.scatter(fil_b[:, 0], fil_b[:, 1], s=4, c='steelblue', alpha=0.4)

    # Bottom-middle: spatial-wind plume filaments
    ax = axes[1, 1]
    base_panel(ax, f"Fast bundle plume — {len(fil_f)} filaments")
    if len(fil_f) > 0:
        ax.scatter(fil_f[:, 0], fil_f[:, 1], s=4, c='darkorange', alpha=0.5)

    # Bottom-right: gas concentration heatmap from spatial plume
    ax = axes[1, 2]
    base_panel(ax, "Spatial-wind gas concentration")
    qx = np.linspace(0.1, W * res - 0.1, 80)
    qy = np.linspace(0.1, H * res - 0.1, 60)
    QX, QY = np.meshgrid(qx, qy)
    conc = np.zeros_like(QX)
    for i in range(QY.shape[0]):
        for j in range(QY.shape[1]):
            conc[i, j] = env_f._plume.concentration_at((QX[i, j], QY[i, j]))
    if conc.max() > 0:
        ax.imshow(conc, origin='lower', extent=extent, cmap='Reds',
                  alpha=0.7, norm=mcolors.PowerNorm(gamma=0.5,
                                                     vmin=0, vmax=conc.max()))

    plt.tight_layout()
    out = os.path.join(out_dir, f'T{template_id}.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  T{template_id}: {len(fil_f)} filaments (fast)  vs  '
          f'{len(fil_b)} (baseline)  →  {out}')


if __name__ == '__main__':
    print("Rendering fast-bundle visualisations...")
    t0 = time.time()
    for tid in [0, 1, 2, 3, 4, 5]:
        render_template(tid, seed=42, n_warmup=80)
    print(f"Done in {time.time()-t0:.1f}s. Output: /tmp/fast_bundle_viz/")
