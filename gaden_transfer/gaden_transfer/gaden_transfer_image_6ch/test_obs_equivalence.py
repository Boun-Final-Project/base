#!/usr/bin/env python3
"""
Equivalence test: feed the same fake env state into rl_osl's SpatialObsWrapper
(training side) and our deployment SpatialObsBuilder (this folder), then diff
each channel of the spatial observation + the ctx vector.

The deployment-side dense_wall and 0.2 m wrapper resolution should produce
bit-exact output vs the training wrapper. If anything differs, we've drifted.

Also dumps a side-by-side PNG per scenario step so you can eyeball them.

Usage:
    python3 test_obs_equivalence.py                 # default scenarios
    python3 test_obs_equivalence.py --templates 2 5 # only u_shape + multi_room
    python3 test_obs_equivalence.py --verbose       # print per-channel diffs
"""

import argparse
import importlib.util
import os
import sys
import types

import numpy as np


# --------------------------------------------------------------------------
# Loader for rl_osl (training side) — bypasses gymnasium by stubbing the env
# --------------------------------------------------------------------------
def load_rl_osl(rl_osl_root='/home/efe/ros2_ws/rl_osl'):
    """Load rl_osl modules without triggering gymnasium imports.

    rl_osl.envs.__init__ imports gas_source_env (needs gymnasium). We bypass
    by loading only the modules we actually call: config, occupancy_grid,
    map_generator, spatial_obs_wrapper.
    """
    if not os.path.isdir(rl_osl_root):
        raise FileNotFoundError(f'rl_osl root not found: {rl_osl_root}')

    # Synthetic top-level package so relative imports inside the modules work.
    rl_osl = types.ModuleType('rl_osl_test')
    rl_osl.__path__ = [rl_osl_root]
    sys.modules['rl_osl_test'] = rl_osl
    envs = types.ModuleType('rl_osl_test.envs')
    envs.__path__ = [os.path.join(rl_osl_root, 'envs')]
    sys.modules['rl_osl_test.envs'] = envs

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(f'rl_osl_test.{name}', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f'rl_osl_test.{name}'] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg     = _load('config',                    f'{rl_osl_root}/config.py')
    occ_mod = _load('envs.occupancy_grid',       f'{rl_osl_root}/envs/occupancy_grid.py')
    mg_mod  = _load('envs.map_generator',        f'{rl_osl_root}/envs/map_generator.py')
    sow_mod = _load('envs.spatial_obs_wrapper',  f'{rl_osl_root}/envs/spatial_obs_wrapper.py')
    return cfg, occ_mod.OccupancyGrid, mg_mod.MapGenerator, sow_mod.SpatialObsWrapper


# --------------------------------------------------------------------------
# Loader for the deployment-side builder
# --------------------------------------------------------------------------
def load_deployment_builder():
    src_base = '/home/efe/ros2_ws/src/base'
    if src_base not in sys.path:
        sys.path.insert(0, src_base)
    from gaden_transfer.gaden_transfer_image_6ch.spatial_obs_builder import (
        SpatialObsBuilder,
    )
    return SpatialObsBuilder


# --------------------------------------------------------------------------
# Fake env: minimal interface SpatialObsWrapper needs
# --------------------------------------------------------------------------
class _FakeLidar:
    def __init__(self, grid, n_rays, max_range):
        self._grid = grid
        self._n_rays = n_rays
        self._max_range = max_range
        step_res = grid.resolution / 2.0
        self._n_steps = int(np.ceil(max_range / step_res))
        t = np.arange(1, self._n_steps + 1) * step_res
        ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        self._dx = np.cos(ang)[:, None] * t[None, :]
        self._dy = np.sin(ang)[:, None] * t[None, :]

    def scan(self, position):  # not used by reveal but kept for API
        return np.ones(self._n_rays)


class _FakeWind:
    def __init__(self, speed=0.5, direction=1.2, max_speed=2.0):
        self.speed = speed
        self.direction = direction
        self.max_speed = max_speed

    def get_observation_spatial(self):
        return (
            self.speed / self.max_speed,
            float(np.cos(self.direction)),
            float(np.sin(self.direction)),
        )


class _FakeEnv:
    def __init__(self, mapdict, wind):
        self._grid = mapdict['grid']
        self._map_width = mapdict['width']
        self._map_height = mapdict['height']
        self._robot_pos = np.array(mapdict['robot_pos'], dtype=float)
        self._source_pos = np.array(mapdict['source_pos'], dtype=float)
        self._current_step = 0
        self._wind = wind
        self._lidar = _FakeLidar(self._grid, n_rays=72, max_range=3.0)


# --------------------------------------------------------------------------
# Adapter to wrap a training OccupancyGrid as the deployment's `occ_map`
# --------------------------------------------------------------------------
class _OccMapAdapter:
    """Quack-types as efe_igdm.mapping.OccupancyGridMap."""
    def __init__(self, training_grid):
        self.grid = training_grid.grid
        self.resolution = training_grid.resolution
        self.grid_width = training_grid.grid_width
        self.grid_height = training_grid.grid_height
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.real_world_width = training_grid.grid_width * training_grid.resolution
        self.real_world_height = training_grid.grid_height * training_grid.resolution


# --------------------------------------------------------------------------
# Core: run wrapper + builder on the same fake env steps, diff outputs
# --------------------------------------------------------------------------
def run_one_scenario(template_id, seed, steps_to_dump, *, verbose=False,
                     out_dir='/tmp/osl_obs_equivalence'):
    cfg, OccupancyGrid, MapGenerator, SpatialObsWrapper = load_rl_osl()
    SpatialObsBuilder = load_deployment_builder()

    rng = np.random.default_rng(seed)
    gen = MapGenerator(rng=rng)
    mapdict = gen.generate(template_id=template_id)

    wind = _FakeWind()
    fake_env = _FakeEnv(mapdict, wind)

    # ----- training-side wrapper: manual reset (skip env.reset) -----
    wrap = SpatialObsWrapper(fake_env)
    h = int(np.floor(fake_env._map_height / wrap.CELL_RES))
    w = int(np.floor(fake_env._map_width / wrap.CELL_RES))
    wrap._map_h_cells = h
    wrap._map_w_cells = w
    for a in ('_known_world', '_wall_world', '_gas_world',
              '_rec_world', '_det_world'):
        setattr(wrap, a, np.zeros((h, w), dtype=np.float32))
    wrap._max_det = 0
    true_res = fake_env._grid.resolution
    n_steps_t = int(np.ceil(wrap._reveal_radius_m / true_res))
    wrap._ray_samples_t = np.arange(1, n_steps_t + 1) * true_res
    # rl_osl's wrapper precomputes _dense_wall in its reset. Replicate:
    sub = int(round(wrap.CELL_RES / true_res))
    gt = fake_env._grid.grid
    gh, gw = gt.shape
    gh_pad = ((gh + sub - 1) // sub) * sub
    gw_pad = ((gw + sub - 1) // sub) * sub
    padded = np.zeros((gh_pad, gw_pad), dtype=gt.dtype)
    padded[:gh, :gw] = gt
    pooled = padded.reshape(gh_pad // sub, sub, gw_pad // sub, sub).max(axis=(1, 3))
    wrap._dense_wall = (pooled[:h, :w] != 0)

    # ----- deployment-side builder -----
    occ_adapter = _OccMapAdapter(fake_env._grid)
    builder = SpatialObsBuilder(
        occ_adapter,
        fake_env._map_width,
        fake_env._map_height,
    )
    # Lock wind context to the same value as the wrapper's get_observation_spatial
    # would produce (the wrapper appends time_frac itself in _get_obs).
    builder._locked_wind_speed = wind.speed
    builder._locked_wind_dir = wind.direction
    builder._wind_locked = True
    builder._initialized = True

    # ----- shared "reset": initial reveal + mark cell with binary=0 -----
    pos = fake_env._robot_pos.copy()
    wrap._reveal(pos)
    wrap._update_robot_cell(pos, binary=0)
    builder.robot_x, builder.robot_y = pos[0], pos[1]
    builder.record_step()  # equivalent to wrapper.reset (seeds grids)

    src = fake_env._source_pos
    rng_walk = np.random.default_rng(seed + 99)

    diffs = []
    os.makedirs(out_dir, exist_ok=True)

    for step in range(1, max(steps_to_dump) + 1):
        # Same fake walk for both sides
        vec = src - pos
        d = np.linalg.norm(vec)
        if d < 0.3:
            break
        unit = vec / d
        jitter = rng_walk.normal(0, 0.3, size=2)
        direction = unit + jitter
        direction /= (np.linalg.norm(direction) + 1e-6)
        next_pos = pos + direction * cfg.STEP_SIZE
        if fake_env._grid.is_valid(position=tuple(next_pos), radius=cfg.ROBOT_RADIUS):
            pos = next_pos
        fake_env._robot_pos = pos
        fake_env._current_step = step
        binary = 1 if np.linalg.norm(pos - src) < 2.5 else 0

        # Training-side step
        wrap._rec_world *= wrap._decay
        wrap._reveal(pos)
        wrap._update_robot_cell(pos, binary)
        sp_t, ctx_t = wrap._get_obs()

        # Deployment-side step
        builder.robot_x, builder.robot_y = pos[0], pos[1]
        builder._latest_binary = binary
        builder.record_step()
        out = builder.build()
        if out is None:
            print(f'  step {step}: builder not ready')
            continue
        sp_d, ctx_d = out

        # Diff
        if step in steps_to_dump:
            ch_max_diffs = []
            for i in range(5):
                d_max = float(np.max(np.abs(sp_t[i] - sp_d[i])))
                ch_max_diffs.append(d_max)
            ctx_max = float(np.max(np.abs(ctx_t - ctx_d)))
            diffs.append({
                'template': template_id, 'seed': seed, 'step': step,
                'pos': tuple(pos), 'd2src': float(d), 'binary': binary,
                'channels': ch_max_diffs, 'ctx': ctx_max,
            })

            if verbose:
                names = ['is_known', 'is_wall', 'gas', 'recency', 'det_count']
                msg = ' '.join(f'{n}={v:.3g}' for n, v in zip(names, ch_max_diffs))
                print(f'  step {step}: {msg}  ctx_max={ctx_max:.3g}')

            # Side-by-side PNG (quick visual confirmation)
            _save_side_by_side(template_id, seed, step, pos, src, sp_t, sp_d,
                               fake_env._grid, out_dir)

    return diffs


def _save_side_by_side(tpl, seed, step, pos, src, sp_t, sp_d, gt_grid, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ['is_known', 'is_wall', 'gas', 'recency', 'det_count']
    fig, axes = plt.subplots(3, 5, figsize=(18, 9.5))
    for i, name in enumerate(names):
        for row, (data, label) in enumerate([(sp_t[i], 'TRAIN'), (sp_d[i], 'DEPLOY')]):
            ax = axes[row, i]
            vmin, vmax = (-1, 1) if name == 'gas' else (0, 1)
            im = ax.imshow(data, origin='lower', cmap='viridis',
                           vmin=vmin, vmax=vmax)
            ax.set_title(f'{label} {name}', fontsize=9)
            ax.axvline(49, color='r', lw=0.3, alpha=0.5)
            ax.axhline(49, color='r', lw=0.3, alpha=0.5)
        # diff
        ax = axes[2, i]
        diff = sp_t[i] - sp_d[i]
        m = float(max(1e-9, np.max(np.abs(diff))))
        ax.imshow(diff, origin='lower', cmap='RdBu_r', vmin=-m, vmax=m)
        ax.set_title(f'diff (max|·|={m:.2g})', fontsize=9)

    fig.suptitle(
        f'Template {tpl} seed {seed} step {step}  '
        f'pos=({pos[0]:.2f},{pos[1]:.2f})  src=({src[0]:.2f},{src[1]:.2f})',
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fn = os.path.join(out_dir, f'tpl{tpl}_seed{seed}_step{step:03d}.png')
    fig.savefig(fn, dpi=80)
    plt.close(fig)
    return fn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--templates', type=int, nargs='+',
                   default=[2, 3, 5],
                   help='template ids to test (0..5)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, nargs='+',
                   default=[1, 20, 50, 100],
                   help='step indices to dump and compare')
    p.add_argument('--out-dir', type=str, default='/tmp/osl_obs_equivalence')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    print('=== Deployment ↔ training-wrapper equivalence test ===')
    print(f'templates: {args.templates}    seed: {args.seed}    '
          f'steps: {args.steps}    out: {args.out_dir}')
    print()

    overall_ok = True
    for tpl in args.templates:
        print(f'--- Template {tpl} ---')
        try:
            diffs = run_one_scenario(
                tpl, args.seed, args.steps,
                verbose=args.verbose, out_dir=args.out_dir,
            )
        except Exception as e:
            print(f'  ERROR: {e}')
            overall_ok = False
            continue
        if not diffs:
            print('  (no steps dumped — robot reached source quickly)')
            continue
        max_per_channel = [
            max(d['channels'][i] for d in diffs) for i in range(5)
        ]
        max_ctx = max(d['ctx'] for d in diffs)
        names = ['is_known', 'is_wall', 'gas', 'recency', 'det_count']
        for n, m in zip(names, max_per_channel):
            tag = 'OK' if m < 1e-6 else 'DIFF'
            print(f'  {n:11s}  max|·|={m:.3g}  [{tag}]')
            if m >= 1e-6:
                overall_ok = False
        tag = 'OK' if max_ctx < 1e-6 else 'DIFF'
        print(f'  ctx          max|·|={max_ctx:.3g}  [{tag}]')
        if max_ctx >= 1e-6:
            overall_ok = False
        print(f'  ({len(diffs)} step snapshots dumped to {args.out_dir})')
        print()

    print('=== Overall:', 'EQUIVALENT' if overall_ok else 'DIVERGED', '===')
    return 0 if overall_ok else 1


if __name__ == '__main__':
    sys.exit(main())
