"""Load CFD library cases as (map_data, wind_field) ready to pass into
GasSourceEnv.reset(options=...).

The training-side wrapper (CFDLibrarySampler) samples a random valid case
at every env reset, so the policy sees a different (geometry, wind) pair
each episode instead of MapGenerator+synthetic-wind.

Interface mirrors reinforcement_learning.test.gaden_loader so the env code
needs no changes — just pass `options={"map_data": ..., "wind_field": ...}`
to reset().

Usage (training-side):
    from cfd_wind_pipeline.cfd_library_loader import CFDLibrarySampler
    sampler = CFDLibrarySampler(library_dir, rng, rl_package_path=...)
    obs, _ = env.reset(options=sampler.sample())

Usage (one-shot inspection):
    case = load_cfd_case(case_dir, rl_package_path=...)
    print(case['map_data']['source_pos'], case['wind_field'].max_speed())
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _import_rl_modules(rl_package_path: str):
    """Lazy import the env modules we need; lets this file be used without
    the RL package on PYTHONPATH (e.g. by callers in other modules)."""
    if rl_package_path not in sys.path:
        sys.path.insert(0, rl_package_path)
    from reinforcement_learning.envs.occupancy_grid import OccupancyGrid
    from reinforcement_learning.test.gaden_loader import GadenWindField
    return OccupancyGrid, GadenWindField


def load_cfd_case(case_dir: str | Path, rl_package_path: str) -> dict:
    """Load a single CFD library case as map_data + wind_field.

    Returns
    -------
    dict with keys::

        map_data:    {grid, source_pos, robot_pos, width, height}
                     The format GasSourceEnv.reset(options["map_data"]) expects.
        wind_field:  GadenWindField (interface-compatible; spatially-varying CFD wind)
        case_dir:    Path
        meta:        full meta.json contents (template_id, seed, openings, ...)
    """
    OccupancyGrid, GadenWindField = _import_rl_modules(rl_package_path)

    case = Path(case_dir)
    if not (case / 'wind_field.npz').exists():
        raise FileNotFoundError(f"{case}/wind_field.npz not found (case not complete)")

    meta = json.loads((case / 'meta.json').read_text())
    g = np.load(case / 'grid.npz')
    grid_arr = g['grid']
    cell_size = float(g['cell_size'])
    map_w = float(g['map_width'])
    map_h = float(g['map_height'])

    # Reconstruct an OccupancyGrid object with the loaded grid array.
    occ = OccupancyGrid(map_w, map_h, cell_size)
    occ.grid = grid_arr.astype(np.int8)
    occ.grid_height, occ.grid_width = grid_arr.shape

    # Wind field — same shape and interface as GADEN.
    w = np.load(case / 'wind_field.npz')
    field = w['field'].astype(np.float64)
    wind = GadenWindField(field=field, resolution=cell_size,
                          occupancy=(grid_arr != 0))

    map_data = {
        'grid': occ,
        'source_pos': tuple(meta['source_pos']),
        'robot_pos': tuple(meta['robot_pos']),
        'width': map_w,
        'height': map_h,
    }
    return {'map_data': map_data, 'wind_field': wind, 'case_dir': case, 'meta': meta}


class CFDLibrarySampler:
    """Yield a random complete case at each call to sample().

    Filters out cases that are missing wind_field.npz or are flagged
    degenerate at construction time so training never sees junk wind.
    """

    def __init__(self, library_dir: str | Path, rng: np.random.Generator,
                 rl_package_path: str,
                 reject_degenerate: bool = True,
                 min_speed: float = 0.05,
                 min_speed_std: float = 0.02,
                 min_circ_std: float = 0.3):
        """Scan the library and cache the valid case dirs.

        Filters mirror cfd_wind_pipeline.library_stats.is_degenerate:
        - mean |U| >= min_speed
        - std |U| >= min_speed_std
        - circular std >= min_circ_std (rules out single-direction stagnation)
        """
        self._rng = rng
        self._rl_pkg = rl_package_path
        lib = Path(library_dir)
        manifest_path = lib / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json at {lib}")
        manifest = json.loads(manifest_path.read_text())

        self._cases = []
        n_skipped = 0
        for entry in manifest:
            # Manifests store absolute paths (legacy) — but a library can be
            # moved/renamed. Always resolve relative to manifest location.
            cd = lib / Path(entry['case_dir']).name
            wf_path = cd / 'wind_field.npz'
            if not wf_path.exists():
                n_skipped += 1
                continue
            if reject_degenerate:
                d = np.load(wf_path)
                g = np.load(cd / 'grid.npz')['grid']
                field = d['field']
                speeds = np.linalg.norm(field, axis=-1)
                free = (g == 0)
                s = speeds[free]
                if s.size == 0 or s.mean() < min_speed or s.std() < min_speed_std:
                    n_skipped += 1
                    continue
                dirs = np.arctan2(field[..., 1], field[..., 0])[free]
                R = np.sqrt(np.mean(np.sin(dirs))**2 + np.mean(np.cos(dirs))**2)
                circ = np.sqrt(-2*np.log(R)) if R > 1e-8 else float('inf')
                if circ < min_circ_std:
                    n_skipped += 1
                    continue
            self._cases.append(cd)
        print(f"[CFDLibrarySampler] {library_dir}: kept {len(self._cases)} cases "
              f"({n_skipped} skipped — incomplete or degenerate)")
        if not self._cases:
            raise RuntimeError(f"No valid cases in {library_dir}")

    def __len__(self) -> int:
        return len(self._cases)

    def sample(self) -> dict:
        """Return {map_data, wind_field} suitable for env.reset(options=...)."""
        cd = self._cases[self._rng.integers(0, len(self._cases))]
        case = load_cfd_case(cd, self._rl_pkg)
        # Drop helper fields so it's pure reset-options
        return {'map_data': case['map_data'], 'wind_field': case['wind_field']}
