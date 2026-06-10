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

    def __init__(self, library_dirs, rng: np.random.Generator,
                 rl_package_path: str,
                 reject_degenerate: bool = True,
                 min_speed: float = 0.05,
                 min_speed_std: float = 0.02,
                 min_circ_std: float = 0.3,
                 max_speed_cap: float = 10.0,
                 template_filter=None,
                 far_plume_frac: float = 0.0,
                 far_plume_range=(4.0, 14.0),
                 far_plume_clearance: float = 0.6,
                 far_plume_min_src_dist: float = 3.0):
        """Scan one or more libraries and cache the valid case dirs.

        library_dirs : str | Path | list
            A single library dir, or a list to pool multiple libraries
            (e.g. easy T0-3 + hard T4-9) into one sampling distribution.
        template_filter : iterable of int, optional
            If given, keep only cases whose template_id is in this set.
            Use to restrict to the champ's known templates (0-5) and avoid
            OOD template shock during finetuning.
        far_plume_frac : float in [0,1]
            Fraction of sampled cases whose robot start is RE-PLACED at a
            far-plume-hit cell (forces explore-to-find-plume search). The
            rest keep the manifest's original (near) robot_pos so the policy
            retains its plume-tracking skill. 0 = behave exactly as before.
        far_plume_range : (lo, hi) meters
            Target plume-hit-distance band for re-placed starts (see
            far_plume_placement.FarPlumePlacer).
        far_plume_clearance : float
            Wall-clearance margin (m) required at the re-placed start.
        far_plume_min_src_dist : float
            Never re-place closer than this to the source.

        Degenerate filters mirror library_stats.is_degenerate:
        mean |U| >= min_speed, std |U| >= min_speed_std,
        circular std >= min_circ_std (rules out single-direction stagnation).
        """
        self._rng = rng
        self._rl_pkg = rl_package_path
        self._far_frac = float(far_plume_frac)
        self._far_range = tuple(far_plume_range)
        self._far_clearance = float(far_plume_clearance)
        self._far_min_src = float(far_plume_min_src_dist)
        self._placer_cache = {}  # case_dir -> FarPlumePlacer (lazy)
        # Reverse curriculum (OSL_REVERSE_CURRICULUM=1): the far-plume start band
        # anneals from NEAR the source to the full far_plume_range over training,
        # so the policy first learns the (solvable) near-source approach then must
        # cross progressively farther — matching the measured 2m-solves / 3m-fails
        # cliff on many_rooms. Progress is approximated by this sampler's own
        # sample() call count vs OSL_RC_TOTAL_SAMPLES (per-worker). near band is
        # OSL_RC_NEAR_LO..OSL_RC_NEAR_HI; final band is far_plume_range; the lo/hi
        # interpolate linearly with progress over the first OSL_RC_FRACTION of run.
        import os as _os
        self._rc_on = _os.environ.get("OSL_REVERSE_CURRICULUM", "0") == "1"
        self._rc_near = (float(_os.environ.get("OSL_RC_NEAR_LO", "1.0")),
                         float(_os.environ.get("OSL_RC_NEAR_HI", "3.0")))
        self._rc_total = float(_os.environ.get("OSL_RC_TOTAL_SAMPLES", "0"))
        self._rc_fraction = float(_os.environ.get("OSL_RC_FRACTION", "0.6"))
        self._rc_calls = 0
        self._rc_cur_range = None  # last band the placers were built for
        if isinstance(library_dirs, (str, Path)):
            library_dirs = [library_dirs]
        tmpl_set = set(template_filter) if template_filter is not None else None

        self._cases = []
        n_skipped = 0
        n_tmpl_filtered = 0
        for library_dir in library_dirs:
            lib = Path(library_dir)
            manifest_path = lib / 'manifest.json'
            if not manifest_path.exists():
                raise FileNotFoundError(f"No manifest.json at {lib}")
            manifest = json.loads(manifest_path.read_text())
            kept_here = 0
            for entry in manifest:
                if tmpl_set is not None and entry.get('template_id') not in tmpl_set:
                    n_tmpl_filtered += 1
                    continue
                # Resolve relative to manifest location (libraries may be moved).
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
                    # Reject CFD solver artifacts: a few cases have a handful of
                    # blow-up cells (max-speed up to 9e9 m/s) that make the plume
                    # tunnel-check allocate a 500GB array → OOM. Real wind here is
                    # <5 m/s (p99 ~4); cap well above that.
                    if not np.isfinite(speeds).all() or speeds.max() > max_speed_cap:
                        n_skipped += 1
                        continue
                    dirs = np.arctan2(field[..., 1], field[..., 0])[free]
                    R = np.sqrt(np.mean(np.sin(dirs))**2 + np.mean(np.cos(dirs))**2)
                    circ = np.sqrt(-2*np.log(R)) if R > 1e-8 else float('inf')
                    if circ < min_circ_std:
                        n_skipped += 1
                        continue
                self._cases.append(cd)
                kept_here += 1
            print(f"[CFDLibrarySampler] {library_dir}: kept {kept_here} cases")
        msg = (f"[CFDLibrarySampler] TOTAL kept {len(self._cases)} "
               f"({n_skipped} incomplete/degenerate")
        if tmpl_set is not None:
            msg += f", {n_tmpl_filtered} template-filtered (keep {sorted(tmpl_set)})"
        print(msg + ")")
        if not self._cases:
            raise RuntimeError(f"No valid cases in {library_dirs}")

    def __len__(self) -> int:
        return len(self._cases)

    def _current_range(self):
        """The far-plume distance band for the current curriculum progress.
        Without reverse curriculum, this is just the fixed far_plume_range."""
        if not self._rc_on or self._rc_total <= 0:
            return self._far_range
        # progress in [0,1] over the first rc_fraction of the run, then clamped.
        prog = self._rc_calls / (self._rc_total * max(self._rc_fraction, 1e-6))
        prog = min(1.0, max(0.0, prog))
        lo = self._rc_near[0] + prog * (self._far_range[0] - self._rc_near[0])
        hi = self._rc_near[1] + prog * (self._far_range[1] - self._rc_near[1])
        return (lo, hi)

    def _get_placer(self, cd: Path):
        """Lazily build + cache the FarPlumePlacer for one case (uses the
        on-disk far_placement.json computed by far_plume_placement). Under the
        reverse curriculum the band changes over time, so the cache is keyed by
        (case, rounded-band) and stale-band placers are simply rebuilt."""
        cur = self._current_range()
        key = (str(cd), round(cur[0], 2), round(cur[1], 2))
        placer = self._placer_cache.get(key)
        if placer is not None:
            return placer
        from far_plume_placement import compute_case_placements, FarPlumePlacer
        placements = compute_case_placements(
            cd, self._rl_pkg, clearance=self._far_clearance)
        placer = FarPlumePlacer(
            placements, self._rng,
            target_phd_range=cur,
            min_source_dist=self._far_min_src)
        self._placer_cache[key] = placer
        return placer

    def sample(self) -> dict:
        """Return {map_data, wind_field} suitable for env.reset(options=...).

        If far_plume_frac > 0, that fraction of samples gets its robot start
        RE-PLACED at a far-plume-hit cell (search scenario); the rest keep the
        original near placement.
        """
        self._rc_calls += 1
        cd = self._cases[self._rng.integers(0, len(self._cases))]
        case = load_cfd_case(cd, self._rl_pkg)
        map_data = case['map_data']
        if self._far_frac > 0.0 and self._rng.random() < self._far_frac:
            placer = self._get_placer(cd)
            if placer.usable:
                start = placer.pick()
                if start is not None:
                    # Copy so we don't mutate any shared map_data dict.
                    map_data = {**map_data, 'robot_pos': start}
        # Drop helper fields so it's pure reset-options
        return {'map_data': map_data, 'wind_field': case['wind_field']}
