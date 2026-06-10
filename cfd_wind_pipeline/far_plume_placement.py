"""Far-plume-hit robot re-placement for CFD library cases.

Idea: the CFD wind field + source are the expensive, fixed part of a case.
The robot's START cell is a free placement choice. To manufacture
ultimate-style SEARCH scenarios (robot must explore to FIND the plume before
it can track it), we re-place the robot at cells with large *plume-hit
distance* = distance to the nearest detectable-gas cell in the warmed-up
plume. This is the right hardness metric (NOT source distance): a robot far
from the source but sitting in a long plume tongue has an easy job, while a
robot in a gas-empty pocket must search regardless of source distance.

Per case we precompute (once, cached to disk) the list of valid candidate
start cells with their plume-hit distance. Valid = free AND >= clearance
margin from walls (spawn in open space, not jammed against a wall). The
source+wind+geometry are fixed per case, so the far-cell structure is stable
across resets; a single plume warmup captures it.

At reset, FarPlumePlacer.pick() samples a start cell to hit a target
plume-hit-distance band (near->far mix), or falls back to the manifest's
original robot_pos.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_VERSION = 2  # bump to invalidate stale caches when the recipe changes
GAS_DETECT_CONC = 0.05  # conc above which a cell counts as "in the plume"


def _import_rl(rl_package_path: str):
    if rl_package_path not in sys.path:
        sys.path.insert(0, rl_package_path)
    from reinforcement_learning.envs.filament_plume import FilamentPlume
    from reinforcement_learning.envs.wind_model import WindModel
    from reinforcement_learning import config as cfg
    return FilamentPlume, WindModel, cfg


def _warm_plume_gas_cells(case: dict, rl_package_path: str,
                          warmup_steps: Optional[int] = None) -> np.ndarray:
    """Warm up the case's plume and return the world-coords of free cells whose
    concentration exceeds GAS_DETECT_CONC. These are the "plume" cells the
    robot would detect. Uses the SAME wind/plume config as training so the
    measured plume matches what the policy will see."""
    FilamentPlume, WindModel, cfg = _import_rl(rl_package_path)
    md = case['map_data']
    wf = case['wind_field']
    grid = md['grid']
    cell = grid.resolution
    occ = grid.grid
    H, W = occ.shape

    speed, direction = wf.spatial_mean()
    plume = FilamentPlume(
        source_pos=np.array(md['source_pos'], dtype=np.float64),
        wind_speed=speed,
        wind_angle=direction,
        occupancy_grid=grid,
        dt=cfg.FILAMENT_DT,
        K=cfg.FILAMENT_K,
        turbulence_scale=cfg.FILAMENT_TURBULENCE_SCALE,
        max_age=cfg.FILAMENT_MAX_AGE,
        filaments_per_step=cfg.FILAMENTS_PER_STEP,
        initial_sigma=cfg.FILAMENT_INITIAL_SIGMA,
        mass=cfg.FILAMENT_MASS,
        min_sigma=cfg.FILAMENT_MIN_SIGMA,
        reflection_energy=cfg.FILAMENT_REFLECTION_ENERGY,
        rng=np.random.default_rng(0),  # deterministic plume for cache stability
        wind_field=wf,  # spatial CFD field drives advection
    )
    steps = warmup_steps if warmup_steps is not None else cfg.FILAMENT_WARMUP_STEPS
    # A longer warmup than the env's 15 gives a steadier plume footprint for a
    # stable cache; the robot still sees the normal 15-step warmup at runtime.
    for _ in range(max(steps, 40)):
        plume.update()

    ys, xs = np.where(occ == 0)
    pts = np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell], axis=1)
    conc = np.array([plume.concentration_at(p) for p in pts])
    return pts[conc > GAS_DETECT_CONC]


def compute_case_placements(case_dir: str | Path, rl_package_path: str,
                            clearance: float = 0.6,
                            recompute: bool = False) -> dict:
    """Compute (and cache) the ranked far-plume-hit candidate start cells for a
    single case. Returns dict with:
        cells:    (N,2) world coords of valid candidate starts
        phd:      (N,) plume-hit distance per cell
        src_dist: (N,) source distance per cell (for reference/logging)
        clearance, n_gas_cells
    Cached to <case_dir>/far_placement.json.
    """
    from scipy.spatial import cKDTree
    if rl_package_path not in sys.path:
        sys.path.insert(0, rl_package_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cfd_library_loader import load_cfd_case

    case_dir = Path(case_dir)
    cache = case_dir / 'far_placement.json'
    if cache.exists() and not recompute:
        d = json.loads(cache.read_text())
        if d.get('version') == CACHE_VERSION and abs(d.get('clearance', -1) - clearance) < 1e-6:
            return {
                'cells': np.array(d['cells'], dtype=np.float64).reshape(-1, 2),
                'phd': np.array(d['phd'], dtype=np.float64),
                'src_dist': np.array(d['src_dist'], dtype=np.float64),
                'clearance': clearance,
                'n_gas_cells': d.get('n_gas_cells', 0),
            }

    case = load_cfd_case(case_dir, rl_package_path)
    md = case['map_data']
    grid = md['grid']
    cell = grid.resolution
    occ = grid.grid
    src = np.array(md['source_pos'], dtype=np.float64)

    # Valid candidate cells: free AND >= clearance margin from walls.
    ys, xs = np.where(occ == 0)
    valid = np.array(
        [grid.is_valid(gx=int(gx), gy=int(gy), radius=clearance)
         for gx, gy in zip(xs, ys)],
        dtype=bool,
    )
    xs, ys = xs[valid], ys[valid]
    cells = np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell], axis=1)

    gas_pts = _warm_plume_gas_cells(case, rl_package_path)
    if len(gas_pts) > 0 and len(cells) > 0:
        tree = cKDTree(gas_pts)
        phd, _ = tree.query(cells)
    else:
        phd = np.full(len(cells), np.inf)
    src_dist = np.linalg.norm(cells - src, axis=1)

    out = {
        'cells': cells, 'phd': phd, 'src_dist': src_dist,
        'clearance': clearance, 'n_gas_cells': int(len(gas_pts)),
    }
    # Persist (finite only; cells with inf phd = no plume reached, still usable).
    cache.write_text(json.dumps({
        'version': CACHE_VERSION,
        'clearance': clearance,
        'n_gas_cells': int(len(gas_pts)),
        'cells': cells.tolist(),
        'phd': [None if not np.isfinite(v) else float(v) for v in phd],
        'src_dist': src_dist.tolist(),
    }))
    # JSON can't store inf; reload-path turns None->nan, so normalize here too.
    out['phd'] = np.where(np.isfinite(out['phd']), out['phd'], np.inf)
    return out


class FarPlumePlacer:
    """Samples a robot start cell for one case to hit a target plume-hit band.

    target_phd_range : (lo, hi) meters. pick() returns a clearance-valid cell
        whose plume-hit distance is sampled uniformly in [lo, hi] (nearest
        available cell to that target). If the case can't reach `hi` (small
        map / plume fills it), it returns its farthest valid cell.
    min_source_dist : never place the robot closer than this to the source
        (keep the task non-trivial; mirrors cfg.MIN_SOURCE_ROBOT_DIST).
    """

    def __init__(self, placements: dict, rng: np.random.Generator,
                 target_phd_range=(4.0, 14.0), min_source_dist: float = 3.0):
        self._rng = rng
        self._lo, self._hi = float(target_phd_range[0]), float(target_phd_range[1])
        cells = placements['cells']
        phd = placements['phd']
        src_dist = placements['src_dist']
        keep = (src_dist >= min_source_dist) & np.isfinite(phd)
        # If nothing has finite phd (plume never reached a valid cell), fall
        # back to all source-distance-valid cells with phd treated as "very far".
        if not keep.any():
            keep = src_dist >= min_source_dist
            phd = np.where(np.isfinite(phd), phd, src_dist)  # proxy
        self._cells = cells[keep]
        self._phd = phd[keep]
        self._ok = len(self._cells) > 0
        self._max_phd = float(self._phd.max()) if self._ok else 0.0

    @property
    def usable(self) -> bool:
        return self._ok

    @property
    def max_phd(self) -> float:
        return self._max_phd

    def pick(self) -> Optional[tuple]:
        """Return (x, y) for a far-plume-hit start, or None if unusable."""
        if not self._ok:
            return None
        target = self._rng.uniform(self._lo, self._hi)
        # nearest available cell to the target plume-hit distance
        idx = int(np.argmin(np.abs(self._phd - target)))
        return (float(self._cells[idx, 0]), float(self._cells[idx, 1]))


def precompute_library(library_dirs, rl_package_path: str,
                       clearance: float = 0.6, template_filter=None,
                       recompute: bool = False, limit: Optional[int] = None):
    """CLI helper: warm + cache far-placement for every case in the libraries.
    Run once offline so training resets stay cheap. Prints a summary."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cfd_library_loader import CFDLibrarySampler  # reuse its case-scanning

    rng = np.random.default_rng(0)
    sampler = CFDLibrarySampler(library_dirs, rng, rl_package_path,
                                template_filter=template_filter)
    cases = sampler._cases if limit is None else sampler._cases[:limit]
    n = len(cases)
    maxphds = []
    nogas = 0
    for i, cd in enumerate(cases):
        p = compute_case_placements(cd, rl_package_path, clearance=clearance,
                                    recompute=recompute)
        finite = np.isfinite(p['phd'])
        mp = float(p['phd'][finite].max()) if finite.any() else 0.0
        maxphds.append(mp)
        if p['n_gas_cells'] == 0:
            nogas += 1
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  [{i+1}/{n}] cached; running max-phd median="
                  f"{np.median(maxphds):.1f}m, no-gas cases={nogas}", flush=True)
    mp = np.array(maxphds)
    print(f"\nDone: {n} cases. max-plume-hit distance per case: "
          f"median={np.median(mp):.1f}m p90={np.percentile(mp,90):.1f}m "
          f"max={mp.max():.1f}m. cases-with-no-plume={nogas}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--library-dir', required=True,
                    help='comma-separated list of CFD library dirs')
    ap.add_argument('--rl-package-path',
                    default='/comp04-storage/efe-mantaroglu/osl/friend_base_local_wind')
    ap.add_argument('--clearance', type=float, default=0.6)
    ap.add_argument('--template-filter', default=None,
                    help='comma-separated template ids to restrict to')
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    libs = [s for s in a.library_dir.split(',') if s]
    tf = ([int(x) for x in a.template_filter.split(',')]
          if a.template_filter else None)
    precompute_library(libs, a.rl_package_path, clearance=a.clearance,
                       template_filter=tf, recompute=a.recompute, limit=a.limit)
