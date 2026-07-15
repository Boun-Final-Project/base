"""Synthetic spatially-varying wind pool = procedural maps + potential-flow wind.

The cheap stand-in for the expensive CFD wind-field library. Per the GADEN wind
report, the project's canonical spatial-wind generator (rl_gaden_wind/envs/
wind_field.py, vendored here as wind_field_potflow.WindField) is exactly the
"uniform wind with varying" idea:

  sample (speed, direction) -> solve 2D potential flow  ∇²φ = 0  with the room's
  outer wall ring as open Dirichlet inflow/outflow (φ = -s(cosθ·x + sinθ·y)) and
  interior obstacles as Neumann no-penetration -> (Ux,Uy) = -∇φ -> OPTIONAL
  divergence-free curl-noise overlay for stationary eddies.

This module pairs that generator with champ's per-episode procedural MapGenerator
maps and packs the result into the SAME pool dict layout as
gpu_cfd_loader.load_cfd_pool, so GpuVecEnvMulti / the local-wind trainer consume
it unchanged (spatial advection via wind_query_bilinear_pool, local-wind obs via
faithful_obs_wind_pool — the report-faithful (Ux/2+1)/2 encoding).

Why use this over the CFD library:
  - cheap: one sparse Laplace solve per map (~30-70 ms CPU), no OpenFOAM/GADEN bake;
  - unlimited: champ's infinite procedural maps instead of a finite CFD case set;
  - the curl-noise term adds eddies, closing part of the irrotational-vs-CFD gap
    the report flags as the limitation of pure potential flow.

LIMITATION (validate against GADEN, do not assume equivalence): even with curl
noise this is a TRAINING distribution, not a faithful GADEN reproduction. The
many_rooms GADEN trap is partly geometry-locked recirculation; confirm a synth-
trained local-wind ckpt against the CPU local2 / GADEN baseline before trusting it.
"""
import os
import sys
import numpy as np


def _train_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "train")        # champ/train (CPU golden src)


def build_synth_pool(n_cases, seed=0, res=0.1, speed_range=(0.1, 1.5), max_speed=2.0,
                     curl_amp=0.0, curl_scale=4.0, min_start_dist=2.0,
                     max_template=None, template_weights=None,
                     width_range=None, height_range=None, verbose=False):
    """Procedural maps + canonical potential-flow wind, in load_cfd_pool's dict layout.

    Parameters
    ----------
    n_cases : int            number of (map, field) cases in the pool.
    speed_range : (lo, hi)   sampled mean inflow speed per map (m/s).
    curl_amp : float         curl-noise amplitude (0 = pure potential flow; >0 adds eddies).
    curl_scale : float       curl-noise smoothing length in cells.
    max_template : int|None  cap template id (curriculum); None = all templates.
    template_weights : list  optional sampling weights over templates [0..max_template].
    """
    sys.path.insert(0, _train_dir())
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from reinforcement_learning.envs.map_generator import MapGenerator
    from wind_field_potflow import WindField

    rng = np.random.default_rng(seed)
    mg = MapGenerator(rng=rng, width_range=width_range, height_range=height_range)
    wf = WindField(speed_range=speed_range, max_speed=max_speed,
                   curl_noise_amplitude=curl_amp, curl_noise_scale=curl_scale)
    n_tpl = len(mg.TEMPLATES)
    hi_tpl = n_tpl - 1 if max_template is None else min(max_template, n_tpl - 1)

    loaded = []
    while len(loaded) < n_cases:
        if template_weights is not None:
            w = np.asarray(template_weights[:hi_tpl + 1], float); w = w / w.sum()
            tid = int(rng.choice(hi_tpl + 1, p=w))
        else:
            tid = int(rng.integers(0, hi_tpl + 1))
        o = mg.generate(template_id=tid)
        occ_grid = o["grid"]                                   # OccupancyGrid (.grid,.resolution)
        grid = np.asarray(occ_grid.grid).astype(np.int8)       # [H,W] 0=free 1=wall
        if (grid == 0).sum() < 50:
            continue
        wf.randomize(occ_grid, rng)                            # potential flow (+curl) for this map
        field = np.stack([wf.Ux, wf.Uy], axis=-1).astype(np.float32)   # [H,W,2]
        src = np.array(o["source_pos"], float)
        loaded.append((grid, field, float(o["width"]), float(o["height"]), src))
        if verbose and len(loaded) % 50 == 0:
            print(f"  synth pool {len(loaded)}/{n_cases}", flush=True)

    K = len(loaded)
    maxH = max(g.shape[0] for g, *_ in loaded)
    maxW = max(g.shape[1] for g, *_ in loaded)
    grids = np.ones((K, maxH, maxW), np.int8)                  # pad = WALL
    fields = np.zeros((K, maxH, maxW, 2), np.float32)
    sources = np.zeros((K, 2)); map_dims = np.zeros((K, 2)); free_cells = []
    for k, (grid, field, mw, mh, src) in enumerate(loaded):
        h, w = grid.shape
        grids[k, :h, :w] = grid
        fields[k, :h, :w] = field
        sources[k] = src; map_dims[k] = [mw, mh]
        ys, xs = np.where(grid == 0)
        wx = (xs + 0.5) * res; wy = (ys + 0.5) * res
        dd = np.hypot(wx - src[0], wy - src[1])
        keep = dd > min_start_dist
        fc = np.stack([wx[keep], wy[keep]], 1) if keep.sum() > 10 else np.stack([wx, wy], 1)
        free_cells.append(fc)
    winds = np.tile(np.array([[1.0, 0.0]]), (K, 1))            # dummy (spatial overrides)
    return dict(grids=grids, sources=sources, winds=winds, free_cells=free_cells,
                fields=fields, map_dims=map_dims, res=res, K=K, maxH=maxH, maxW=maxW,
                n_flipped=0)


def _spatial_variation(pool):
    """Diagnostic: mean fraction of free cells whose local wind points >60deg off
    the map's mean wind (0% = uniform; higher = more spatial structure)."""
    coss = []
    for k in range(pool["K"]):
        h = int(round(pool["map_dims"][k, 1] / pool["res"]))
        w = int(round(pool["map_dims"][k, 0] / pool["res"]))
        f = pool["fields"][k, :h, :w]
        occ = pool["grids"][k, :h, :w] != 0
        loc = f[~occ]; nz = np.linalg.norm(loc, axis=1) > 1e-6
        if nz.sum() < 10:
            continue
        mu = loc[nz].mean(0); mn = mu / (np.linalg.norm(mu) + 1e-9)
        c = (loc[nz] @ mn) / np.linalg.norm(loc[nz], axis=1)
        coss.append((c < 0.5).mean())
    return float(np.mean(coss)) if coss else 0.0


if __name__ == "__main__":
    import time, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--curl-amp", type=float, default=0.0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    pool = build_synth_pool(n_cases=a.n, seed=1, curl_amp=a.curl_amp, verbose=True)
    dt = time.perf_counter() - t0
    print(f"synth pool K={pool['K']} padded[{pool['maxH']},{pool['maxW']}] res={pool['res']} "
          f"| build {dt:.1f}s ({dt/pool['K']*1e3:.0f} ms/map)")
    print(f"spatial variation (curl_amp={a.curl_amp}): "
          f"{_spatial_variation(pool)*100:.1f}% of cells >60deg off mean wind")
