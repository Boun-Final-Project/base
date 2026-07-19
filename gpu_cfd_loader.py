"""Load CFD-library cases into the GPU env's padded pool (heterogeneous map sizes ->
pad to common max, pad cells = wall). Returns arrays ready for GpuVecEnvMulti."""
import json
from pathlib import Path
import numpy as np


def _is_case(c):
    return (c / "grid.npz").exists() and (c / "wind_field.npz").exists() \
        and (c / "meta.json").exists()


def _case_dirs(lib_dirs):
    """Return [(case_path, shard_name), ...] for every CFD case under lib_dirs.

    Handles both the flat layout (lib_dir/case_*/, e.g. library_v3) and the sharded v4
    layout (lib_dir/shard_*/case_*/). For the flat layout the shard name is the lib dir
    name; for the sharded layout it is the shard dir name (e.g. 'shard_07'), which the
    train/val split keys on."""
    out = []
    for d in lib_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for c in sorted(d.iterdir()):
            if not c.is_dir():
                continue
            if _is_case(c):
                out.append((c, d.name))                      # flat layout
            else:
                for cc in sorted(c.iterdir()):               # one level down (a shard dir)
                    if cc.is_dir() and _is_case(cc):
                        out.append((cc, c.name))
    return out


def load_cfd_pool(lib_dirs, n_cases, min_start_dist=2.0, seed=0, res_pick=None, mirror_prob=0.0,
                  holdout_shards=None, split=None):
    """Returns grids[K,maxH,maxW] int8, sources[K,2], winds[K,2] (dummy), free_cells(list),
    free_dists(list of [n_k] dist-to-source per free cell), fields[K,maxH,maxW,2],
    map_dims[K,2] (true w,h), res.

    mirror_prob>0: x-mirror each case with that probability (flip grid horizontally, negate
    wind u-component, reflect source x). The CFD library's inlet is the x=0 face in 100% of
    cases -> wind always blows +x; mirroring ~half balances wind direction +-x (the GPU
    equivalent of the CPU pipeline's randomized per-episode wind direction), removing the
    L/R bias that lets a from-scratch policy shortcut 'go upwind' to 'go -x'.

    holdout_shards/split: deterministic, leakage-free train/val split by shard. With
    split='train' the held-out shards are EXCLUDED; with split='val' ONLY they are kept; the
    same `holdout_shards` set + disjoint splits guarantee the val pool is geometry the train
    pool never saw. split=None keeps everything (back-compat)."""
    rng = np.random.default_rng(seed)
    allc = _case_dirs(lib_dirs)
    hold = set(holdout_shards or [])
    if split == "train":
        allc = [(c, s) for (c, s) in allc if s not in hold]
    elif split == "val":
        allc = [(c, s) for (c, s) in allc if s in hold]
    allc = [c for (c, s) in allc]                            # paths only past the split
    rng.shuffle(allc)
    loaded = []
    for c in allc:
        if len(loaded) >= n_cases:
            break
        g = np.load(c / "grid.npz"); grid = g["grid"].astype(np.int8)
        res = float(g["cell_size"]); mw = float(g["map_width"]); mh = float(g["map_height"])
        if res_pick is None:
            res_pick = res
        if abs(res - res_pick) > 1e-9:
            continue                                       # keep a single common resolution
        field = np.load(c / "wind_field.npz")["field"].astype(np.float32)
        if field.shape[:2] != grid.shape:
            continue
        # skip the handful of bad CFD solves with non-finite / extreme wind (inf max_speed ->
        # OverflowError in FilamentPlume); physical field speeds are well under a few m/s
        if (not np.isfinite(field).all()) or float(np.abs(field).max()) > 20.0:
            continue
        meta = json.loads((c / "meta.json").read_text())
        src = np.array(meta["source_pos"], float)
        tid = int(meta.get("template_id", -1))
        loaded.append((grid, field, mw, mh, src, tid))
    K = len(loaded)
    assert K > 0, "no CFD cases loaded"
    maxH = max(g.shape[0] for g, *_ in loaded); maxW = max(g.shape[1] for g, *_ in loaded)
    grids = np.ones((K, maxH, maxW), np.int8)              # pad = WALL
    fields = np.zeros((K, maxH, maxW, 2), np.float32)
    sources = np.zeros((K, 2)); map_dims = np.zeros((K, 2)); free_cells = []; free_dists = []
    template_ids = np.full(K, -1, dtype=np.int32)
    n_flipped = 0
    for k, (grid, field, mw, mh, src, tid) in enumerate(loaded):
        if mirror_prob > 0 and rng.random() < mirror_prob:    # x-mirror to balance the x=0 inlet
            grid = grid[:, ::-1].copy()
            field = field[:, ::-1, :].copy(); field[:, :, 0] *= -1.0     # flip cols + negate u (wind now blows -x)
            src = np.array([mw - src[0], src[1]]); n_flipped += 1
        h, w = grid.shape
        grids[k, :h, :w] = grid
        fields[k, :h, :w] = field
        sources[k] = src; map_dims[k] = [mw, mh]; template_ids[k] = tid
        ys, xs = np.where(grid == 0)
        wx = (xs + 0.5) * res_pick; wy = (ys + 0.5) * res_pick
        d = np.hypot(wx - src[0], wy - src[1])
        keep = d > min_start_dist
        if keep.sum() > 10:
            fc = np.stack([wx[keep], wy[keep]], 1); fd = d[keep]
        else:
            fc = np.stack([wx, wy], 1); fd = d
        free_cells.append(fc); free_dists.append(fd.astype(np.float32))
    winds = np.tile(np.array([[1.0, 0.0]]), (K, 1))        # dummy (spatial wind overrides)
    return dict(grids=grids, sources=sources, winds=winds, free_cells=free_cells,
                free_dists=free_dists, fields=fields, map_dims=map_dims,
                template_ids=template_ids, res=res_pick,
                K=K, maxH=maxH, maxW=maxW, n_flipped=n_flipped)
