"""Efficiency-based held-out validation (SPL) on the SAME generated distribution.

Why this exists
---------------
The default val gate (success@0.5m, 600 steps) SATURATES: STEP_SIZE=0.5 x MAX_STEPS=600 gives the
robot 300 m of travel on a 10-20 m map, so it can effectively sweep the whole thing. Every decent
checkpoint scores ~95% and the metric cannot RANK them -- which is what forced checkpoint selection
onto the GADEN test maps.

The fix is not a different distribution, it is a metric that does not saturate. We keep the exact
same generator, maps and train/val shard split, and change what we measure:

  * SPL (Success weighted by Path Length), the standard embodied-navigation metric:
        SPL = mean_i [ S_i * L_i / max(P_i, L_i) ]
    L_i = GEODESIC shortest-path length from the episode start to the success ball (Dijkstra on the
    occupancy grid -- NOT euclidean, which is meaningless in a labyrinth).
    P_i = distance actually travelled = steps * step_size (action is direction-only with a fixed
    0.5 m step, so this is exact; a collision leaves the robot in place and is counted as wasted
    travel, which is what we want).
  * DETOUR = P/L on successful episodes -- "how many times longer than optimal is the path".
    This is the direct read on the 'solves it but inefficiently' complaint.
  * success@budget under a TIGHT step budget, so brute-force sweeping stops working.

Knobs that make the val discriminative (all on the maps you already have):
  --budget       max_steps (600 = original saturating gate; ~120 forces plume-following)
  --start-pct    keep only start cells in this GEODESIC percentile band per map
                 (e.g. 60,100 = hard/far starts only; default 0,100 = unchanged)

Selection metric for the paper = this held-out in-distribution val. GADEN stays a pure TEST set.
Run over the checkpoint sweep and rank-correlate against GADEN to see whether it can select.
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("OSL_LOCAL_WIND_OBS", "1")

import gpu_env as ge
from eval_cfd_holdout import template_tier
from eval_cfd_holdout_disc import _load_pool_and_ecfg, _load_agent

R_STEP = -1.0       # reinforcement_learning/config.py -- per-step cost
R_DET = 0.75        # reinforcement_learning/config.py -- paid on EVERY step gas is detected

SQ2 = float(np.sqrt(2.0))
_NEI = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]


def geodesic_from_source(grid, src_xy, res):
    """True shortest-path distance (metres) through free space from the source to every free cell.

    grid: [H,W] int8 with 0=free, 1=wall (padding is wall, so it blocks naturally).
    Returns [H,W] float32, +inf in walls and in any region unreachable from the source.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    H, W = grid.shape
    free = (grid == 0)
    fy, fx = np.nonzero(free)
    n = fy.size
    out = np.full((H, W), np.inf, np.float32)
    if n == 0:
        return out
    nid = -np.ones((H, W), np.int64)
    nid[fy, fx] = np.arange(n)

    rows, cols, vals = [], [], []
    for dy, dx, w in _NEI:
        ys, xs = fy + dy, fx + dx
        inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        a = nid[fy[inb], fx[inb]]
        b = nid[ys[inb], xs[inb]]
        ok = b >= 0
        rows.append(a[ok]); cols.append(b[ok])
        vals.append(np.full(int(ok.sum()), w * res, np.float32))
    g = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()

    # source cell; if it landed in a wall, snap to the nearest free cell
    sx = int(np.clip(src_xy[0] / res, 0, W - 1))
    sy = int(np.clip(src_xy[1] / res, 0, H - 1))
    if not free[sy, sx]:
        d2 = (fy - sy) ** 2 + (fx - sx) ** 2
        j = int(np.argmin(d2)); sy, sx = int(fy[j]), int(fx[j])
    d = dijkstra(g, indices=int(nid[sy, sx]), directed=False)
    out[fy, fx] = d.astype(np.float32)
    return out


def build_geodesics(pool, cache_path=None):
    """[K,maxH,maxW] float32 geodesic-to-source field per map. Cached (Dijkstra x K is ~30 s)."""
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        if z["geo"].shape == (pool["K"], pool["maxH"], pool["maxW"]):
            print(f"  geodesics: loaded cache {cache_path}", flush=True)
            return z["geo"]
    K, res = pool["K"], pool["res"]
    geo = np.full((K, pool["maxH"], pool["maxW"]), np.inf, np.float32)
    for k in range(K):
        geo[k] = geodesic_from_source(pool["grids"][k], pool["sources"][k], res)
        if (k + 1) % 100 == 0:
            print(f"  geodesics: {k+1}/{K}", flush=True)
    if cache_path:
        np.savez_compressed(cache_path, geo=geo)
    return geo


def restrict_starts(pool, geo, lo_pct, hi_pct):
    """Keep only free cells whose GEODESIC distance to source falls in [lo,hi] percentile per map.

    lo=60,hi=100 -> far/hard starts only (the robot must search rather than start inside the plume).
    Also drops cells unreachable from the source, which would make SPL undefined.
    """
    res = pool["res"]
    fc_new, fd_new = [], []
    for k, fc in enumerate(pool["free_cells"]):
        cy = np.clip((fc[:, 1] / res).astype(int), 0, geo.shape[1] - 1)
        cx = np.clip((fc[:, 0] / res).astype(int), 0, geo.shape[2] - 1)
        d = geo[k][cy, cx]
        ok = np.isfinite(d)
        if ok.sum() < 10:                       # degenerate map: leave it alone
            fc_new.append(fc); fd_new.append(np.where(np.isfinite(d), d, 1e3).astype(np.float32)); continue
        dv = d[ok]
        lo = np.percentile(dv, lo_pct); hi = np.percentile(dv, hi_pct)
        sel = ok & (d >= lo) & (d <= hi)
        if sel.sum() < 10:
            sel = ok
        fc_new.append(fc[sel]); fd_new.append(d[sel].astype(np.float32))
    pool = dict(pool)
    pool["free_cells"] = fc_new
    pool["free_dists"] = fd_new                 # geodesic now, not euclidean
    return pool


def make_spl_env(pool, ecfg, device, E=256, flip=True, seed=12345, budget=600, gas=None):
    res = pool["res"]
    diff = template_tier(pool)
    cfg = ecfg(res, flip)
    cfg.max_steps = int(budget)                 # the step budget IS the difficulty knob
    if gas:                                     # GADEN-calibrated filament physics (see
        for k, v in gas.items():                # calibrate_gaden_like_val.py -- matched on
            setattr(cfg, k, v)                  # detect%/blank-run, NOT on GADEN success scores)
    env = ge.GpuVecEnvMulti(
        pool["grids"], pool["sources"], pool["winds"], pool["free_cells"], res,
        cfg, device, E=E, seed=seed,
        wind_fields=list(pool["fields"]), map_dims=pool["map_dims"],
        free_dists=pool.get("free_dists"))
    env.set_difficulty(diff)
    env.set_curriculum(diff.max())
    return env


@torch.no_grad()
def evaluate_spl(agent, env, geo_t, device, n_episodes=1500, n_tiers=10,
                 step_size=0.5, d_success=0.5, res=0.1):
    """Greedy rollout. Per episode we need only: geodesic L at the start, #steps taken, success.
    (Path length is exact: direction-only action with fixed step_size, collisions = wasted travel.)"""
    agent.eval()
    obs = env.reset_all().float()
    E = obs.shape[0]
    H, W = geo_t.shape[1], geo_t.shape[2]

    def start_L(idx_sel=None):
        rb = env.robot if idx_sel is None else env.robot[idx_sel]
        mi = env.map_idx if idx_sel is None else env.map_idx[idx_sel]
        cx = torch.clamp((rb[:, 0] / res).long(), 0, W - 1)
        cy = torch.clamp((rb[:, 1] / res).long(), 0, H - 1)
        return geo_t[mi, cy, cx]

    cur_L = start_L()
    cur_steps = torch.zeros(E, device=device)
    cur_gas = torch.zeros(E, device=device)     # steps with binary detection (in-plume)
    rec = [[] for _ in range(n_tiers)]          # per-tier list of (L, steps, success, gas_steps)
    allrec = []

    while len(allrec) < n_episodes:
        pre_idx = env.map_idx.clone()
        mean, _ = agent.get_actor_params(obs)
        nobs, r, term, trunc, info = env.step(mean)
        cur_steps += 1
        cur_gas += info["binary"].to(cur_gas.dtype)
        done = info["done"]
        di = torch.nonzero(done, as_tuple=False).squeeze(1)
        if di.numel():
            tiers = env.map_difficulty[pre_idx[di]].tolist()
            Ls = cur_L[di].tolist()
            Ss = cur_steps[di].tolist()
            Gs = cur_gas[di].tolist()
            ok = info["success"][di].tolist()
            for t, L, st, s, gsum in zip(tiers, Ls, Ss, ok, Gs):
                if not np.isfinite(L):          # unreachable start: SPL undefined, skip
                    continue
                row = (float(L), float(st), bool(s), float(gsum))
                rec[int(t)].append(row); allrec.append(row)
            cur_steps[di] = 0
            cur_gas[di] = 0
            cur_L[di] = start_L(di)             # env already auto-reset these to their new start
        obs = nobs.float()

    def summarize(rows):
        if not rows:
            return None
        L = np.array([r[0] for r in rows], np.float64)
        st = np.array([r[1] for r in rows], np.float64)
        S = np.array([r[2] for r in rows], np.float64)
        G = np.array([r[3] for r in rows], np.float64)
        # optimal travel = geodesic to the edge of the success ball
        Le = np.maximum(L - d_success, step_size)
        P = st * step_size                                      # actual travel
        spl = S * Le / np.maximum(P, Le)
        det = (P / Le)[S > 0]                                   # detour ratio on successes only
        gasf = float((G / np.maximum(st, 1)).mean())            # fraction of steps in detection
        # R_STEP=-1 but R_DETECTION=+0.75 fires on every in-plume step, so the effective per-step
        # cost is (r_step + gasf*r_det). The ratio to the clean-air cost is how much detour the
        # REWARD is willing to buy: a plume route is preferred over a direct one up to this factor.
        eff = R_STEP + gasf * R_DET
        return dict(succ=100.0 * float(S.mean()),
                    spl=100.0 * float(spl.mean()),
                    detour=float(det.mean()) if det.size else float("nan"),
                    steps=float(st[S > 0].mean()) if (S > 0).any() else float("nan"),
                    gas=100.0 * gasf, effcost=float(eff),
                    tol=float(abs(R_STEP) / max(abs(eff), 1e-6)),
                    optL=float(Le.mean()), n=len(rows))

    return dict(overall=summarize(allrec), per_tier=[summarize(v) for v in rec])


# GADEN test scores (20-ep overall %) for the job-25311 sweep -- ground truth we are trying to predict.
GADEN = {
    131072000: 69, 262144000: 61, 393216000: 77, 524288000: 76,
    655360000: 73, 786432000: 71, 917504000: 73, 1048576000: 80,
    1179648000: 79, 1310720000: 73, 1441792000: 77, 1572864000: 77,
    1703936000: 63, 1835008000: 61, 1966080000: 76, 2097152000: 66,
}


def _steps_of(path):
    import re
    m = re.search(r"_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _main():
    import argparse
    from scipy.stats import spearmanr, pearsonr

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--cfd-dirs", default="cfd_test/library_v4_4dir")
    ap.add_argument("--holdout-shards", default="shard_38,shard_39")
    ap.add_argument("--cases", type=int, default=600)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--budget", type=int, default=120, help="max_steps (600 = original gate)")
    ap.add_argument("--start-pct", default="0,100", help="geodesic percentile band for start cells")
    ap.add_argument("--flip", action="store_true", default=True)
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--gaden-gas", action="store_true",
                    help="use the GADEN-CALIBRATED filament physics (detect%%=22.4 blank=14.4, vs "
                         "GADEN's 23.9/17.8) instead of the training physics (56%%/4.5)")
    a = ap.parse_args()

    # calibrate_gaden_like_val.py best match -- fitted to GADEN's gas-signal STATISTICS only
    GADEN_GAS = dict(warmup=0, max_age=10, filaments_per_step=1,
                     turb_sigma=0.4, turb_scale=0.4, K=0.02, initial_sigma=0.02, mass=0.3)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool, ecfg = _load_pool_and_ecfg(a.cfd_dirs, a.holdout_shards, a.cases, a.seed)
    cache = os.path.join(ROOT, f"geo_cache_{a.holdout_shards.replace(',','_')}_{a.cases}_{a.seed}.npz")
    geo = build_geodesics(pool, cache)

    lo, hi = [float(x) for x in a.start_pct.split(",")]
    if (lo, hi) != (0.0, 100.0):
        pool = restrict_starts(pool, geo, lo, hi)
    band = f"{lo:.0f}-{hi:.0f}%"

    res = pool["res"]
    geo_t = torch.from_numpy(geo).to(dev)
    gas = GADEN_GAS if a.gaden_gas else None
    env = make_spl_env(pool, ecfg, dev, E=a.envs, flip=a.flip, seed=a.seed, budget=a.budget, gas=gas)
    optL = np.mean([np.mean(fd[np.isfinite(fd)]) for fd in pool["free_dists"]])
    print(f"\nVAL K={pool['K']}  budget={a.budget} steps ({a.budget*0.5:.0f} m of travel)  "
          f"start-band={band} geodesic  mean optimal path={optL:.1f} m  "
          f"gas={'GADEN-calibrated' if gas else 'training-default'}\n", flush=True)

    hdr = (f"  {'ckpt':32s} {'succ':>6} {'SPL':>7} {'detour':>7} {'steps':>7} "
           f"{'gas%':>6} {'effcost':>8} {'tol':>5}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for ck in a.ckpts:
        agent = _load_agent(ck, dev)
        r = evaluate_spl(agent, env, geo_t, dev, n_episodes=a.episodes,
                         step_size=0.5, d_success=0.5, res=res)
        o = r["overall"]
        print(f"  {os.path.basename(ck):32s} {o['succ']:5.0f}% {o['spl']:6.1f} "
              f"{o['detour']:7.2f} {o['steps']:7.0f} {o['gas']:5.0f}% "
              f"{o['effcost']:8.2f} {o['tol']:5.2f}x", flush=True)
        rows.append((_steps_of(ck), o))

    # rank-correlate every candidate metric against the GADEN test scores
    paired = [(s, o) for s, o in rows if s in GADEN]
    if len(paired) >= 4:
        g = np.array([GADEN[s] for s, _ in paired], float)
        print(f"\n  correlation vs GADEN test (n={len(paired)} checkpoints)")
        print(f"  {'metric':>8} {'spearman':>9} {'pearson':>8}   picks -> GADEN")
        for key, sign in (("spl", +1), ("succ", +1), ("detour", -1), ("steps", -1)):
            v = np.array([o[key] for _, o in paired], float)
            sp = spearmanr(v, g).correlation * sign
            pe = pearsonr(v, g)[0] * sign
            best = paired[int(np.argmax(sign * v))][0]
            print(f"  {key:>8} {sp:+9.2f} {pe:+8.2f}   upd{best//262144:<5d} -> {GADEN[best]}%")
        print(f"\n  GADEN's own best: upd{max(GADEN, key=GADEN.get)//262144} -> {max(GADEN.values())}%")


if __name__ == "__main__":
    _main()
