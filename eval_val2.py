"""Layer-2 held-out validation: unbiased, paired, GADEN-shaped, with a sparse-signal slice.

What it fixes over the previous val gates
-----------------------------------------
1. UNBIASED ESTIMATOR. The old evals drained an auto-resetting vec env until they had
   n_episodes -- survivorship bias: fast successes recycle and crowd out slow failures, which
   only land at truncation. Here every env contributes EXACTLY ONE episode per cohort (its
   first) and a cohort only ends when all envs have finished.
2. PAIRED STARTS. The global torch RNG is reseeded per cohort (map/start sampling at
   gpu_env._reset_envs and filament turbulence both draw from it, in fixed-shape draws), so
   every checkpoint sees the same start states. Paired comparison removes most eval noise.
   (Within a cohort, mid-cohort auto-resets consume variable RNG, so plume noise diverges
   slightly across checkpoints after the first env finishes; starts of the RECORDED first
   episodes are identical.)
3. GADEN-SHAPED EPISODES. Corrected filament physics (K=0.0005 etc, straight off the GADEN
   sim.yaml -- see project_surrogate_gas_physics_bug) + far starts (geodesic percentile band,
   matching GADEN's recommended far start poses).
4. SPARSE-SIGNAL SLICE. GADEN punishes exactly the maps where the gas signal is sparse
   (many_rooms: 7.6%% detect) but those are a sliver of the 486-map pool, so the pooled
   average cannot see regressions on them. A reference policy scores per-map detect%% once
   (cached); the bottom-decile maps become a dedicated slice with its own success number.
5. MULTI-METRIC. success, SPL, detour, closest-approach (mean min-dist), gas%%; the same for
   the sparse slice. The correlation table vs the known GADEN scores of the job-25311 sweep
   says which metric (if any) can rank checkpoints -- that is the Layer-4 meta-test.

Usage (meta-test on the 25311 sweep):
  python eval_val2.py runs/..job25311/checkpoints/localwind_agent_{131072000,...}.pt \
      --ref runs/..job25311/checkpoints/localwind_agent_1048576000.pt
"""
import os
import sys
import json
import hashlib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("OSL_LOCAL_WIND_OBS", "1")

from eval_cfd_holdout import template_tier
from eval_cfd_holdout_disc import _load_pool_and_ecfg, _load_agent
from eval_spl_val import (build_geodesics, restrict_starts, make_spl_env, GADEN, _steps_of,
                          R_STEP, R_DET)

# GADEN-faithful filament physics, mapped 1:1 from the real sim.yaml (NOT fitted to scores):
#   filamentGrowthGamma 10 cm^2/s -> K=0.0005 (config ships 0.02 = 40x too fast)
#   filamentInitialSigma 10 cm -> 0.10 | numFilaments_sec 10 @ dt=0.5 -> fps 5
#   no age cull in GADEN -> max_age stays at the config default (120)
GAS_FAITHFUL = dict(K=0.0005, initial_sigma=0.10, filaments_per_step=5)


def subset_pool(pool, sel):
    """Restrict a loaded pool to the map indices in sel (list/array of ints)."""
    p = dict(pool)
    for k in ("grids", "sources", "winds", "free_cells", "free_dists",
              "fields", "map_dims", "template_ids"):
        v = p.get(k)
        if v is None:
            continue
        p[k] = v[np.asarray(sel)] if isinstance(v, np.ndarray) else [v[i] for i in sel]
    p["K"] = len(sel)
    return p


@torch.no_grad()
def per_map_detect(agent, env, K, n_steps=4000):
    """detect%% per map under a reference policy: per-STEP accumulation by map index.
    (Per-step stats don't need episode bookkeeping; ~n_steps*E/K samples per map.)"""
    agent.eval()
    obs = env.reset_all().float()
    det = torch.zeros(K, device=obs.device)
    tot = torch.zeros(K, device=obs.device)
    ones = torch.ones(env.E, device=obs.device)
    for _ in range(n_steps):
        mi = env.map_idx.clone()                       # before auto-reset can change it
        mean, _ = agent.get_actor_params(obs)
        nobs, r, term, trunc, info = env.step(mean)
        tot.index_add_(0, mi, ones)
        det.index_add_(0, mi, info["binary"].to(ones.dtype))
        obs = nobs.float()
    return (100.0 * det / torch.clamp(tot, min=1)).cpu().numpy()


@torch.no_grad()
def evaluate_v2(agent, env, geo_t, device, n_episodes, seed, n_tiers=10,
                step_size=0.5, d_success=0.5, res=0.1):
    """Fixed-cohort greedy rollout, one episode per env per cohort, reseeded per cohort.
    Records (tier, geodesic L at start, steps, success, gas steps, min dist)."""
    agent.eval()
    H, W = geo_t.shape[1], geo_t.shape[2]
    rec, rec_tier = [], [[] for _ in range(n_tiers)]
    cohort = 0
    while len(rec) < n_episodes:
        torch.manual_seed(seed + 7919 * cohort)        # seeds ALL devices -> paired starts
        obs = env.reset_all().float()
        E = obs.shape[0]
        pending = torch.ones(E, dtype=torch.bool, device=device)
        tier0 = env.map_difficulty[env.map_idx].clone()
        cx = torch.clamp((env.robot[:, 0] / res).long(), 0, W - 1)
        cy = torch.clamp((env.robot[:, 1] / res).long(), 0, H - 1)
        L0 = geo_t[env.map_idx, cy, cx].clone()
        mind = torch.linalg.norm(env.robot - env.source, dim=1)
        steps = torch.zeros(E, device=device)
        gas = torch.zeros(E, device=device)
        while bool(pending.any()):
            mean, _ = agent.get_actor_params(obs)      # greedy = actor mean
            nobs, r, term, trunc, info = env.step(mean)
            p = pending.to(steps.dtype)
            steps += p
            gas += info["binary"].to(steps.dtype) * p
            mind = torch.where(pending, torch.minimum(mind, info["dist"]), mind)
            done = info["done"] & pending
            di = torch.nonzero(done, as_tuple=False).squeeze(1)
            if di.numel():
                for t, L, st, s, g, md in zip(tier0[di].tolist(), L0[di].tolist(),
                                              steps[di].tolist(), info["success"][di].tolist(),
                                              gas[di].tolist(), mind[di].tolist()):
                    row = (float(L), float(st), bool(s), float(g), float(md))
                    rec.append(row); rec_tier[int(t)].append(row)
                pending[di] = False
            obs = nobs.float()
        cohort += 1

    def summarize(rows):
        if not rows:
            return None
        L = np.array([r[0] for r in rows], np.float64)
        st = np.array([r[1] for r in rows], np.float64)
        S = np.array([r[2] for r in rows], np.float64)
        G = np.array([r[3] for r in rows], np.float64)
        M = np.array([r[4] for r in rows], np.float64)
        fin = np.isfinite(L)
        Le = np.maximum(L[fin] - d_success, step_size)
        P = st[fin] * step_size
        spl = S[fin] * Le / np.maximum(P, Le)
        det = (P / Le)[S[fin] > 0]
        gasf = float((G / np.maximum(st, 1)).mean())
        return dict(succ=100.0 * float(S.mean()),
                    spl=100.0 * float(spl.mean()) if fin.any() else float("nan"),
                    detour=float(det.mean()) if det.size else float("nan"),
                    mind=float(M.mean()),
                    gas=100.0 * gasf,
                    n=len(rows))

    return dict(overall=summarize(rec), per_tier=[summarize(v) for v in rec_tier])


def _main():
    import argparse
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--ref", required=True,
                    help="reference ckpt for the per-map detect%% slice (fixed across sweeps)")
    ap.add_argument("--cfd-dirs", default="cfd_test/library_v4_4dir")
    ap.add_argument("--holdout-shards", default="shard_38,shard_39")
    ap.add_argument("--cases", type=int, default=486)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=1536)
    ap.add_argument("--slice-episodes", type=int, default=768)
    ap.add_argument("--budget", type=int, default=600, help="max_steps (600 = GADEN protocol)")
    ap.add_argument("--start-pct", default="50,100",
                    help="geodesic percentile band for starts (GADEN starts are far)")
    ap.add_argument("--slice-pct", type=float, default=10.0,
                    help="bottom detect%% percentile of maps forming the sparse slice")
    ap.add_argument("--detect-steps", type=int, default=4000,
                    help="env steps for the per-map detect%% pass (~steps*E/K samples per map)")
    ap.add_argument("--flip", action="store_true", default=True)
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--training-gas", action="store_true",
                    help="use the (bugged, K=0.02) training physics instead of GADEN-faithful")
    ap.add_argument("--gaden-corr", action="store_true",
                    help="correlate against the job-25311 GADEN scores. ONLY valid when the "
                         "ckpts ARE the 25311 sweep -- the pairing is by raw step count, so any "
                         "other run whose step counts coincide would be paired with a DIFFERENT "
                         "policy's GADEN scores (meaningless).")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gas = None if a.training_gas else GAS_FAITHFUL
    pool, ecfg = _load_pool_and_ecfg(a.cfd_dirs, a.holdout_shards, a.cases, a.seed)
    cache = os.path.join(ROOT, f"geo_cache_{a.holdout_shards.replace(',', '_')}_{a.cases}_{a.seed}.npz")
    geo = build_geodesics(pool, cache)

    lo, hi = [float(x) for x in a.start_pct.split(",")]
    if (lo, hi) != (0.0, 100.0):
        pool = restrict_starts(pool, geo, lo, hi)

    env = make_spl_env(pool, ecfg, dev, E=a.envs, flip=a.flip, seed=a.seed,
                       budget=a.budget, gas=gas)
    geo_t = torch.from_numpy(geo).to(dev)

    # ---- sparse slice: per-map detect% under the reference policy (cached) ----
    key = hashlib.md5(f"{a.holdout_shards}|{a.cases}|{a.seed}|{lo}-{hi}|{gas}|"
                      f"{os.path.basename(a.ref)}|{a.budget}".encode()).hexdigest()[:10]
    dcache = os.path.join(ROOT, f"detect_cache_{key}.json")
    if os.path.exists(dcache):
        detmap = np.array(json.load(open(dcache)), np.float32)
        print(f"  per-map detect%: loaded cache {dcache}", flush=True)
    else:
        ref = _load_agent(a.ref, dev)
        detmap = per_map_detect(ref, env, pool["K"], n_steps=a.detect_steps)
        json.dump([float(x) for x in detmap], open(dcache, "w"))
    thr = float(np.percentile(detmap, a.slice_pct))
    sel = np.nonzero(detmap <= thr)[0]
    print(f"  sparse slice: {len(sel)}/{pool['K']} maps with detect% <= {thr:.1f} "
          f"(pool median {np.median(detmap):.1f}%)", flush=True)

    spool = subset_pool(pool, sel)
    senv = make_spl_env(spool, ecfg, dev, E=a.envs, flip=a.flip, seed=a.seed + 1,
                        budget=a.budget, gas=gas)
    sgeo_t = torch.from_numpy(geo[sel]).to(dev)

    print(f"\nVAL2 K={pool['K']} slice={len(sel)} budget={a.budget} start-band={lo:.0f}-{hi:.0f}% "
          f"gas={'training(K=0.02)' if a.training_gas else 'GADEN-faithful(K=0.0005)'} "
          f"E={a.envs} eps={a.episodes}+{a.slice_episodes}\n", flush=True)
    hdr = (f"  {'ckpt':32s} {'succ':>6} {'SPL':>6} {'detour':>7} {'mind':>6} {'gas%':>5} "
           f"| {'sl_succ':>7} {'sl_mind':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    rows = []
    for ck in a.ckpts:
        agent = _load_agent(ck, dev)
        r = evaluate_v2(agent, env, geo_t, dev, a.episodes, a.seed, res=pool["res"])
        rs = evaluate_v2(agent, senv, sgeo_t, dev, a.slice_episodes, a.seed + 1, res=pool["res"])
        o, so = r["overall"], rs["overall"]
        o["slice_succ"], o["slice_mind"] = so["succ"], so["mind"]
        o["combo"] = 0.5 * (o["succ"] + so["succ"])
        print(f"  {os.path.basename(ck):32s} {o['succ']:5.1f}% {o['spl']:5.1f} "
              f"{o['detour']:7.2f} {o['mind']:6.2f} {o['gas']:4.0f}% "
              f"| {so['succ']:6.1f}% {so['mind']:7.2f}", flush=True)
        rows.append((_steps_of(ck), o))

    # ---- Layer-4 meta-test: can any metric rank the 25311 sweep like GADEN does? ----
    paired = [(s, o) for s, o in rows if s in GADEN] if a.gaden_corr else []
    if len(paired) >= 4:
        g = np.array([GADEN[s] for s, _ in paired], float)
        gbest = max(GADEN[s] for s, _ in paired)
        print(f"\n  correlation vs GADEN test (n={len(paired)} ckpts, GADEN best={gbest}%)")
        print(f"  {'metric':>10} {'spearman':>9}   picks -> GADEN  (regret)")
        for key_, sign in (("succ", +1), ("spl", +1), ("detour", -1), ("mind", -1),
                           ("slice_succ", +1), ("slice_mind", -1), ("combo", +1)):
            v = np.array([o[key_] for _, o in paired], float)
            sp = spearmanr(v, g).correlation * sign
            pick = paired[int(np.argmax(sign * v))][0]
            print(f"  {key_:>10} {sp:+9.2f}   upd{pick//262144:<5d} -> {GADEN[pick]}%  "
                  f"({GADEN[pick]-gbest:+d})")


if __name__ == "__main__":
    _main()
