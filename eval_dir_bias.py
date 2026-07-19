"""Test whether a checkpoint has a SOURCE-DIRECTION bias (the confound the biased-7 test can't see).

The surrogate held-out val is direction-balanced (v4 4-inlet, symmetric source placement), so an
east-vs-west success gap WITHIN a checkpoint is the policy's bias, not the sim's. We group val maps
by source hemisphere (source_x / map_width) and report success per hemisphere. If upd 4000 is much
better on WEST-source maps than EAST-source maps, its "80% on the biased-7" is inflated by the
source-left confound.

Reuses evaluate_disc by overloading map_difficulty as the GROUP id (0=west,1=mid,2=east).
"""
import os, sys, numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
os.environ.setdefault("OSL_LOCAL_WIND_OBS", "1")
from eval_cfd_holdout_disc import make_disc_env, evaluate_disc, _load_pool_and_ecfg, _load_agent

def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--cfd-dirs", default="cfd_test/library_v4_4dir")
    ap.add_argument("--holdout-shards", default="shard_38,shard_39")
    ap.add_argument("--cases", type=int, default=600)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool, ecfg = _load_pool_and_ecfg(a.cfd_dirs, a.holdout_shards, a.cases, a.seed)
    env = make_disc_env(pool, ecfg, dev, E=a.envs, seed=a.seed)

    # group each map by source hemisphere: source_x / map_width
    src_x = pool["sources"][:, 0]
    mapw = np.asarray(pool["map_dims"])[:, 0] if pool.get("map_dims") is not None else None
    if mapw is None:
        mapw = np.array([g.shape[1] * pool["res"] for g in pool["grids"]])
    frac = np.asarray(src_x) / np.asarray(mapw)
    group = np.where(frac < 0.4, 0, np.where(frac > 0.6, 2, 1)).astype(np.int64)  # 0=W 1=M 2=E
    nW, nM, nE = (group == 0).sum(), (group == 1).sum(), (group == 2).sum()
    print(f"VAL maps by source hemisphere: WEST(<0.4)={nW}  MID={nM}  EAST(>0.6)={nE}\n", flush=True)

    env.set_difficulty(group)          # overload: tier = hemisphere group
    env.set_curriculum(2)              # enable all groups
    names = {0: "WEST-src", 1: "MID", 2: "EAST-src"}
    print(f"  {'ckpt':30s} {'group':>9} {'succ@0.5':>8} {'mean_mind':>9} {'n':>6}")
    print("  " + "-" * 70)
    for ck in a.ckpts:
        agent = _load_agent(ck, dev)
        res = evaluate_disc(agent, env, dev, n_episodes=a.episodes, n_tiers=3)
        base = os.path.basename(ck)
        for g in (0, 2, 1):
            pt = res["per_tier"][g]
            if pt is None:
                continue
            print(f"  {base:30s} {names[g]:>9} {pt['s50']:7.0f}% {pt['mean_mind']:9.3f} {pt['n']:>6}", flush=True)
        w = res["per_tier"][0]; e = res["per_tier"][2]
        if w and e:
            print(f"  {'':30s} {'W-E gap':>9} {w['s50']-e['s50']:+6.0f}pt   (positive => WEST-source easier for this policy)\n", flush=True)

if __name__ == "__main__":
    _main()
