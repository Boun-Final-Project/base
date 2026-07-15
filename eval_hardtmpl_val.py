"""Test the user's hypothesis: does HARD-TEMPLATE (T7/T8/T9) held-out surrogate val track GADEN?

Same held-out CFD val pool + discriminative closest-approach metric as eval_cfd_holdout_disc.py,
but restricts sampling to the hardest templates only (serpentine/dense_multi_room/hybrid) by
overriding map_weight. If this val PEAKS near the GADEN peak (upd ~4000) it's a valid cheap
selector; if it climbs monotonically to upd 8000 it fails the same way pooled surrogate val did.
"""
import os, sys, numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
os.environ.setdefault("OSL_LOCAL_WIND_OBS", "1")
from eval_cfd_holdout_disc import make_disc_env, evaluate_disc, _load_pool_and_ecfg, _load_agent

HARD = {7, 8, 9}

def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--cfd-dirs", default="cfd_test/library_v4_4dir")
    ap.add_argument("--holdout-shards", default="shard_38,shard_39")
    ap.add_argument("--cases", type=int, default=600)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool, ecfg = _load_pool_and_ecfg(a.cfd_dirs, a.holdout_shards, a.cases, a.seed)
    env = make_disc_env(pool, ecfg, dev, E=a.envs, seed=a.seed)
    # restrict sampling to hard templates only
    env.map_weight = (env.map_difficulty >= 7).to(env.map_weight.dtype)
    nhard = int((env.map_difficulty >= 7).sum())
    print(f"VAL pool K={pool['K']}  hard-template maps (T7/8/9)={nhard}\n", flush=True)
    print(f"  {'ckpt':44s} {'mean_mind':>9} {'s@0.25':>7} {'s@0.50':>7}")
    print("  " + "-" * 72)
    for ck in a.ckpts:
        agent = _load_agent(ck, dev)
        res = evaluate_disc(agent, env, dev, n_episodes=a.episodes)
        o = res["overall"]
        print(f"  {os.path.basename(ck):44s} {o['mean_mind']:9.3f} {o['s25']:6.0f}% {o['s50']:6.0f}%", flush=True)

if __name__ == "__main__":
    _main()
