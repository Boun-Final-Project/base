"""Discriminative held-out CFD validation.

The default val gate (eval_cfd_holdout.py) measures binary success@0.5m, which SATURATES
at 100% once the policy converges (STEP_SIZE=0.5 + D_SUCCESS=0.5 lets the robot cover 300m
in 600 steps on 10-15m maps) -- so it cannot rank checkpoints past ~upd 500 and forces
checkpoint selection onto the GADEN *test* maps (test-set overfitting).

This variant keeps train/val in the SAME distribution (held-out shards, disjoint geometry)
but replaces the saturating metric with two graded ones that never saturate:

  * mean CLOSEST APPROACH to source (metres) over the episode -- continuous, lower is better.
    We set d_success ~= 0 so the episode is NOT terminated at 0.5m; the robot keeps acting and
    we record the minimum distance it ever achieves. This measures how *precisely* the policy
    localises the source, which stays discriminative long after success@0.5 hits 100%.
  * success@R for tight radii R in {0.15, 0.25, 0.35, 0.5} -- derived from the same min-dist,
    so success@0.5 reproduces the old gate while the tighter radii separate converged policies.

Selection metric for the paper = held-out (in-distribution) val. GADEN stays a pure TEST set,
scored once at the end. Run sweep_disc() over the checkpoints to check this val rank-correlates
with the GADEN test scores.
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("OSL_LOCAL_WIND_OBS", "1")

import gpu_env as ge
from eval_cfd_holdout import template_tier, TMPL_NAMES

RADII = (0.15, 0.25, 0.35, 0.50)


def make_disc_env(pool, ecfg, device, E=256, flip=True, seed=12345, d_success=0.05):
    """Val env over held-out pool; d_success driven near 0 so episodes run to max_steps and we
    measure closest approach instead of early-terminating at 0.5m."""
    res = pool["res"]
    diff = template_tier(pool)
    cfg = ecfg(res, flip)
    cfg.d_success = float(d_success)            # SimpleNamespace: disable 0.5m termination
    env = ge.GpuVecEnvMulti(
        pool["grids"], pool["sources"], pool["winds"], pool["free_cells"], res,
        cfg, device, E=E, seed=seed,
        wind_fields=list(pool["fields"]), map_dims=pool["map_dims"],
        free_dists=pool.get("free_dists"))
    env.set_difficulty(diff)
    env.set_curriculum(diff.max())
    return env


@torch.no_grad()
def evaluate_disc(agent, env, device, n_episodes=1500, n_tiers=10):
    """Greedy rollout tracking per-episode minimum distance to source. Returns per-tier and
    overall: mean min-dist + success@R for each R in RADII."""
    agent.eval()
    obs = env.reset_all().float()
    E = obs.shape[0]
    run_min = torch.full((E,), float("inf"), device=device)   # closest approach this episode
    mind_by_tier = [[] for _ in range(n_tiers)]
    all_mind = []
    while len(all_mind) < n_episodes:
        pre_idx = env.map_idx.clone()
        mean, _ = agent.get_actor_params(obs)
        nobs, r, term, trunc, info = env.step(mean)
        run_min = torch.minimum(run_min, info["dist"])
        done = info["done"]
        di = torch.nonzero(done, as_tuple=False).squeeze(1)
        if di.numel():
            tiers = env.map_difficulty[pre_idx[di]].tolist()
            md = run_min[di].tolist()
            for t, m in zip(tiers, md):
                mind_by_tier[int(t)].append(float(m)); all_mind.append(float(m))
            run_min[di] = float("inf")                          # reset tracker for next episode
        obs = nobs.float()

    def summarize(vals):
        if not vals:
            return None
        a = np.asarray(vals)
        return dict(mean_mind=float(a.mean()),
                    **{f"s{int(R*100)}": 100.0 * float((a <= R).mean()) for R in RADII},
                    n=len(a))

    per_tier = [summarize(v) for v in mind_by_tier]
    overall = summarize(all_mind)
    return dict(overall=overall, per_tier=per_tier)


def _load_pool_and_ecfg(cfd_dirs, holdout_shards, cases, seed):
    import gpu_cfd_loader as cl
    from train_localwind_gpu import ecfg
    osl_root = os.path.dirname(os.path.dirname(ROOT))
    lib_dirs = [os.path.join(osl_root, d) for d in cfd_dirs.split(",") if d]
    pool = cl.load_cfd_pool(lib_dirs, n_cases=cases, seed=seed,
                            holdout_shards=holdout_shards.split(","), split="val")
    return pool, ecfg


def _load_agent(ckpt, device):
    from reinforcement_learning.models.actor_critic import ActorCriticDualBackbone
    agent = ActorCriticDualBackbone().to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    agent.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck, strict=False)
    return agent


def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="one or more checkpoints (sweep if >1)")
    ap.add_argument("--cfd-dirs", default="cfd_test/library_v4_4dir")
    ap.add_argument("--holdout-shards", default="shard_38,shard_39")
    ap.add_argument("--cases", type=int, default=600)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--flip", action="store_true", default=True)
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool, ecfg = _load_pool_and_ecfg(a.cfd_dirs, a.holdout_shards, a.cases, a.seed)
    print(f"VAL pool: K={pool['K']} held-out {a.holdout_shards}  (discriminative metric)\n", flush=True)
    env = make_disc_env(pool, ecfg, dev, E=a.envs, flip=a.flip, seed=a.seed)

    hdr = f"  {'ckpt':44s} {'mean_mind':>9} " + " ".join(f"s@{R:.2f}".rjust(7) for R in RADII)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for ck in a.ckpts:
        agent = _load_agent(ck, dev)
        res = evaluate_disc(agent, env, dev, n_episodes=a.episodes)
        o = res["overall"]
        name = os.path.basename(ck)
        line = f"  {name:44s} {o['mean_mind']:9.3f} " + " ".join(f"{o[f's{int(R*100)}']:6.0f}%" for R in RADII)
        print(line, flush=True)
        rows.append((name, o))
    return rows


if __name__ == "__main__":
    _main()
