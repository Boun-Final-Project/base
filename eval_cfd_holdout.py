"""Held-out CFD validation gate.

Rolls a policy (GREEDY / actor mean) on the GPU env over a pool of CFD cases the trainer
NEVER saw (held-out shards), and reports overall + per-wall-tier success. This is the honest
in-distribution gate that v4's size unlocks: prior "it cracked many_rooms" claims were proxy-
optimistic because train and eval shared geometry. Here val geometry is disjoint from train.

Used two ways:
  * imported by train_localwind_gpu.py for in-training val + best-by-val checkpoint selection;
  * standalone CLI to score any checkpoint on the held-out pool.

The ULTIMATE gate is still gpu_fromscratch_gaden_eval.py (real replayed GADEN gas). This is the
fast on-device proxy used for selection/early-stop only.
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("OSL_LOCAL_WIND_OBS", "1")

import gpu_env as ge


TMPL_NAMES = ["empty","single_wall","u_shape","three_walls","complex_maze",
              "multi_room","dead_end","serpentine","dense_multi_room","hybrid"]


def template_tier(pool):
    """Per-map difficulty tier = template_id (T0=empty .. T9=hybrid). Matches the CPU
    trainer's curriculum ordering: templates have explicit difficulty intent, unlike
    wall-fraction which placed 77% of v4 maps into a single tier-0 bucket."""
    return pool["template_ids"].copy()   # [K] int32, 0-9


def make_val_env(pool, ecfg, device, E=256, flip=True, seed=12345):
    """Build a GpuVecEnvMulti over the val pool with all maps enabled and full start radius."""
    res = pool["res"]
    diff = template_tier(pool)
    env = ge.GpuVecEnvMulti(
        pool["grids"], pool["sources"], pool["winds"], pool["free_cells"], res,
        ecfg(res, flip), device, E=E, seed=seed,
        wind_fields=list(pool["fields"]), map_dims=pool["map_dims"],
        free_dists=pool.get("free_dists"))
    env.set_difficulty(diff)
    env.set_curriculum(diff.max())          # all maps sampleable
    return env


@torch.no_grad()
def evaluate(agent, env, device, n_episodes=1500, n_tiers=10):
    """Greedy rollout in fixed cohorts: every env contributes EXACTLY ONE episode per cohort
    (its first), and a cohort only ends when ALL envs have finished. Cohorts repeat until
    n_episodes is reached.

    The previous version drained an auto-resetting vec env until it had n_episodes —
    survivorship bias: successes finish early and recycle, failures only land at truncation
    (step max_steps), so once the policy is fast enough that E envs yield n_episodes
    successes before the first truncation, the loop exits having recorded ZERO failures.
    That is the mechanism behind the fake 100.0% val plateaus (jobs 25311/25963/25964);
    even below the plateau it over-samples fast successes and biases val upward."""
    agent.eval()
    succ_by_tier = [[] for _ in range(n_tiers)]
    all_succ = []
    while len(all_succ) < n_episodes:
        obs = env.reset_all().float()
        E = obs.shape[0]
        pending = torch.ones(E, dtype=torch.bool, device=device)
        tier0 = env.map_difficulty[env.map_idx].clone()     # tier of each env's FIRST episode
        while bool(pending.any()):
            mean, _ = agent.get_actor_params(obs)           # greedy action = actor mean
            nobs, r, term, trunc, info = env.step(mean)
            done = (term | trunc) & pending                 # auto-reset episodes are ignored
            di = torch.nonzero(done, as_tuple=False).squeeze(1)
            if di.numel():
                s = info["success"][di].tolist()
                for t, sv in zip(tier0[di].tolist(), s):
                    succ_by_tier[int(t)].append(float(sv)); all_succ.append(float(sv))
                pending[di] = False
            obs = nobs.float()
    overall = 100.0 * float(np.mean(all_succ)) if all_succ else 0.0
    per_tier = [100.0 * float(np.mean(v)) if v else None for v in succ_by_tier]
    return dict(overall=overall, per_tier=per_tier,
                n=len(all_succ), n_by_tier=[len(v) for v in succ_by_tier])


def _main():
    import argparse
    import gpu_cfd_loader as cl
    from reinforcement_learning.models.actor_critic import ActorCriticDualBackbone
    from train_localwind_gpu import ecfg

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
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
    osl_root = os.path.dirname(os.path.dirname(ROOT))
    lib_dirs = [os.path.join(osl_root, d) for d in a.cfd_dirs.split(",") if d]
    pool = cl.load_cfd_pool(lib_dirs, n_cases=a.cases, seed=a.seed,
                            holdout_shards=a.holdout_shards.split(","), split="val")
    print(f"VAL pool: K={pool['K']} from held-out {a.holdout_shards} "
          f"padded[{pool['maxH']},{pool['maxW']}]", flush=True)

    agent = ActorCriticDualBackbone().to(dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    agent.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck, strict=False)

    env = make_val_env(pool, ecfg, dev, E=a.envs, flip=a.flip, seed=a.seed)
    res = evaluate(agent, env, dev, n_episodes=a.episodes)
    print(f"\nHeld-out CFD val ({a.ckpt}), flip={a.flip}")
    print(f"  {'tier':>4} {'success':>8} {'n':>6}")
    for t, (sr, n) in enumerate(zip(res["per_tier"], res["n_by_tier"])):
        print(f"  {t:>4} {('%.0f%%' % sr) if sr is not None else '   -':>8} {n:>6}")
    print(f"\n  OVERALL held-out val success = {res['overall']:.0f}%  (n={res['n']})")


if __name__ == "__main__":
    _main()
