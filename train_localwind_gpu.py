"""GPU local-wind-observation trainer (CFD spatially-varying wind).

The "local wind observation" lever (the local2 finetune that cracked many_rooms): instead of a
single mean wind vector, the policy OBSERVES the LOCAL wind at the robot's position, which only
differs from the mean when wind is SPATIALLY VARYING. So this trainer trains on the real CFD
wind-field library (gpu_cfd_loader) with the GPU env's spatial-wind path:
  - advection: per-filament bilinear query of the CFD field (gpu_wind.wind_query_bilinear_pool)
  - obs:       local wind at the robot (gpu_wind.faithful_obs_wind_pool), encoded to [0,1]
Both are validated EXACT vs the CPU WindModel in the original GPU port (see project_gpu_env_port).
This reuses the validated GpuVecEnvMulti spatial path — the SAME on-device speed as the champ port.

Faithful to the local2 lineage: local-wind obs is a FINETUNE on top of a competent base, so
--resume from a champ checkpoint is the intended use (lr 1e-4). It also runs from scratch.

Usage (SLURM): sbatch train_localwind_gpu.sh
  python -m ... not needed; run as a script with PYTHONPATH=<this dir>.
"""
import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("OSL_LOCAL_WIND_OBS", "1")     # CPU env (eval) observes local wind too

from reinforcement_learning import config as cfg
from reinforcement_learning.models.actor_critic import ActorCriticDualBackbone
from reinforcement_learning.training.ppo import RolloutBuffer, compute_gae, ppo_update
import gpu_env as ge
import gpu_cfd_loader as cl

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ecfg(res, flip, step_size=None):
    """GPU env cfg namespace (mirrors gpu_two_stage_train.ecfg)."""
    ss = step_size if step_size is not None else cfg.STEP_SIZE
    return types.SimpleNamespace(
        res=res, filaments_per_step=cfg.FILAMENTS_PER_STEP, max_age=cfg.FILAMENT_MAX_AGE,
        mass=cfg.FILAMENT_MASS, initial_sigma=cfg.FILAMENT_INITIAL_SIGMA,
        min_sigma=cfg.FILAMENT_MIN_SIGMA, dt=cfg.FILAMENT_DT, K=cfg.FILAMENT_K,
        turb_sigma=cfg.FILAMENT_TURBULENCE_SCALE, turb_scale=cfg.FILAMENT_TURBULENCE_SCALE,
        reflection_energy=cfg.FILAMENT_REFLECTION_ENERGY, warmup=cfg.FILAMENT_WARMUP_STEPS,
        lidar_rays=cfg.LIDAR_NUM_RAYS, lidar_range=cfg.LIDAR_MAX_RANGE, max_speed=cfg.WIND_MAX_SPEED,
        flip=flip, gas_hist=cfg.GAS_HISTORY_LENGTH, max_steps=cfg.MAX_STEPS, step_size=ss,
        d_success=cfg.D_SUCCESS, thr_weight=cfg.SENSOR_THRESHOLD_WEIGHT, robot_radius=cfg.ROBOT_RADIUS,
        r_step=cfg.R_STEP, r_coll=cfg.R_COLLISION, r_det=cfg.R_DETECTION, r_success=cfg.R_SUCCESS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reinforcement_learning/runs/gpu_localwind")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--cfd-cases", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)               # finetune default
    ap.add_argument("--resume", default=None, help="champ checkpoint to finetune from (intended use)")
    ap.add_argument("--flip", action="store_true", default=False,
                    help="apply the (u,v)->(-u,+v) deploy-anemometer flip to the wind obs")
    ap.add_argument("--cfd-dirs", default="cfd_test/library_v4_4dir")
    ap.add_argument("--no-mirror", action="store_true")
    # held-out validation split (leakage-free, by shard) + in-training eval gate
    ap.add_argument("--holdout-shards", default="shard_38,shard_39")
    ap.add_argument("--val-cases", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=100, help="updates between held-out val evals")
    ap.add_argument("--eval-episodes", type=int, default=1500)
    # opt-in reverse curriculum (near->far robot start). OFF by default (random full-map starts).
    ap.add_argument("--reverse-curriculum", action="store_true", default=False)
    ap.add_argument("--rc-start-radius", type=float, default=3.0, help="initial start radius (m)")
    ap.add_argument("--rc-frac", type=float, default=0.6, help="fraction of updates to reach full radius")
    # wind source: 'cfd' = real CFD library (expensive, finite); 'synth' = cheap
    # procedural maps + potential-flow(+curl-noise) wind (GADEN-tuned), no CFD bake.
    ap.add_argument("--wind", choices=["cfd", "synth"], default="cfd")
    ap.add_argument("--speed-lo", type=float, default=0.05)         # GADEN-tuned (config.py)
    ap.add_argument("--speed-hi", type=float, default=0.5)
    ap.add_argument("--curl-amp", type=float, default=0.6)          # eddies; match GADEN CFD
    ap.add_argument("--curl-scale", type=float, default=4.0)
    ap.add_argument("--step-size", type=float, default=None,
                    help="robot step size in metres (default: cfg.STEP_SIZE = 0.5)")
    ap.add_argument("--max-template", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    # ---- gas physics (default = unchanged config.py values; the GADEN-faithful arm overrides) ----
    # The surrogate's filament variance grows at 2*K per second (gpu_filament.advect_diffuse_batch:
    # sigma <- sqrt(sigma^2 + 2*K*dt)). config.py ships K=0.02 m^2/s = 400 cm^2/s, but the real GADEN
    # sim.yaml we deploy against specifies filamentGrowthGamma = 10 cm^2/s => K = 0.0005. The surrogate
    # therefore disperses filaments 40x too fast, smearing GADEN's thin intermittent ribbons into a fat
    # continuous cloud (measured: robot in detection 54-71% of steps vs GADEN's 23.9%).
    ap.add_argument("--gas-k", type=float, default=None,
                    help="target FILAMENT_K (GADEN-faithful = 0.0005; config default 0.02)")
    ap.add_argument("--gas-k-start", type=float, default=None,
                    help="K to start from, annealed down to --gas-k (default: config.py K)")
    ap.add_argument("--gas-anneal-frac", type=float, default=0.3,
                    help="fraction of updates over which K is annealed (log-space) to --gas-k")
    ap.add_argument("--gas-sigma", type=float, default=None,
                    help="FILAMENT_INITIAL_SIGMA (GADEN filamentInitialSigma=10cm => 0.1)")
    ap.add_argument("--gas-fps", type=int, default=None,
                    help="FILAMENTS_PER_STEP (GADEN numFilaments_sec=10 at dt=0.5 => 5)")
    a = ap.parse_args()

    _step_size = a.step_size if a.step_size is not None else cfg.STEP_SIZE
    print(f"device={DEV} | LOCAL-WIND (CFD spatial) trainer | E={a.envs} flip={a.flip} step_size={_step_size}", flush=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    agent = ActorCriticDualBackbone().to(DEV)
    if a.resume:
        ck = torch.load(a.resume, map_location=DEV, weights_only=False)
        agent.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck, strict=False)
        print(f"  resumed (finetune) from {a.resume}", flush=True)

    # ---- pool with spatially-varying wind fields (CFD library OR cheap synth) ----
    val_pool = None
    holdout = [s for s in a.holdout_shards.split(",") if s]
    if a.wind == "synth":
        import synth_wind as sw
        pool = sw.build_synth_pool(n_cases=a.cfd_cases, seed=a.seed,
                                   speed_range=(a.speed_lo, a.speed_hi),
                                   curl_amp=a.curl_amp, curl_scale=a.curl_scale,
                                   max_template=a.max_template, verbose=True)
        print(f"  SYNTH wind: potential-flow + curl({a.curl_amp}) | "
              f"speed({a.speed_lo},{a.speed_hi}) | variation="
              f"{sw._spatial_variation(pool)*100:.1f}% cells off-mean", flush=True)
    else:
        lib_dirs = [d for d in a.cfd_dirs.split(",") if d]
        cwd = os.path.join(os.path.dirname(os.path.dirname(ROOT)))   # osl root (cfd_test lives there)
        ld = [os.path.join(cwd, d) for d in lib_dirs]
        # TRAIN pool: held-out shards EXCLUDED (leakage-free split)
        pool = cl.load_cfd_pool(ld, n_cases=a.cfd_cases, seed=a.seed,
                                mirror_prob=(0.0 if a.no_mirror else 0.5),
                                holdout_shards=holdout, split="train")
        # VAL pool: ONLY held-out shards (no mirror — keep it a clean, fixed gate)
        val_pool = cl.load_cfd_pool(ld, n_cases=a.val_cases, seed=a.seed + 777,
                                    mirror_prob=0.0, holdout_shards=holdout, split="val")
    res, mdims = pool["res"], pool["map_dims"]
    # difficulty tier = template_id (T0=empty ... T9=hybrid), matching the CPU trainer's
    # curriculum ordering. This is meaningful (templates have explicit difficulty intent) unlike
    # wall-fraction which gives 77% of maps the same tier-0 label.
    diff = pool["template_ids"].copy()                   # [K] int, 0-9
    n_tiers = int(diff.max()) + 1                        # typically 10
    import functools as _ft

    def _gas_cfg(res_, flip_, step_size=None, K=None):
        """env cfg with the gas-physics overrides applied (no-op when the flags are unset)."""
        c = ecfg(res_, flip_, step_size)
        if K is not None:
            c.K = float(K)
        elif a.gas_k is not None:
            c.K = float(a.gas_k)
        if a.gas_sigma is not None:
            c.initial_sigma = float(a.gas_sigma)
        if a.gas_fps is not None:
            c.filaments_per_step = int(a.gas_fps)   # NB: ring buffer F = max_age*fps*2 grows with this
        return c

    _ecfg = _ft.partial(_gas_cfg, step_size=a.step_size)   # val env gets the TARGET gas physics
    # K is annealed in the loop, so the TRAIN env starts at gas_k_start (default: config.py K).
    k0 = a.gas_k_start if a.gas_k_start is not None else cfg.FILAMENT_K
    train_cfg = _gas_cfg(res, a.flip, a.step_size, K=(k0 if a.gas_k is not None else None))
    if a.gas_k is not None:
        print(f"  GAS: K {k0:g} -> {a.gas_k:g} (log-anneal over first {a.gas_anneal_frac*100:.0f}% "
              f"of updates) | initial_sigma={train_cfg.initial_sigma:g} "
              f"filaments_per_step={train_cfg.filaments_per_step} max_age={train_cfg.max_age}",
              flush=True)
    env = ge.GpuVecEnvMulti(pool["grids"], pool["sources"], pool["winds"], pool["free_cells"], res,
                            train_cfg, DEV, E=a.envs, seed=a.seed,
                            wind_fields=list(pool["fields"]), map_dims=mdims,
                            free_dists=pool.get("free_dists"))
    env.set_difficulty(diff)
    tmpl_names = ["empty","single_wall","u_shape","three_walls","complex_maze",
                  "multi_room","dead_end","serpentine","dense_multi_room","hybrid"]
    print(f"  pool[{a.wind}]: K={pool['K']} padded[{pool['maxH']},{pool['maxW']}] "
          f"mirrored={pool.get('n_flipped', 0)}", flush=True)
    print(f"  template tiers: {np.bincount(diff, minlength=n_tiers).tolist()} "
          f"(T0={tmpl_names[0]} .. T{n_tiers-1}={tmpl_names[n_tiers-1]})", flush=True)

    # ---- held-out validation env (built once; reused each eval) ----
    val_env = None
    if val_pool is not None:
        import eval_cfd_holdout as evh
        val_env = evh.make_val_env(val_pool, _ecfg, DEV, E=a.envs, flip=a.flip, seed=a.seed + 777)
        print(f"  VAL pool: K={val_pool['K']} held-out shards={holdout}", flush=True)
    if a.reverse_curriculum and not env.has_dists:
        raise RuntimeError("--reverse-curriculum needs free_dists (cfd wind path)")

    # ---- PPO loop (curriculum by tier, champ hyperparameters) ----
    opt = torch.optim.Adam(agent.parameters(), lr=a.lr, eps=1e-5)
    buf = RolloutBuffer(a.rollout, a.envs, cfg.STATE_DIM, 2, DEV)
    obs = env.reset_all().float()
    succ_w = []
    tier, tier_since, gate = 0, 0, 0.8
    env.set_curriculum(tier)
    max_dwell = max(1, int(a.updates * 0.85 / max(1, n_tiers - 1)))
    anneal_start, min_lr = 0.5, 1e-4
    Path(f"{a.out}/checkpoints").mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    full_radius = float(np.hypot(mdims[:, 0].max(), mdims[:, 1].max()))   # covers any map
    if a.reverse_curriculum:
        env.set_start_radius(a.rc_start_radius)
        print(f"  reverse curriculum ON: start radius {a.rc_start_radius}m -> {full_radius:.1f}m "
              f"over first {a.rc_frac*100:.0f}% of updates", flush=True)
    t0 = time.perf_counter(); total = 0
    for upd in range(a.updates):
        progress = upd / a.updates
        cur_lr = a.lr if progress < anneal_start else max(
            min_lr, (1.0 - (progress - anneal_start) / (1.0 - anneal_start)) * a.lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        if a.reverse_curriculum:                              # expand start radius near->far
            frac = min(1.0, progress / max(1e-9, a.rc_frac))
            env.set_start_radius(full_radius if frac >= 1.0
                                 else a.rc_start_radius + frac * (full_radius - a.rc_start_radius))
        if a.gas_k is not None:
            # log-space anneal of the filament diffusivity: start at the (too fast) config value so
            # early exploration still gets a rich plume, end at the GADEN-faithful value. cfg.K is read
            # live every plume step (gpu_env._plume_step), so mutating it here takes effect immediately.
            f = min(1.0, progress / max(1e-9, a.gas_anneal_frac))
            env.cfg.K = k0 * (a.gas_k / k0) ** f
        buf.reset()
        for _ in range(a.rollout):
            with torch.no_grad():
                act, lp, _, v = agent.get_action_and_value(obs)
            nobs, r, term, trunc, info = env.step(act)
            done = (term | trunc).float()
            buf.insert(obs, act, lp, r, done, v.flatten())
            for d in torch.nonzero(done, as_tuple=False).squeeze(1).tolist():
                succ_w.append(info["success"][d].item())
            obs = nobs.float(); total += a.envs
        with torch.no_grad():
            nv = agent.get_value(obs).flatten()
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, nv, cfg.GAMMA, cfg.GAE_LAMBDA)
        ppo_update(agent, opt, buf, adv, ret, cfg.CLIP_EPSILON, cfg.ENTROPY_COEFF,
                   cfg.VALUE_LOSS_COEFF, cfg.MAX_GRAD_NORM, cfg.UPDATE_EPOCHS, cfg.NUM_MINIBATCHES)
        rs = np.mean(succ_w[-400:]) if len(succ_w) >= 100 else 0.0
        tier_since += 1
        if tier < n_tiers - 1 and (rs > gate or tier_since >= max_dwell):
            tier += 1; env.set_curriculum(tier); tier_since = 0
            print(f"    >> curriculum -> tier {tier} (rs={rs*100:.0f}%)", flush=True)
        if (upd + 1) % 10 == 0:
            print(f"  upd {upd+1:>4}/{a.updates} tier {tier} succ={rs*100:4.0f}% "
                  f"steps={total:,} {total/(time.perf_counter()-t0):,.0f}/s", flush=True)
        if (upd + 1) % a.ckpt_every == 0:
            p = f"{a.out}/checkpoints/localwind_agent_{total}.pt"
            torch.save({"model_state_dict": agent.state_dict(), "global_step": total}, p)
            print(f"    [ckpt] {p}", flush=True)
        # ---- held-out validation gate: log val success + keep best-by-val checkpoint ----
        if val_env is not None and (upd + 1) % a.eval_every == 0:
            vr = evh.evaluate(agent, val_env, DEV, n_episodes=a.eval_episodes)
            agent.train()                                    # evaluate() leaves agent in eval()
            pt = " ".join(f"t{t}:{('%.0f' % s) if s is not None else '-'}"
                          for t, s in enumerate(vr["per_tier"]))
            print(f"    [val] upd {upd+1} held-out succ={vr['overall']:.1f}% "
                  f"(n={vr['n']}) [{pt}]", flush=True)
            if vr["overall"] > best_val:
                best_val = vr["overall"]
                bp = f"{a.out}/checkpoints/localwind_best_val.pt"
                torch.save({"model_state_dict": agent.state_dict(), "global_step": total,
                            "val_success": best_val}, bp)
                print(f"    [val] new best held-out val {best_val:.1f}% -> {bp}", flush=True)
    torch.save({"model_state_dict": agent.state_dict(), "global_step": total},
               f"{a.out}/checkpoints/localwind_final.pt")
    print(f"LOCAL-WIND TRAINING COMPLETE. best held-out val={best_val:.1f}%", flush=True)


if __name__ == "__main__":
    main()
