"""Calibrate a GADEN-LIKE validation set out of our OWN generator.

The domain gap is gas-signal SPARSITY, measured (gaden_gas_stats.py, ckpt upd4000, same policy in
both domains):

    GADEN     : detect% = 23.9%   mean blank-run = 17.8 steps
    surrogate : detect% = 71%     (training filament physics -- a mature dense plume on step 1)

This sweeps the filament physics knobs on the HELD-OUT val shards and reports the same two
statistics, to find the config that reproduces GADEN's signal structure.

CRITICAL (why this is not just cheating one level up): we tune to match a PHYSICAL STATISTIC of the
target domain (detect% / blank-run), NOT its success scores. No GADEN success number enters this
calibration. Only after the config is fixed do we run the checkpoint correlation -- once -- as an
honest test.

Knobs (reinforcement_learning/config.py defaults):
    FILAMENT_WARMUP_STEPS     15    plume pre-built before the robot's first obs  -> 0 = must search
    FILAMENT_MAX_AGE         120    ~60 s filament life -> long dense plume       -> lower = shorter reach
    FILAMENTS_PER_STEP         2    release rate                                  -> lower = sparser
    FILAMENT_TURBULENCE_SCALE 0.2   turbulence as fraction of wind                -> higher = patchier
"""
import os
import sys
import itertools
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("OSL_LOCAL_WIND_OBS", "1")

import gpu_env as ge
from eval_cfd_holdout import template_tier
from eval_cfd_holdout_disc import _load_pool_and_ecfg, _load_agent

GADEN_DETECT = 23.9      # %, target
GADEN_BLANK = 17.8       # steps, target


def make_env(pool, ecfg, device, E, seed, warmup, max_age, fps, turb,
             K=None, sigma=None, mass=None, budget=600):
    res = pool["res"]
    diff = template_tier(pool)
    cfg = ecfg(res, True)
    cfg.max_steps = int(budget)
    cfg.warmup = int(warmup)                 # plume steps before first obs
    cfg.max_age = int(max_age)               # filament cull age
    cfg.filaments_per_step = int(fps)        # release rate
    cfg.turb_sigma = float(turb)             # turbulence
    cfg.turb_scale = float(turb)
    # thinness knobs -- these drive INTERMITTENCY (sharp on/off signal => long blank runs),
    # which the extent knobs above cannot produce
    if K is not None:
        cfg.K = float(K)                     # atmospheric diffusivity: lower = filaments stay tight
    if sigma is not None:
        cfg.initial_sigma = float(sigma)     # thinner filaments = you either hit one or you don't
    if mass is not None:
        cfg.mass = float(mass)               # lower peak concentration = fewer threshold crossings
    env = ge.GpuVecEnvMulti(
        pool["grids"], pool["sources"], pool["winds"], pool["free_cells"], res,
        cfg, device, E=E, seed=seed,
        wind_fields=list(pool["fields"]), map_dims=pool["map_dims"],
        free_dists=pool.get("free_dists"))
    env.set_difficulty(diff)
    env.set_curriculum(diff.max())
    return env


@torch.no_grad()
def gas_stats(agent, env, device, n_episodes=600):
    """detect% and mean blank-run length, plus success -- the same statistics gaden_gas_stats.py
    reports on the real maps, so the two domains are directly comparable."""
    agent.eval()
    obs = env.reset_all().float()
    E = obs.shape[0]
    prev = torch.zeros(E, dtype=torch.bool, device=device)
    tot_steps = torch.zeros((), device=device)
    tot_det = torch.zeros((), device=device)
    n_runs = torch.zeros((), device=device)      # blank runs that ended in a detection
    succ, done_n = [], 0
    while done_n < n_episodes:
        mean, _ = agent.get_actor_params(obs)
        nobs, r, term, trunc, info = env.step(mean)
        b = info["binary"].bool()
        tot_steps += E
        tot_det += b.sum()
        n_runs += (b & ~prev).sum()              # 0 -> 1 transition closes a blank run
        prev = b
        d = info["done"]
        di = torch.nonzero(d, as_tuple=False).squeeze(1)
        if di.numel():
            succ += info["success"][di].tolist()
            done_n += int(di.numel())
            prev[di] = False
        obs = nobs.float()
    det = 100.0 * float(tot_det / tot_steps)
    blank = float((tot_steps - tot_det) / torch.clamp(n_runs, min=1))
    return det, blank, 100.0 * float(np.mean(succ))


def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="reference policy -- MUST be the same ckpt used on GADEN")
    ap.add_argument("--cfd-dirs", default="cfd_test/library_v4_4dir")
    ap.add_argument("--holdout-shards", default="shard_38,shard_39")
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool, ecfg = _load_pool_and_ecfg(a.cfd_dirs, a.holdout_shards, a.cases, a.seed)
    agent = _load_agent(a.ckpt, dev)

    # (warmup, max_age, fps, turb, K, sigma, mass); first row = the training default
    GRID = [(15, 120, 2, 0.2, 0.02, 0.05, 1.0)]
    # Row 2 = the hacky match from sweep 2: hits GADEN's detect% by TRUNCATING the plume
    # (max_age=10, warmup=0) -> hard spatial cutoff, EMPTY far field. Right number, wrong world.
    GRID += [(0, 10, 1, 0.4, 0.02, 0.02, 0.3)]
    #
    # PHYSICS-FAITHFUL family, mapped from the real GADEN sim.yaml:
    #   filamentGrowthGamma  = 10 cm^2/s   -> sigma^2 grows at 2K  => K = 0.0005 m^2/s
    #                                          (current K=0.02 => 400 cm^2/s, i.e. 40x too fast)
    #   filamentInitialSigma = 10 cm       -> initial_sigma = 0.1 m (current 0.05)
    #   numFilaments_sec     = 10 /s       -> FILAMENT_DT=0.5 => filaments_per_step = 5 (current 2)
    #   no age cull in GADEN               -> keep max_age HIGH (filaments persist, they DILUTE)
    #   start_time           = 100 s       -> plume is MATURE at t=0 => keep warmup HIGH
    # GADEN is sparse because its filaments stay THIN RIBBONS, not because the plume is short/young.
    GRID += [(w, 120, 5, t, K, 0.1, 1.0)
             for w, t, K in itertools.product(
                 (15, 60), (0.05, 0.1, 0.2), (0.0005, 0.001, 0.002, 0.005))]

    print(f"\nTarget (GADEN, same ckpt): detect%={GADEN_DETECT:.1f}  blank={GADEN_BLANK:.1f}\n")
    print(f"  {'warm':>4} {'age':>4} {'fps':>3} {'turb':>5} {'K':>6} {'sig':>5} {'mass':>5} | "
          f"{'detect%':>8} {'blank':>6} {'succ':>5} | {'err':>6}")
    print("  " + "-" * 78)
    rows = []
    for (w, m, f, t, K, s, ms) in GRID:
        env = make_env(pool, ecfg, dev, a.envs, a.seed, w, m, f, t, K=K, sigma=s, mass=ms)
        det, bl, sc = gas_stats(agent, env, dev, n_episodes=a.episodes)
        # normalised distance to the GADEN signal statistics
        err = abs(det - GADEN_DETECT) / GADEN_DETECT + abs(bl - GADEN_BLANK) / GADEN_BLANK
        tag = ("  <- TRAINING DEFAULT" if (w, m, f, t, K, s, ms) == (15, 120, 2, 0.2, 0.02, 0.05, 1.0)
               else "  <- truncation hack (right number, wrong mechanism)" if m == 10
               else "  <- GADEN-faithful K" if K == 0.0005 else "")
        print(f"  {w:>4} {m:>4} {f:>3} {t:>5.2f} {K:>7.4f} {s:>5.2f} {ms:>5.1f} | "
              f"{det:7.1f}% {bl:6.1f} {sc:4.0f}% | {err:6.2f}{tag}", flush=True)
        rows.append(((w, m, f, t, K, s, ms), det, bl, sc, err))

    best = min(rows, key=lambda r: r[4])
    (w, m, f, t, K, s, ms), det, bl, sc, err = best
    print(f"\n  BEST MATCH: warmup={w} max_age={m} fps={f} turb={t} K={K} sigma={s} mass={ms}")
    print(f"              detect%={det:.1f} (GADEN {GADEN_DETECT})  blank={bl:.1f} (GADEN {GADEN_BLANK})"
          f"  succ={sc:.0f}%  err={err:.2f}")
    sat = [r for r in rows if r[3] >= 99.0]
    print(f"\n  NOTE: {len(sat)}/{len(rows)} configs still leave success SATURATED at >=99% -- sparser")
    print(f"  gas alone does not make the val discriminative; it needs the tight budget + far starts too.")


if __name__ == "__main__":
    _main()
