"""Measure the GAS SIGNAL STATISTICS of the real GADEN maps (not the success score).

Purpose: we want a validation set built from OUR OWN generator that predicts GADEN. Every attempt
so far failed because the val kept the TRAINING gas physics, which is the exact axis the domain gap
lives on (the surrogate hands the robot a mature dense plume on step 1 -- measured 71% of steps in
detection -- while GADEN makes it search).

To fix that WITHOUT fitting the validation set to the test scores (which would just move the
cheating up a level), we calibrate the surrogate's filament physics to match a PHYSICAL STATISTIC
of GADEN rather than its success numbers. This script measures that statistic:

  * detect%  -- fraction of steps with a binary gas detection (surrogate reference: 71%)
  * t_first  -- steps until the FIRST detection (inf = plume never reached the robot)
  * blank    -- mean length of a no-detection run (plume intermittency)

Those three numbers are the calibration target for FILAMENT_WARMUP_STEPS / FILAMENT_MAX_AGE /
FILAMENTS_PER_STEP / FILAMENT_TURBULENCE_SCALE in the GADEN-like val config.
"""
import os
import sys

for k in ("OSL_LOCAL_WIND_OBS", "OSL_DEPLOY_ANEMO", "GADEN_ANEMO_FRAME", "GADEN_FAITHFUL_WIND"):
    os.environ.setdefault(k, "1")

from pathlib import Path
import argparse
import numpy as np
import torch

_PKG = Path("champ_far02_python_eval").resolve()
sys.path.insert(0, str(_PKG))
from cf_eval.envs.gas_source_env import GasSourceEnv
from cf_eval.models.actor_critic import ActorCriticDualBackbone
from cf_eval.test.gaden_loader import load_full_map, GadenGasField

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--maps", default="4rooms,uleft,uright,labyrinth_left,labyrinth_right,many_rooms,ultimate")
ap.add_argument("--eps", type=int, default=5)
a = ap.parse_args()

DEV = torch.device("cpu")
agent = ActorCriticDualBackbone()
ck = torch.load(a.ckpt, map_location=DEV, weights_only=False)
agent.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck, strict=False)
agent.eval()


def greedy(o):
    with torch.no_grad():
        return agent._actor_dist(agent._encode_shared(o)).mean.cpu().numpy()[0]


def fresh(fm):
    gf = fm.get("gas_field")
    if gf is None:
        return None
    g = GadenGasField(gf._dir, gf._start_iteration, gf._max_iteration, tuple(gf._origin),
                      z_query=gf._z, save_dt=gf.save_dt)
    g.set_occupancy(fm["grid"].grid, fm["grid"].resolution)
    g.set_iters_per_step(5)
    return g


def blank_runs(seq):
    """mean length of consecutive no-detection runs (plume intermittency)."""
    runs, cur = [], 0
    for v in seq:
        if v:
            if cur:
                runs.append(cur); cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


print(f"GADEN gas-signal statistics  ckpt={os.path.basename(a.ckpt)}  eps={a.eps}\n")
print(f"  {'map':17s} {'succ':>5} {'detect%':>8} {'t_first':>8} {'blank':>7} {'steps':>6}")
print("  " + "-" * 58)
agg = []
for mp in a.maps.split(","):
    try:
        fm = load_full_map(Path("gaden_scenarios"), Path("base/gaden_maps/recommended_configs.yaml"),
                           mp, replay_gas=True)
    except Exception as e:
        print(f"  {mp:17s}  skip ({type(e).__name__})")
        continue
    md = {k: fm[k] for k in ("grid", "source_pos", "robot_pos", "width", "height")}
    md["start_time"] = fm.get("start_time", 0)
    det, tf, bl, sc, st = [], [], [], [], []
    for ep in range(a.eps):
        env = GasSourceEnv()
        obs, _ = env.reset(seed=7000 + ep, options={
            "map_data": md, "wind_field": fm["wind_field"], "gas_field": fresh(fm),
            "deploy_motion": True, "sensor_noise": False})
        seq, succ = [], 0.0
        while True:
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            obs, r, term, trunc, info = env.step(greedy(o))
            seq.append(int(info.get("gas_reading", 0)))
            if term:
                succ = 1.0
            if term or trunc:
                break
        hits = np.nonzero(seq)[0]
        det.append(float(np.mean(seq)))
        tf.append(float(hits[0]) if hits.size else float("inf"))
        bl.append(blank_runs(seq))
        sc.append(succ); st.append(len(seq))
    finite = [t for t in tf if np.isfinite(t)]
    tfs = f"{np.mean(finite):.0f}" if finite else "never"
    if finite and len(finite) < len(tf):
        tfs += f"({len(finite)}/{len(tf)})"
    print(f"  {mp:17s} {100*np.mean(sc):4.0f}% {100*np.mean(det):7.1f}% {tfs:>8} "
          f"{np.mean(bl):7.1f} {np.mean(st):6.0f}")
    agg.append((100 * np.mean(det), np.mean(bl)))

if agg:
    d = np.mean([x[0] for x in agg]); b = np.mean([x[1] for x in agg])
    print(f"\n  GADEN mean detect% = {d:.1f}%   mean blank-run = {b:.1f} steps")
    print(f"  SURROGATE val (training physics) detect% = 71%   <-- the gap to close")
