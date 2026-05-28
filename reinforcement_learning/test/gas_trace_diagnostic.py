"""
Gas-trace diagnostic — dump per-step (gas, distance-to-source) for the Python
GADEN-eval harness, in the same line format as the real deployment node.log,
so the two can be diffed apples-to-apples on the same map + checkpoint.

The real node logs lines like:
    [Ep 0 Step  12] Pos (9.19,12.69) ... d2src=11.77m gas=0.000 ...
This script emits the matching subset:
    [Ep 0 Step  12] Pos (x,y) d2src=<m> gas=<conc> gas_bin=<0/1>

Why this exists: §4 of GADEN_EVAL_FINDINGS.md found that big-map failures are a
gas-availability problem, not a step-budget one. The prime suspect is plume
fidelity — does the Python filament plume put gas where real GADEN does? This
dumps the harness's gas trace along the agent's (greedy) trajectory so you can
overlay it on the real sim's node.log gas=/d2src= trace for the same map.

Usage:
    python -m reinforcement_learning.test.gas_trace_diagnostic \
        --checkpoint /path/agent.pt --arch dual --map ultimate \
        [--gaden-root <dir>] [--max-steps 600] [--seed 0] [--out trace.log]

Notes:
- Warmup now scales with the map's recommended start_time (see
  gas_source_env.reset); this script reports the warmup it used so you can
  confirm it matches the real GADEN playback start.
- Greedy (deterministic) actions, matching eval_checkpoints_gaden.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.gas_source_env import GasSourceEnv
from reinforcement_learning.envs.spatial_obs_wrapper import SpatialObsWrapper
from reinforcement_learning.models.actor_critic import ActorCriticDualBackbone
from reinforcement_learning.models.actor_critic_spatial import ActorCriticSpatial
from reinforcement_learning.test.gaden_loader import load_full_map

_DEFAULT_GADEN_ROOT = Path(_SCRIPT_DIR).parents[1] / "gaden_maps"


def _load_agent(checkpoint, arch, device):
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if arch == "dual":
        agent = ActorCriticDualBackbone().to(device)
    elif arch == "spatial":
        agent = ActorCriticSpatial().to(device)
    else:
        raise ValueError(f"unsupported arch '{arch}' (use dual|spatial)")
    agent.load_state_dict(state)
    agent.eval()
    return agent


def _greedy_action(agent, arch, obs, device):
    with torch.no_grad():
        if arch == "dual":
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            enc = agent._encode_shared(obs_t)
            return agent._actor_dist(enc).mean.cpu().numpy()[0]
        spatial, wind = obs
        spatial_t = torch.as_tensor(spatial, dtype=torch.float32, device=device).unsqueeze(0)
        wind_t = torch.as_tensor(wind, dtype=torch.float32, device=device).unsqueeze(0)
        enc = agent._encode_shared(spatial_t, wind_t)
        return agent._actor_dist(enc).mean.cpu().numpy()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--arch", default="dual", choices=["dual", "spatial"])
    p.add_argument("--map", required=True, help="GADEN map key, e.g. ultimate")
    p.add_argument("--gaden-root", default=str(_DEFAULT_GADEN_ROOT))
    p.add_argument("--max-steps", type=int, default=cfg.MAX_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write trace to this file (else stdout)")
    args = p.parse_args()

    device = torch.device("cpu")
    gaden_root = Path(args.gaden_root)
    yaml_path = gaden_root / "recommended_configs.yaml"
    full_map = load_full_map(gaden_root, yaml_path, args.map)
    spec = full_map.get("spec", {})
    start_time = spec.get("start_time", 0)
    src = np.asarray(full_map["source_pos"], dtype=np.float64)

    agent = _load_agent(args.checkpoint, args.arch, device)

    map_data = {k: full_map[k] for k in ("grid", "source_pos", "robot_pos",
                                          "width", "height")}
    map_data["start_time"] = start_time

    if args.arch == "dual":
        env = GasSourceEnv()
    else:
        env = SpatialObsWrapper(GasSourceEnv())
    obs, _ = env.reset(
        seed=args.seed,
        options={"map_data": map_data, "wind_field": full_map["wind_field"]},
    )
    # The underlying GasSourceEnv (unwrapped) holds the plume + robot pos.
    base = env._env if isinstance(env, SpatialObsWrapper) else env
    warmup = int(round(float(start_time) / cfg.FILAMENT_DT)) if start_time else cfg.FILAMENT_WARMUP_STEPS

    lines = []
    hdr = (f"# gas-trace  map={args.map}  ckpt={os.path.basename(args.checkpoint)}  "
           f"arch={args.arch}  start_time={start_time}s  warmup={warmup} steps  "
           f"source=({src[0]:.2f},{src[1]:.2f})  n_filaments_post_warmup={base._plume.n_active}")
    lines.append(hdr)

    n_gas = 0
    min_d = float("inf")
    success = False
    step = 0
    while True:
        pos = np.asarray(base._robot_pos, dtype=np.float64)
        conc = base._plume.concentration_at(pos)
        d2src = float(np.linalg.norm(pos - src))
        gas_bin = 1 if conc > 0.0 else 0
        n_gas += gas_bin
        min_d = min(min_d, d2src)
        lines.append(f"[Ep 0 Step {step:4d}] Pos ({pos[0]:.2f},{pos[1]:.2f}) "
                     f"d2src={d2src:.2f}m gas={conc:.3f} gas_bin={gas_bin}")

        action = _greedy_action(agent, args.arch, obs, device)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated:
            success = True
            break
        if truncated or step >= args.max_steps:
            break

    total = step + 1
    summary = (f"# SUMMARY  steps={total}  success={success}  "
               f"nonzero_gas={n_gas}/{total} ({100*n_gas/total:.1f}%)  "
               f"min_d2src={min_d:.2f}m  result={'SUCCESS' if success else 'max_steps/failed'}")
    lines.append(summary)

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
        print(summary)
        print(hdr)
    else:
        print(text)


if __name__ == "__main__":
    main()
