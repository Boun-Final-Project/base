"""
Visualize a trained agent (greedy) on one hardcoded map per template.

Maps are drawn from test/test_envs.json by env_id so that the visualized
environments exactly match the held-out test set.

The script reads the run's config.json (found next to the checkpoints directory)
to reproduce the exact environment settings used during training — including the
gas dispersion model (filament or IGDM/Gaussian).  The latest checkpoint in the
run is loaded automatically.

Output: one sub-folder per template under test/test_env_test_steps/<run-name>/, each
containing per-step PNG frames (step_XXXX.png) produced by StepVisualizer.

Usage
-----
    python3 visualize_agent.py ppo_dual-010
    python3 visualize_agent.py filament_run_001 --device cpu
    python3 visualize_agent.py filament_run_001 --output-dir /tmp/viz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — make the top-level package importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_RL_ROOT = _SCRIPT_DIR.parent          # …/rl_5_channel/
_PKG_ROOT = _RL_ROOT.parent            # …/src/base/  (parent of the package)
sys.path.insert(0, str(_PKG_ROOT))

from rl_5_channel import config as cfg
from rl_5_channel.envs.gas_source_env import GasSourceEnv
from rl_5_channel.models.actor_critic import (
    ActorCritic,
    ActorCriticDualBackbone,
    ActorCriticModular,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEMPLATE_NAMES = {
    0: "Empty",
    1: "Single_Wall",
    2: "U-Shape",
    3: "Three_Walls",
    4: "Complex_Maze",
    5: "Multi-Room",
}

# One env_id per template (indices into test_envs.json)
ENV_IDS = [
    # 3, 
    # 13, 
    # 33, 
    # 43, 
    # 51, 
    83
    ]

_SKIP_KEYS = {"seed", "template", "arch", "curriculum", "anneal_lr",
              "anneal_start", "target_kl", "output_dir"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_run_config(run_cfg: dict) -> None:
    """Patch the cfg module in-place with values from a run's config.json.

    Only uppercase keys that already exist in cfg are applied; lowercase
    metadata keys (arch, seed, …) are ignored.  After patching, STATE_DIM
    is recalculated from the (possibly updated) LIDAR_NUM_RAYS and
    GAS_HISTORY_LENGTH so the agent's input dimension is always correct.

    If GAS_MODEL is absent from the config (older runs trained before the
    filament model was added), it defaults to "igdm".

    If SENSOR_THRESHOLD_DECAY is absent (older runs trained before decaying
    threshold was added), it defaults to 1.0 (no decay).
    """
    run_cfg = dict(run_cfg)

    if "GAS_MODEL" not in run_cfg:
        run_cfg["GAS_MODEL"] = "filament" if "FILAMENTS_PER_STEP" in run_cfg else "igdm"

    if "SENSOR_THRESHOLD_DECAY" not in run_cfg:
        run_cfg["SENSOR_THRESHOLD_DECAY"] = 1.0

    for key, val in run_cfg.items():
        if key in _SKIP_KEYS:
            continue
        if key.isupper() and hasattr(cfg, key):
            setattr(cfg, key, val)

    # Recalculate derived constant
    cfg.STATE_DIM = (
        cfg.GAS_HISTORY_LENGTH * cfg.GAS_FEATURES_PER_STEP
        + cfg.LIDAR_NUM_RAYS + 2 + 2 + 1
    )


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    checkpoints = sorted(
        ckpt_dir.glob("agent_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No agent_*.pt files found in {ckpt_dir}")
    return checkpoints[-1]


def load_agent(ckpt_path: Path, arch: str, device: torch.device):
    # Constructor defaults are bound at import time, before apply_run_config
    # patches cfg.  Pass the current cfg values explicitly so internal layer
    # sizes match the checkpoint.
    gas_len = cfg.GAS_HISTORY_LENGTH * cfg.GAS_FEATURES_PER_STEP
    common = dict(
        obs_dim=cfg.STATE_DIM,
        gas_len=gas_len,
        lidar_len=cfg.LIDAR_NUM_RAYS,
        gru_hidden=cfg.GAS_GRU_HIDDEN,
        conv_channels=cfg.LIDAR_CONV_CHANNELS,
        conv_kernel=cfg.LIDAR_CONV_KERNEL,
    )

    if arch == "dual":
        agent = ActorCriticDualBackbone(**common)
    elif arch == "modular":
        agent = ActorCriticModular(**common)
    else:
        agent = ActorCritic(obs_dim=cfg.STATE_DIM)

    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.to(device).eval()
    return agent


def greedy_action(agent, arch: str, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        if arch == "dual":
            encoded = agent._encode_shared(obs_t)
            dist = agent._actor_dist(encoded)
            return dist.mean.cpu().numpy()
        else:
            if arch == "modular":
                features = agent.fusion(agent._encode(obs_t))
            else:
                features = agent.backbone(obs_t)
            ab = agent.actor(features)
            alpha = ab[:, 0:1] + 1.0
            beta  = ab[:, 1:2] + 1.0
            return (alpha / (alpha + beta)).cpu().numpy()


def run_episode(agent, arch: str, env: GasSourceEnv,
                device: torch.device) -> dict:
    obs, _ = env.reset()
    env.render()

    total_reward = 0.0
    steps = 0
    success = False

    while True:
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=device).unsqueeze(0)
        action = greedy_action(agent, arch, obs_t)
        obs, reward, terminated, truncated, info = env.step(action[0])
        total_reward += reward
        steps += 1
        env.render()

        if terminated:
            success = True
            break
        if truncated:
            break

    return {
        "reward": total_reward,
        "steps": steps,
        "success": success,
        "final_distance": info["distance_to_source"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_name", help="Name of the run (e.g. ppo_dual-010)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Root output directory.  Plots are written to "
             "<output-dir>/<run-name>/template_N_<name>/ "
             "(default: test/test_env_test_steps/)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="torch device ('cuda' or 'cpu').  Auto-detected if omitted.",
    )
    args = parser.parse_args()

    run_name = args.run_name

    # Locate run directory
    run_dir = _RL_ROOT / "runs" / run_name
    if not run_dir.exists():
        sys.exit(f"[error] Run directory not found: {run_dir}")

    # Apply run config to cfg module
    config_path = run_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"[error] config.json not found at {config_path}")
    with open(config_path) as f:
        run_cfg = json.load(f)
    apply_run_config(run_cfg)

    arch = run_cfg.get("arch", "flat")
    gas_model = cfg.GAS_MODEL
    print(f"Run:       {run_name}")
    print(f"Arch:      {arch}")
    print(f"Gas model: {gas_model}")
    print(f"STATE_DIM: {cfg.STATE_DIM}")

    # Load agent
    ckpt_path = find_latest_checkpoint(run_dir / "checkpoints")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    agent = load_agent(ckpt_path, arch, device)
    print(f"Checkpoint: {ckpt_path.name}  (device={device})\n")

    # Output root
    if args.output_dir is None:
        output_root = _RL_ROOT / "test" / "test_env_test_steps" / run_name
    else:
        output_root = Path(args.output_dir) / run_name

    # Load test environments from the canonical JSON
    test_envs_path = _RL_ROOT / "test" / "test_envs.json"
    if not test_envs_path.exists():
        sys.exit(f"[error] test_envs.json not found at {test_envs_path}")
    with open(test_envs_path) as f:
        test_envs = json.load(f)["envs"]

    # Index by env_id for fast lookup
    envs_by_id = {e["env_id"]: e for e in test_envs}

    for env_id in ENV_IDS:
        entry = envs_by_id.get(env_id)
        if entry is None:
            print(f"  env_id {env_id}: not found in test_envs.json, skipping.")
            continue

        tid = entry["template_id"]
        env_seed = entry["seed"]
        template_label = f"template_{tid}_{TEMPLATE_NAMES[tid]}"
        viz_dir = output_root / template_label

        env = GasSourceEnv(
            seed=env_seed,
            template_id=tid,
            viz_output_dir=str(viz_dir),
        )

        result = run_episode(agent, arch, env, device)

        flag = "SUCCESS" if result["success"] else "timeout"
        print(
            f"  env {env_id:3d}  Template {tid} ({TEMPLATE_NAMES[tid]:12s}) [{flag:7s}]  "
            f"reward={result['reward']:+8.1f}  steps={result['steps']:3d}  "
            f"dist={result['final_distance']:.2f}m  "
            f"→ {viz_dir}"
        )

    print(f"\nDone.  All frames saved under {output_root}/")


if __name__ == "__main__":
    main()
