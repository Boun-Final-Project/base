#!/bin/bash
#
# train_champ.sh — reproduce the champion gas-source-localization agent.
#
# Launches the exact PPO recipe that produced the deployed champion checkpoint
# `agent_91750400.pt`: a 200M-step, dual-architecture run (originally
# `runs/lidar-007`), whose best / early-stopped checkpoint was taken at
# ~91.75M steps.
#
# The full hyperparameter snapshot from that run is committed next to this
# script as `champ_config.json`. The CLI flags below set every value that
# defines the champion run; all remaining hyperparameters (rewards, entropy,
# minibatches, filament plume model, etc.) come from `config.py` on this
# branch, which already matches champ_config.json — notably R_STEP=-1.0 and
# R_DETECTION=0.75. If you port this script to another branch, verify those
# two reward values against champ_config.json first.
#
# Usage:
#   sbatch reinforcement_learning/train_champ.sh     # on SLURM
#   bash   reinforcement_learning/train_champ.sh     # locally
#   VENV_PY=/path/to/python bash reinforcement_learning/train_champ.sh
#
#SBATCH --job-name=ppo_champ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=72:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users

set -euo pipefail

# Resolve the repo root from this script's location (portable, no hardcoded
# absolute paths) and run from there so the module import path resolves.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Python interpreter. The env requires Python >= 3.12; override with VENV_PY.
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"

RUN_NAME="ppo_champ_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="reinforcement_learning/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

# --- Champion recipe (source: champ_config.json / runs/lidar-007) ------------
#   dual architecture, 256 parallel envs, curriculum on,
#   clip-epsilon 0.3, target-KL 0.05, LR 3e-4 annealed from 50% of training,
#   seed 1, 200M total steps (champion = early-stopped best at ~91.75M).
"${VENV_PY}" -u -m reinforcement_learning.training.train \
    --arch dual \
    --num-envs 256 \
    --rollout-length 1024 \
    --total-timesteps 200000000 \
    --clip-epsilon 0.3 \
    --lr 3e-4 \
    --anneal-lr --anneal-start 0.5 \
    --target-kl 0.05 \
    --curriculum \
    --seed 1 \
    --output-dir "${OUT_DIR}"
