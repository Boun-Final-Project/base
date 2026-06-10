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
# defines the champion run; the remaining recipe-defining values that differ
# from this branch's config.py defaults (rewards R_STEP/R_DETECTION, no loop
# penalty, the slower T0-T5 curriculum) are pinned via the OSL_* env-var
# overrides exported below — config.py reads them, so this script reproduces
# the champion recipe without touching the branch defaults. Everything else
# (entropy, minibatches, filament plume model, ...) already matches
# champ_config.json. One known later improvement is baked into this branch:
# the sub-cell continuous lidar wall distance fix (the original run used the
# coarser cell-quantized lidar).
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

# Pin the champion recipe values that differ from this branch's config.py
# defaults (see header). Verified against champ_config.json.
export OSL_R_STEP=-1.0
export OSL_R_DETECTION=0.75
export OSL_R_LOOP_BASE=0            # champion predates the loop penalty
export OSL_TEMPLATE_STAGES=0:1,0.25:3,0.5:5   # T0-T5 only, slower unlocks

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
