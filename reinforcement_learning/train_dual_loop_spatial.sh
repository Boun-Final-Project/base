#!/bin/bash
# Fresh training: friend's spatial wind (post-merge, origin/feature/rl head)
# + trajectory-loop proximity penalty (R_LOOP_BASE=-0.05, LOOP_DECAY=0.85,
# D_LOOP=0.3, LOOP_HISTORY=10).
#
# Differs from 5161 in:
#   - spatial wind plumbing (wall-zero uniform during training; local Ux,Uy
#     in obs at robot position; spatial wind respected by plume advection)
#   - obs format (Ux_norm, Uy_norm) instead of (speed_norm, dir_norm)
#   - R_STEP = -1.0 (was -0.3 in 5161)
#   - R_DETECTION = 0.75 (was 2.0 in 5161 stale-branch default)
#   - no R_NEW_CELL / R_REVISIT visited-cell grid (friend removed it)
#   - no R_MAX_STEPS truncation penalty
#   - curriculum unlocks up to T9 (no T10/wall_trap in this run)
#   - weighted template sampling (T6/T7 = 3x weight)
#
# 200M timesteps, seed 456, default LR + anneal, curriculum on.
#SBATCH --job-name=ppo_dual_loop_spatial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=72:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users
#SBATCH --output=/comp04-storage/efe-mantaroglu/osl/friend_base_loop_spatial/reinforcement_learning/runs/slurm-%j.out

set -euo pipefail
cd /comp04-storage/efe-mantaroglu/osl/friend_base_loop_spatial

RUN_NAME="ppo_dual_loop_spatial_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="reinforcement_learning/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

VENV_PY=/home/efe-mantaroglu/simenv/bin/python

${VENV_PY} -u -m reinforcement_learning.training.train \
    --arch dual \
    --num-envs 48 \
    --rollout-length 1024 \
    --total-timesteps 200000000 \
    --curriculum \
    --seed 456 \
    --output-dir "${OUT_DIR}"
