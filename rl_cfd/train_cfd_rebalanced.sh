#!/bin/bash
# Run B: same as train_cfd.sh (local-cell wind ctx) plus rebalanced rewards.
# A/B comparison vs train_cfd.sh — both should be launched from the same
# code state so only the reward profile differs.
#
# Reward profile:
#   R_SUCCESS    15.0    (was 200)   — terminal bonus 5× the next-largest, doesn't dominate variance
#   R_COLLISION  -3.0    (was -10)   — rebalanced to the new success scale
#   R_MAX_STEPS  -3.0    (was -20)   — same
#   R_NEW_CELL   0.05    (was 0.2)   — coverage:step ratio 1:1, no Goodhart farming
#   R_STEP       -0.05   (was -0.1)
#   R_DETECTION  0.1     (was 0.2)   — keep relative pressure
#SBATCH --job-name=ppo_cfd_rb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=72:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users
#SBATCH --output=/comp04-storage/efe-mantaroglu/osl/base/rl_cfd/runs/slurm-%j.out

set -euo pipefail
cd /comp04-storage/efe-mantaroglu/osl

RUN_NAME="ppo_cfd_rebalanced_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="base/rl_cfd/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

VENV_PY=/home/efe-mantaroglu/simenv/bin/python

${VENV_PY} -u -m base.rl_cfd.training.train \
    --arch spatial \
    --batched-obs \
    --num-envs 48 \
    --rollout-length 1024 \
    --entropy-coeff 0.02 \
    --entropy-coeff-end 0.005 \
    --target-kl 0.02 \
    --r-success 15.0 \
    --r-collision -3.0 \
    --r-max-steps -3.0 \
    --r-new-cell 0.05 \
    --r-step -0.05 \
    --r-detection 0.1 \
    --output-dir "${OUT_DIR}"
