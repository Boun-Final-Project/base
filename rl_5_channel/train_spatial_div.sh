#!/bin/bash
# Trains rl_5_channel from feature/rl-map-extension worktree with the
# diversified 10-template curriculum (T0-T9). Mirrors friend's
# train_spatial.sh structure; --curriculum is added so the new
# TEMPLATE_CURRICULUM_STAGES + TEMPLATE_SAMPLING_WEIGHTS actually
# engage (without it, all templates fire uniformly from step 0 and
# the curriculum schedule is bypassed).
#SBATCH --job-name=ppo_spatial_div
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=72:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users
#SBATCH --output=/comp04-storage/efe-mantaroglu/osl/base-rl-map-extension/rl_5_channel/runs/slurm-%j.out

set -euo pipefail
# IMPORTANT: cd into the worktree, not osl/. The training module is
# invoked as `rl_5_channel.training.train` (not `base.rl_5_channel...`)
# because the worktree path contains a hyphen, which Python cannot use
# as a package name. Relative imports inside the code work regardless
# of how the top level is named.
cd /comp04-storage/efe-mantaroglu/osl/base-rl-map-extension

RUN_NAME="ppo_spatial_div_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="rl_5_channel/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

VENV_PY=/home/efe-mantaroglu/simenv/bin/python

${VENV_PY} -u -m rl_5_channel.training.train \
    --arch spatial \
    --batched-obs \
    --num-envs 48 \
    --rollout-length 1024 \
    --curriculum \
    --output-dir "${OUT_DIR}"
