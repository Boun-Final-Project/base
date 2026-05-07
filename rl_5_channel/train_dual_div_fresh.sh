#!/bin/bash
# A/B fresh comparator to train_dual_div.sh (which resumes from friend's
# checkpoint). Same architecture (dual: gas_gru + lidar_conv + gated
# fusion + actor/critic with residual blocks), same diversified
# 10-template curriculum, same seed (42) — but starts from random init.
#
# Tells us whether the friend's 183.5M-step pretraining helps or hurts
# generalization to the new map distribution. If fresh ends up close
# to resume at matched step count, the pretraining was either neutral
# or already-saturated.
#SBATCH --job-name=ppo_dual_div_fresh
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
cd /comp04-storage/efe-mantaroglu/osl/base-rl-map-extension

RUN_NAME="ppo_dual_div_fresh_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="rl_5_channel/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

VENV_PY=/home/efe-mantaroglu/simenv/bin/python

${VENV_PY} -u -m rl_5_channel.training.train \
    --arch dual \
    --num-envs 48 \
    --rollout-length 1024 \
    --total-timesteps 100000000 \
    --curriculum \
    --seed 42 \
    --output-dir "${OUT_DIR}"
