#!/bin/bash
# Resumes friend's dual-arch checkpoint (agent_183500800.pt, 183.5M steps,
# ~500k params, gas_gru + lidar_conv + gated fusion) and continues training
# on the diversified 10-template curriculum (T0-T9).
#
# Total budget set to 300M so the resume picks up from 183.5M and adds
# ~117M more steps. --reset-lr means fresh LR (don't keep the friend's
# annealed end-LR), since we're now fine-tuning on a different map
# distribution.
#
# Note: friend's other scripts on feature/rl use --arch spatial, but the
# given checkpoint is a dual-arch model (different observation interface,
# 28x smaller). This script matches the checkpoint's architecture.
#SBATCH --job-name=ppo_dual_div
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
# cd into the worktree (not osl/) — see train_spatial_div.sh for why.
cd /comp04-storage/efe-mantaroglu/osl/base-rl-map-extension

RUN_NAME="ppo_dual_div_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="rl_5_channel/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

VENV_PY=/home/efe-mantaroglu/simenv/bin/python
RESUME_CKPT=/comp04-storage/efe-mantaroglu/osl/agent_183500800.pt

${VENV_PY} -u -m rl_5_channel.training.train \
    --arch dual \
    --num-envs 48 \
    --rollout-length 1024 \
    --total-timesteps 300000000 \
    --curriculum \
    --resume "${RESUME_CKPT}" \
    --reset-lr \
    --output-dir "${OUT_DIR}"
