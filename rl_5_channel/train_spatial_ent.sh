#!/bin/bash
#SBATCH --job-name=ppo_spatial_ent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=72:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users
#SBATCH --output=/comp04-storage/efe-mantaroglu/osl/base/rl_5_channel/runs/slurm-%j.out

set -euo pipefail
cd /comp04-storage/efe-mantaroglu/osl

RUN_NAME="ppo_spatial_ent_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="base/rl_5_channel/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

VENV_PY=/home/efe-mantaroglu/simenv/bin/python

${VENV_PY} -u -m base.rl_5_channel.training.train \
    --arch spatial \
    --batched-obs \
    --num-envs 48 \
    --rollout-length 1024 \
    --entropy-coeff 0.01 \
    --target-kl 0.02 \
    --curriculum \
    --output-dir "${OUT_DIR}"
