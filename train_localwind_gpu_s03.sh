#!/bin/bash
#
# train_localwind_gpu_s03.sh — same as train_localwind_gpu.sh but with step_size=0.3m.
#
# Trains with a smaller robot step (0.3m vs default 0.5m), which makes the task harder
# and more faithful to the real robot's motion. Curriculum, flip, and val gate are identical.
#
# Usage:
#   sbatch train_localwind_gpu_s03.sh
#
#SBATCH --job-name=ppo_lw_s03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=72:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"

RUN_NAME="ppo_localwind_s03_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="${SCRIPT_DIR}/reinforcement_learning/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

"${VENV_PY}" -u train_localwind_gpu.py \
    --wind cfd \
    --cfd-dirs cfd_test/library_v4_4dir \
    --holdout-shards shard_38,shard_39 \
    --flip \
    --envs 1024 \
    --rollout 256 \
    --updates 8000 \
    --cfd-cases 3000 \
    --val-cases 600 \
    --eval-every 100 \
    --lr 3e-4 \
    --step-size 0.3 \
    --out "${OUT_DIR}" \
    "$@"
