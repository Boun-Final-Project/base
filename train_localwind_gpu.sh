#!/bin/bash
#
# train_localwind_gpu.sh — GPU local-wind-observation trainer (CFD spatially-varying wind).
#
# Trains the dual policy on the real CFD wind-field library with the GPU env's spatial-wind path:
# the policy OBSERVES the LOCAL wind at the robot (the local2 lever that cracked many_rooms),
# and filaments advect through the spatially-varying field. Reuses the validated spatial-wind GPU
# kernels (gpu_wind.wind_query_bilinear_pool / faithful_obs_wind_pool). Local-wind is a FINETUNE on
# a competent base, so pass --resume <champ_ckpt> (lr 1e-4 default); it also runs from scratch.
#
# Usage:
#   sbatch train_localwind_gpu.sh
#   sbatch train_localwind_gpu.sh --resume /comp04-storage/efe-mantaroglu/osl/ali_champ/agent_91750400.pt
#
#SBATCH --job-name=ppo_localwind_gpu
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
# SLURM spools the batch script to a non-writable dir, so BASH_SOURCE points there under sbatch.
# Prefer SLURM_SUBMIT_DIR (where we launched from = this trainer dir); fall back to BASH_SOURCE
# for direct ./train_localwind_gpu.sh runs.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"

RUN_NAME="ppo_localwind_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="${SCRIPT_DIR}/reinforcement_learning/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

# Wind source = real CFD library v4 (cfd_test/library_v4_4dir), LOCAL-wind obs.
# From scratch, flip-ON (matches the flip-ON GADEN real-gas eval / deploy anemometer).
# Train pool = all shards EXCEPT --holdout-shards; held-out shards form the val gate that
# selects the best checkpoint (localwind_best_val.pt). Random full-map starts by default;
# add --reverse-curriculum to enable the near->far start lever if convergence stalls.
# --cfd-cases caps the train pool (~4000 is plenty per RL norms; raise only if GPU mem/time allow).
"${VENV_PY}" -u train_localwind_gpu.py \
    --wind cfd \
    --cfd-dirs cfd_test/library_v4_4dir \
    --holdout-shards shard_38,shard_39 \
    --flip \
    --envs 256 \
    --rollout 128 \
    --updates 8000 \
    --cfd-cases 3000 \
    --val-cases 600 \
    --eval-every 100 \
    --lr 3e-4 \
    --out "${OUT_DIR}" \
    "$@"
