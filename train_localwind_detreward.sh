#!/bin/bash
#
# train_localwind_detreward.sh — DETECTION-REWARD-SHAPE finetune off the gasfix s2 checkpoint.
#
# Diagnosis (real-GADEN trajectory logs, 2026-07-15): the deployed s2 policy LOITERS in the
# plume. R_DETECTION=+0.75 is paid EVERY in-gas step, so on a plume-filled map the agent banks
# ~+96 just for dwelling (~half of R_SUCCESS=200) and there is no reward gradient distinguishing
# 8 m-in-gas from 0.5 m-in-gas. It roams the whole plume instead of closing on the source
# (10x6 median steps-to-success 135 control -> 164 s2; a successful run reached 1.5 m at step 40
# then wandered back out for 143 more steps before committing).
#
# This finetune changes ONLY the detection-reward TRIGGER (--r-det-mode), one variable vs s2:
#   R_DET_MODE=edge  (arm B) : pay +r_det on each 0->1 re-acquisition. Self-adapting — collapses to
#                              one-shot on a continuous plume (kills the loiter) but keeps rewarding
#                              re-finding an INTERMITTENT plume (protects many_rooms/ultimate).
#   R_DET_MODE=once  (arm A) : pay +r_det only on FIRST contact of the episode. Maximal anti-loiter;
#                              risk = strips the reward that protects sparse-plume tracking.
# The obs is untouched (real per-step binary still feeds gas_xyb); only the reward term changes.
#
# Everything else is byte-identical to s2 (job 25964): resume its upd3800 ckpt, corrected gas
# (K=0.0005 held, NO re-anneal), flip ON, library_v4_4dir, holdout shard_38/39.
#
# SELECTION = val2 (eval_val2_sweep.sh). Unlike GADEN success ranking, val2 reports SPL / detour /
# steps directly, so EFFICIENCY is in-distribution and val2 is a legitimate selector here. Rule:
# best efficiency (SPL up / detour down / steps down) SUBJECT TO succ (esp. sparse slice) not
# regressing vs s2. Then one-shot test29 + ROS.
#
# Usage:
#   R_DET_MODE=edge sbatch train_localwind_detreward.sh     # arm B (recommended)
#   R_DET_MODE=once sbatch train_localwind_detreward.sh     # arm A
#
#SBATCH --job-name=ppo_lw_detrew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"

MODE="${R_DET_MODE:?set R_DET_MODE=edge or R_DET_MODE=once}"
RESUME_CKPT="${RESUME_CKPT:-reinforcement_learning/runs/ppo_lw_gasfix_s2_20260714_111531_job25964/checkpoints/localwind_agent_996147200.pt}"
RUN_NAME="ppo_lw_detrew_${MODE}_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="${SCRIPT_DIR}/reinforcement_learning/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

# lr 1e-4 (finetune); ~2000 updates to re-equilibrate the critic after removing the dwell reward.
# gas held at corrected physics from update 0 (anneal-frac 0 => gas-k-start==gas-k, no re-anneal).
"${VENV_PY}" -u train_localwind_gpu.py \
    --wind cfd \
    --cfd-dirs cfd_test/library_v4_4dir \
    --holdout-shards shard_38,shard_39 \
    --flip \
    --envs 1024 \
    --rollout 256 \
    --updates 2000 \
    --cfd-cases 3000 \
    --val-cases 600 \
    --eval-every 100 \
    --lr 1e-4 \
    --seed 2 \
    --resume "${RESUME_CKPT}" \
    --r-det-mode "${MODE}" \
    --gas-k 0.0005 \
    --gas-k-start 0.0005 \
    --gas-anneal-frac 0.0 \
    --gas-sigma 0.10 \
    --gas-fps 5 \
    --out "${OUT_DIR}" \
    "$@"
