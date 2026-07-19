#!/bin/bash
#
# train_localwind_gasfix_resume.sh — CONTINUE the under-converged gasfix seed-1 arm (job 25963).
#
# 25963 was scanceled at upd 5110/8000 while its val2 curve was still climbing (best 83.9% at
# upd 5100 vs seed 2's clean 91.7% peak). This resumes from the upd-5100 checkpoint and runs the
# REMAINING 2900 updates so seed 1 gets the same total budget (8000) as control and seed 2.
#
# Differences from train_localwind_gasfix.sh, all to CONTINUE (not restart) the original schedule:
#   --resume <upd5100 ckpt>      weights only; optimizer restarts (same as any finetune here)
#   --updates 2900               the remaining budget (5100 + 2900 = 8000)
#   --gas-k-start 0.0005         K stays at corrected physics from update 0 — do NOT re-anneal
#                                from 0.02 (the original anneal finished at upd 2400)
#   --lr 2.2e-4                  the original 3e-4 schedule's value at progress 5100/8000; decays
#                                to 1e-4 by the end, matching the original schedule's endpoint
#   tier curriculum              restarts at T0 but the rs>0.8 gate fast-forwards a converged
#                                policy back to T9 within a few dozen updates
#
# Selection stays pre-registered: sweep val2, take best succ. Then the ONE-SHOT test29 for the
# final seed-1 number (the upd-5100 test result, 67%, is the stopped-early datapoint).
#
#SBATCH --job-name=ppo_lw_gasfix_r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=36:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"

SEED="${SEED:-1}"
RESUME_CKPT="${RESUME_CKPT:-reinforcement_learning/runs/ppo_lw_gasfix_s1_20260714_100456_job25963/checkpoints/localwind_agent_1336934400.pt}"
RUN_NAME="ppo_lw_gasfix_s${SEED}r_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="${SCRIPT_DIR}/reinforcement_learning/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

"${VENV_PY}" -u train_localwind_gpu.py \
    --wind cfd \
    --cfd-dirs cfd_test/library_v4_4dir \
    --holdout-shards shard_38,shard_39 \
    --flip \
    --envs 1024 \
    --rollout 256 \
    --updates 2900 \
    --cfd-cases 3000 \
    --val-cases 600 \
    --eval-every 100 \
    --lr 2.2e-4 \
    --seed "${SEED}" \
    --resume "${RESUME_CKPT}" \
    --gas-k 0.0005 \
    --gas-k-start 0.0005 \
    --gas-anneal-frac 0.0 \
    --gas-sigma 0.10 \
    --gas-fps 5 \
    --out "${OUT_DIR}" \
    "$@"
