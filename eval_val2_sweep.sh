#!/bin/bash
#
# eval_val2_sweep.sh — run the Layer-2 validator over every checkpoint of a run.
#
#   sbatch eval_val2_sweep.sh <run_dir> [extra eval_val2.py args...]
#
# Purpose right now: check whether val2 SATURATES on the physics-matched gasfix arms
# (their training gas == val2 gas, so they may climb higher than 25311's 91% ceiling),
# and start building the pre-registered selection curve. This touches NO GADEN data.
#
# The slice reference stays the 25311 upd4000 policy for EVERY arm, so the sparse slice
# is the same 49 maps everywhere and slice numbers are comparable across arms.
#
#SBATCH --job-name=val2_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=6:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"

RUN_DIR="${1:?usage: sbatch eval_val2_sweep.sh <run_dir> [extra args]}"; shift || true
CKDIR="${RUN_DIR}/checkpoints"
REF="reinforcement_learning/runs/ppo_localwind_20260626_162308_job25311/checkpoints/localwind_agent_1048576000.pt"

# all agent checkpoints, sorted by training step; CKPT_STRIDE=n keeps every n-th
# (counted from the last, so the newest checkpoint is always included)
STRIDE="${CKPT_STRIDE:-1}"
ALL=($(ls "${CKDIR}" | grep -oP 'localwind_agent_\K\d+(?=\.pt)' | sort -n))
CKPTS=()
for i in "${!ALL[@]}"; do
    if (( (${#ALL[@]} - 1 - i) % STRIDE == 0 )); then
        CKPTS+=("${CKDIR}/localwind_agent_${ALL[$i]}.pt")
    fi
done
echo "sweeping ${#CKPTS[@]} checkpoints from ${CKDIR}"

"${VENV_PY}" -u eval_val2.py "${CKPTS[@]}" \
    --ref "${REF}" \
    --cases 486 \
    --envs 256 \
    --episodes 1536 \
    --slice-episodes 768 \
    --budget 600 \
    --start-pct 50,100 \
    --slice-pct 10 \
    "$@"
