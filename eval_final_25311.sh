#!/bin/bash
#SBATCH --job-name=eval_final_25311
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --time=2:0:0 --partition=batch --qos=users --account=users
set -euo pipefail
OSL_ROOT="${SLURM_SUBMIT_DIR:-/comp04-storage/efe-mantaroglu/osl}"
cd "${OSL_ROOT}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"
EVAL="${OSL_ROOT}/champ/train_gpu_localwind/gpu_fromscratch_gaden_eval.py"
CKPT_DIR="${OSL_ROOT}/champ/train_gpu_localwind/reinforcement_learning/runs/ppo_localwind_20260626_162308_job25311/checkpoints"
for CKPT in \
    "${CKPT_DIR}/localwind_agent_1835008000.pt" \
    "${CKPT_DIR}/localwind_agent_1966080000.pt" \
    "${CKPT_DIR}/localwind_agent_2097152000.pt" \
    "${CKPT_DIR}/localwind_final.pt"; do
    echo "====== $(basename ${CKPT}) ======"
    "${VENV_PY}" "${EVAL}" "${CKPT}" --eps 10
    echo ""
done
