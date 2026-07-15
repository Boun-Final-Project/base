#!/bin/bash
#SBATCH --job-name=eval_winners_25311
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --time=2:0:0 --partition=batch --qos=users --account=users
set -euo pipefail
OSL_ROOT="${SLURM_SUBMIT_DIR:-/comp04-storage/efe-mantaroglu/osl}"
cd "${OSL_ROOT}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"
EVAL="${OSL_ROOT}/champ/train_gpu_localwind/gpu_fromscratch_gaden_eval.py"
CKPT_DIR="${OSL_ROOT}/champ/train_gpu_localwind/reinforcement_learning/runs/ppo_localwind_20260626_162308_job25311/checkpoints"

echo "====== upd 4000 (1048M steps) — 83% in 10-ep sweep ======"
"${VENV_PY}" "${EVAL}" "${CKPT_DIR}/localwind_agent_1048576000.pt" --eps 20

echo ""
echo "====== upd 6000 (1572M steps) — 83% + many_rooms 20% in 10-ep sweep ======"
"${VENV_PY}" "${EVAL}" "${CKPT_DIR}/localwind_agent_1572864000.pt" --eps 20
