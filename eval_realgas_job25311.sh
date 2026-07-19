#!/bin/bash
#
# eval_realgas_job25311.sh — Python real-gas eval of the two job-25311 checkpoints.
# Runs gpu_fromscratch_gaden_eval.py (flip-ON, replay_gas=True, 20 eps per map)
# on both the upd-400 (74.1% val) and upd-500 (val-saturated) checkpoints.
# Must run from the osl root so champ_far02_python_eval/ and gaden_scenarios/ resolve.
#
#SBATCH --job-name=eval_lw_25311
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users

set -euo pipefail
OSL_ROOT="${SLURM_SUBMIT_DIR:-/comp04-storage/efe-mantaroglu/osl}"
cd "${OSL_ROOT}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"
EVAL="${OSL_ROOT}/champ/train_gpu_localwind/gpu_fromscratch_gaden_eval.py"

echo "====== upd-400 (74.1% val, last discriminative) ======"
"${VENV_PY}" "${EVAL}" \
    best_ckpts/gpu_cfd_localwind_step05_job25311_upd400_val74pct.pt \
    --eps 20

echo ""
echo "====== upd-500 (val-saturated 100%) ======"
"${VENV_PY}" "${EVAL}" \
    best_ckpts/gpu_cfd_localwind_step05_job25311_upd500_val100pct.pt \
    --eps 20
