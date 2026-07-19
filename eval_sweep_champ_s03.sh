#!/bin/bash
#
# eval_sweep_champ_s03.sh — GADEN sweep for champ step=0.3 (uniform wind, no local-wind obs).
# Uses OSL_LOCAL_WIND_OBS=0 to match training convention.
#
#SBATCH --job-name=sweep_champ_s03
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --time=4:0:0 --partition=batch --qos=users --account=users

set -euo pipefail
OSL_ROOT="${SLURM_SUBMIT_DIR:-/comp04-storage/efe-mantaroglu/osl}"
cd "${OSL_ROOT}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"
EVAL="${OSL_ROOT}/champ/train_gpu_localwind/gpu_fromscratch_gaden_eval.py"
CKPT_DIR="${OSL_ROOT}/champ/train/reinforcement_learning/runs/ppo_champ_s03_20260627_030636_job25314/checkpoints"

# Uniform wind, mean obs — override OSL_LOCAL_WIND_OBS before the script's setdefault fires
export OSL_LOCAL_WIND_OBS=0

echo "====== GADEN sweep — job 25314 champ step=0.3 (uniform wind, NO local-wind obs) ======"
echo ""

for CKPT in \
    "${CKPT_DIR}/agent_26214400.pt" \
    "${CKPT_DIR}/agent_52428800.pt" \
    "${CKPT_DIR}/agent_78643200.pt" \
    "${CKPT_DIR}/agent_91750400.pt" \
    "${CKPT_DIR}/agent_104857600.pt" \
    "${CKPT_DIR}/agent_131072000.pt" \
    "${CKPT_DIR}/agent_157286400.pt" \
    "${CKPT_DIR}/agent_183500800.pt" \
    "${CKPT_DIR}/agent_199753728.pt"; do
    echo "====== $(basename ${CKPT}) ======"
    "${VENV_PY}" "${EVAL}" "${CKPT}" --eps 10
    echo ""
done
