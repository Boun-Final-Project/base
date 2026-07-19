#!/bin/bash
#
# eval_sweep_job25312.sh — GADEN sweep for GPU local-wind step=0.3 (job 25312).
# Same convention as job 25311: flip-ON, OSL_LOCAL_WIND_OBS=1.
# Job still running at submit time — skips checkpoints not yet written.
#
#SBATCH --job-name=sweep_lw_s03
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --time=6:0:0 --partition=batch --qos=users --account=users

set -euo pipefail
OSL_ROOT="${SLURM_SUBMIT_DIR:-/comp04-storage/efe-mantaroglu/osl}"
cd "${OSL_ROOT}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"
EVAL="${OSL_ROOT}/champ/train_gpu_localwind/gpu_fromscratch_gaden_eval.py"
CKPT_DIR="${OSL_ROOT}/champ/train_gpu_localwind/reinforcement_learning/runs/ppo_localwind_s03_20260627_023509_job25312/checkpoints"

echo "====== GADEN sweep — job 25312 GPU local-wind step=0.3, flip-ON ======"
echo ""

for CKPT in \
    "${CKPT_DIR}/localwind_agent_131072000.pt" \
    "${CKPT_DIR}/localwind_agent_262144000.pt" \
    "${CKPT_DIR}/localwind_agent_393216000.pt" \
    "${CKPT_DIR}/localwind_agent_524288000.pt" \
    "${CKPT_DIR}/localwind_agent_655360000.pt" \
    "${CKPT_DIR}/localwind_agent_786432000.pt" \
    "${CKPT_DIR}/localwind_agent_917504000.pt" \
    "${CKPT_DIR}/localwind_agent_1048576000.pt" \
    "${CKPT_DIR}/localwind_agent_1179648000.pt" \
    "${CKPT_DIR}/localwind_agent_1310720000.pt" \
    "${CKPT_DIR}/localwind_agent_1441792000.pt" \
    "${CKPT_DIR}/localwind_agent_1520435200.pt"; do
    if [[ ! -f "${CKPT}" ]]; then
        echo "SKIP (not yet saved): $(basename ${CKPT})"
        continue
    fi
    echo "====== $(basename ${CKPT}) ======"
    "${VENV_PY}" "${EVAL}" "${CKPT}" --eps 10
    echo ""
done
