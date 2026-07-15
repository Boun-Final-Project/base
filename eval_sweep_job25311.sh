#!/bin/bash
#
# eval_sweep_job25311.sh — quick GADEN sweep across post-val-saturation checkpoints.
# Val saturated at upd 500 so can't rank further; GADEN real-gas eval is the only signal.
# Samples every ~500 updates (131M steps) from upd 500 onward with --eps 10 for speed.
# Run full --eps 20 on the winner separately.
#
#SBATCH --job-name=sweep_lw_25311
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users

set -euo pipefail
OSL_ROOT="${SLURM_SUBMIT_DIR:-/comp04-storage/efe-mantaroglu/osl}"
cd "${OSL_ROOT}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"
EVAL="${OSL_ROOT}/champ/train_gpu_localwind/gpu_fromscratch_gaden_eval.py"
CKPT_DIR="${OSL_ROOT}/champ/train_gpu_localwind/reinforcement_learning/runs/ppo_localwind_20260626_162308_job25311/checkpoints"

# Every 500 updates = 131,072,000 steps apart (1024 envs x 256 rollout x 500 upds)
CKPTS=(
    "${CKPT_DIR}/localwind_agent_131072000.pt"    # upd  500
    "${CKPT_DIR}/localwind_agent_262144000.pt"    # upd 1000
    "${CKPT_DIR}/localwind_agent_393216000.pt"    # upd 1500
    "${CKPT_DIR}/localwind_agent_524288000.pt"    # upd 2000
    "${CKPT_DIR}/localwind_agent_655360000.pt"    # upd 2500
    "${CKPT_DIR}/localwind_agent_786432000.pt"    # upd 3000
    "${CKPT_DIR}/localwind_agent_917504000.pt"    # upd 3500
    "${CKPT_DIR}/localwind_agent_1048576000.pt"   # upd 4000
    "${CKPT_DIR}/localwind_agent_1179648000.pt"   # upd 4500
    "${CKPT_DIR}/localwind_agent_1310720000.pt"   # upd 5000
    "${CKPT_DIR}/localwind_agent_1441792000.pt"   # upd 5500
    "${CKPT_DIR}/localwind_agent_1572864000.pt"   # upd 6000
    "${CKPT_DIR}/localwind_agent_1703936000.pt"   # upd 6500
    "${CKPT_DIR}/localwind_agent_1782579200.pt"   # upd 6800 (latest at submit time)
)

echo "====== GADEN sweep — job 25311 (flip-ON, step=0.5, CFD local-wind from scratch) ======"
echo "Each checkpoint: 10 eps x 7 maps. Best will be re-evaluated at 20 eps."
echo ""

BEST_OVERALL=-1
BEST_CKPT=""

for CKPT in "${CKPTS[@]}"; do
    if [[ ! -f "${CKPT}" ]]; then
        echo "SKIP (not yet saved): $(basename ${CKPT})"
        continue
    fi
    echo "----------------------------------------------------------------------"
    echo "CKPT: $(basename ${CKPT})"
    "${VENV_PY}" "${EVAL}" "${CKPT}" --eps 10
    echo ""
done

echo "====== sweep done — re-run winner at --eps 20 ======"
