#!/bin/bash
#
# eval_val2_metatest.sh — Layer-4 meta-test of the Layer-2 validator (eval_val2.py).
#
# Runs the new val (unbiased cohort estimator + paired starts + GADEN-faithful gas + far
# starts + sparse-signal slice) over the 16 checkpoints of job 25311 whose GADEN test
# scores are already known (the GADEN dict in eval_spl_val.py), and prints spearman +
# argmax-regret per metric. This is the ONE honest shot the validator gets: if no metric
# can rank these checkpoints, the validator is dead — no tweaking against this table.
#
# Needs one GPU; queues behind the two gasfix training runs if none is free.
#
#SBATCH --job-name=val2_meta
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

CKDIR="reinforcement_learning/runs/ppo_localwind_20260626_162308_job25311/checkpoints"
# the 16 checkpoints with known GADEN test scores (multiples of 131,072,000 steps)
CKPTS=()
for i in $(seq 1 16); do
    CKPTS+=("${CKDIR}/localwind_agent_$((i * 131072000)).pt")
done
# reference policy for the per-map detect% slice = upd4000 (GADEN's own best)
REF="${CKDIR}/localwind_agent_1048576000.pt"

"${VENV_PY}" -u eval_val2.py "${CKPTS[@]}" \
    --ref "${REF}" \
    --cases 486 \
    --envs 256 \
    --episodes 1536 \
    --slice-episodes 768 \
    --budget 600 \
    --start-pct 50,100 \
    --slice-pct 10 \
    --gaden-corr \
    "$@"
