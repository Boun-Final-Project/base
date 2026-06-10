#!/bin/bash
# Precompute + cache far-plume-hit placement (far_placement.json) for every
# case in the CFD libraries, so training resets stay cheap. CPU-only.
#
# Usage: sbatch precompute_far_placement.sh "<lib1>,<lib2>" <rl-package-path>
#SBATCH --job-name=far_placement
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --output=/comp04-storage/efe-mantaroglu/osl/%x-%j.out

set -euo pipefail
LIB_DIR=${1:?"usage: sbatch precompute_far_placement.sh <lib1,lib2> <rl-package-path>"}
RL_PKG=${2:?"need rl-package-path"}
VENV_PY=${CFD_PYTHON_BIN:-/home/efe-mantaroglu/simenv/bin/python}
CFD_PKG=/comp04-storage/efe-mantaroglu/osl/base/cfd_wind_pipeline
CLEARANCE=${FAR_PLUME_CLEARANCE:-0.6}

echo "Precomputing far-plume placement: libs=${LIB_DIR} clearance=${CLEARANCE}"
${VENV_PY} -u ${CFD_PKG}/far_plume_placement.py \
    --library-dir "${LIB_DIR}" \
    --rl-package-path "${RL_PKG}" \
    --clearance "${CLEARANCE}"
echo "DONE far-placement precompute."
