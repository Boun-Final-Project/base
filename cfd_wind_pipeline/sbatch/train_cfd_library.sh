#!/bin/bash
# Train PPO using a CFD wind library as the training distribution.
#
# Usage:  sbatch train_cfd_library.sh <library-dir> <rl-package-path> [extra-train-args...]
# Example:
#   sbatch train_cfd_library.sh \
#     /comp04-storage/efe-mantaroglu/osl/cfd_test/library_v2 \
#     /comp04-storage/efe-mantaroglu/osl/friend_base_loop_spatial \
#     --arch dual --num-envs 48 --total-timesteps 200000000
#
# Env vars:
#   CFD_MIX_SYNTHETIC  fraction (0..1) of resets that use the original
#                      MapGenerator+synthetic-wind instead of the library
#                      (default: 0.2 — keeps some procedural diversity as safety)
#SBATCH --job-name=ppo_cfd_lib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --partition=batch
#SBATCH --output=/comp04-storage/efe-mantaroglu/osl/%x-%j.out

set -euo pipefail

LIB_DIR=${1:?"usage: sbatch train_cfd_library.sh <library-dir> <rl-package-path> [train-args...]"}
RL_PKG=${2:?"need rl-package-path"}
shift 2

MIX=${CFD_MIX_SYNTHETIC:-0.2}
VENV_PY=${CFD_PYTHON_BIN:-/home/efe-mantaroglu/simenv/bin/python}
CFD_PKG=/comp04-storage/efe-mantaroglu/osl/base/cfd_wind_pipeline

# Propagate the local-wind-observation flag (read by the RL package config).
# Set OSL_LOCAL_WIND_OBS=1 in the submitting shell to train the policy on
# local point wind instead of the spatial mean. MUST match at eval time.
export OSL_LOCAL_WIND_OBS=${OSL_LOCAL_WIND_OBS:-0}
echo "OSL_LOCAL_WIND_OBS=${OSL_LOCAL_WIND_OBS}"

# --template-filter is a LAUNCHER arg (parsed by train_with_cfd_library.py
# before the `--`), NOT a train.py arg. Pass it via CFD_TEMPLATE_FILTER so it
# lands on the correct side of the `--`. e.g. CFD_TEMPLATE_FILTER=0,1,2,3,4,5
TEMPLATE_FILTER_ARG=()
if [ -n "${CFD_TEMPLATE_FILTER:-}" ]; then
    TEMPLATE_FILTER_ARG=(--template-filter "${CFD_TEMPLATE_FILTER}")
    echo "CFD_TEMPLATE_FILTER=${CFD_TEMPLATE_FILTER}"
fi

cd ${RL_PKG}

${VENV_PY} -u ${CFD_PKG}/train_with_cfd_library.py \
    --library-dir ${LIB_DIR} \
    --rl-package-path ${RL_PKG} \
    --mix-synthetic ${MIX} \
    "${TEMPLATE_FILTER_ARG[@]}" \
    -- "$@"
