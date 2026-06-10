#!/bin/bash
#
# finetune_local_wind.sh — finetune a trained checkpoint on a CFD wind library
# with LOCAL point-wind observation (the "local2" recipe).
#
# This is the recipe behind our best finetunes: resume the champion
# (agent_91750400.pt, mean-wind observation) and finetune ~15M steps on the
# CFD library (~2,400 pre-computed wind fields) with
#   * OSL_LOCAL_WIND_OBS=1  — observe the local wind at the robot's cell
#     instead of the episode's spatial mean, and
#   * 20% far-plume spawns — robot re-placed 4-14 m (plume-hit distance) from
#     the plume so it must SEARCH before it can track.
# On the offline real-gas eval harness this took the champion from 57% to
# 76-80% overall success (20 eps/map), including the first non-zero results
# on many_rooms.
#
# Requirements:
#   * A CFD wind library built with cfd_wind_pipeline (see its README).
#   * An RL package checkout — this repository's root works: its
#     reinforcement_learning package supports OSL_LOCAL_WIND_OBS and the
#     inline real-gas GADEN eval (OSL_INLINE_GADEN_*, exported with defaults
#     by train_cfd_library.sh; needs GADEN_SCENARIOS_ROOT replay data, writes
#     runs/<name>/gaden_curve.csv with a step-0 baseline row when resuming).
#   * Optional speed-up: precompute the per-case far-placement cache once with
#     sbatch cfd_wind_pipeline/sbatch/precompute_far_placement.sh (otherwise
#     it is computed lazily at first reset).
#
# Usage:
#   bash cfd_wind_pipeline/sbatch/finetune_local_wind.sh \
#        <library-dir>[,<library-dir2>,...] \
#        <rl-package-path> \
#        <resume-checkpoint.pt> \
#        [total-timesteps] [run-name]
#
# NOTE --total-timesteps is ABSOLUTE (resumed checkpoint step + extra), not
# additional. The champion base is 91.75M, so the default 152M leaves ~60M of
# headroom. In practice GADEN transfer peaks ~15M steps into the finetune and
# drifts back down afterwards while train success stays healthy — evaluate
# checkpoints with reinforcement_learning/eval_gaden.sh and pick the best by
# GADEN success, NOT by training reward.
#
# DEPLOY/EVAL: the resulting policy observes LOCAL wind. Evaluating or
# deploying it without OSL_LOCAL_WIND_OBS=1 is a train/deploy mismatch and
# will underperform.
set -euo pipefail

LIBS=${1:?"usage: finetune_local_wind.sh <library-dir(s)> <rl-package-path> <resume-ckpt> [total-timesteps] [run-name]"}
RL_PKG=${2:?"need rl-package-path (config.py must support OSL_LOCAL_WIND_OBS)"}
CKPT=${3:?"need checkpoint to resume from (e.g. .../checkpoints/agent_91750400.pt)"}
TOTAL=${4:-152000000}
RUN_NAME=${5:-local_wind_ft}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SH="${SCRIPT_DIR}/train_cfd_library.sh"

export OSL_LOCAL_WIND_OBS=1   # the lever: observe wind at the robot cell
export CFD_MIX_SYNTHETIC=0    # pure CFD library, no synthetic-wind resets
export CFD_FAR_PLUME_FRAC=${CFD_FAR_PLUME_FRAC:-0.2}   # 20% search-style spawns
export CFD_FAR_PLUME_RANGE=${CFD_FAR_PLUME_RANGE:-4,14} # plume-hit distance band (m)
# CFD_TEMPLATE_FILTER (e.g. "0,1,2,3,4,5") passes through to the launcher if
# set in the submitting shell; default is every template in the library.
# Inline real-gas GADEN eval defaults are exported by train_cfd_library.sh.

JID=$(sbatch --parsable --mem=24G --job-name=ppo_local_wind_ft \
  "${TRAIN_SH}" "${LIBS}" "${RL_PKG}" \
  --resume "${CKPT}" \
  --arch dual --num-envs 96 \
  --lr 5e-5 --min-lr 1e-5 --anneal-lr --anneal-start 0.5 \
  --clip-epsilon 0.3 --target-kl 0.05 \
  --save-interval 50 \
  --total-timesteps "${TOTAL}" \
  --output-dir "reinforcement_learning/runs/${RUN_NAME}")

echo "local-wind finetune -> job ${JID}"
echo "  libraries: ${LIBS}"
echo "  resume:    ${CKPT}"
echo "  output:    ${RL_PKG}/reinforcement_learning/runs/${RUN_NAME}"
