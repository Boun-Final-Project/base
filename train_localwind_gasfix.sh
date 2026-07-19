#!/bin/bash
#
# train_localwind_gasfix.sh — CORRECTED-GAS-PHYSICS arm of the A/B against job 25311.
#
# HYPOTHESIS. The surrogate's filament diffusivity is wrong by 40x. gpu_filament.advect_diffuse_batch
# grows filament variance at 2*K per second; config.py ships FILAMENT_K=0.02 m^2/s = 400 cm^2/s, while
# the real GADEN sim.yaml we deploy against specifies filamentGrowthGamma = 10 cm^2/s (=> K = 0.0005).
# The surrogate therefore smears GADEN's thin intermittent ribbons into a fat continuous cloud. Measured
# with the SAME policy (upd4000) in both domains:
#
#     GADEN (real)            detect% = 23.9   mean blank-run = 17.8 steps
#     surrogate K=0.02        detect% = 54-71  mean blank-run =  4.5      <- what 25311 trained on
#     surrogate K=0.0005      detect% = 25.5   mean blank-run = 12.2      <- this arm
#
# So 25311 trained a plume-TRACKER on a plume that does not exist. This arm fixes the constants,
# straight off the GADEN sim.yaml -- no fitting, no calibration to test scores:
#
#     FILAMENT_K              0.02 -> 0.0005   (filamentGrowthGamma: 10 cm^2/s)
#     FILAMENT_INITIAL_SIGMA  0.05 -> 0.10     (filamentInitialSigma: 10 cm)
#     FILAMENTS_PER_STEP      2    -> 5        (numFilaments_sec: 10, at FILAMENT_DT=0.5)
#     FILAMENT_MAX_AGE        120  -> 120      (unchanged: GADEN has NO age cull -- filaments dilute)
#
# K is log-annealed 0.02 -> 0.0005 over the first 30% of updates so early exploration still gets a rich
# plume (detection drops 54% -> 25%, thinning the R_DETECTION learning signal) and ends at correct physics.
#
# CONTROL = job 25311 (already trained, same script, same everything, K=0.02). Its GADEN checkpoint curve
# is already known, so the control is free and the A/B is one-variable.
#
# PRE-REGISTERED PREDICTION (write it down BEFORE looking):
#   * many_rooms (sparsest map: 7.6% detect, 27-step blanks, stuck at 0% through every prior fix) SHOULD MOVE.
#   * ultimate (27.7% detect, 49-step blanks, currently 60%) should improve.
#   * dense easy maps (4rooms: 41% detect, already 100%) should be FLAT or slightly worse.
#   If instead the easy maps improve while many_rooms stays 0%, the hypothesis is WRONG regardless of
#   the headline average.
#
# SELECTION / COMPARISON (do NOT select on the surrogate val -- the two arms train on DIFFERENT gas
# physics, so their val sets are different distributions and the numbers are not commensurable; each arm
# looks best on its own val by construction). GADEN is the only common arbiter:
#   val  = the 7 original GADEN maps (already burned by earlier sweeps) -> pick each arm's best ckpt
#   test = the 23 new balanced GADEN scenarios (virgin)                 -> score both, ONCE
# The script's built-in localwind_best_val.pt gate selects on the surrogate val and is KNOWN BROKEN for
# GADEN selection (spearman <= +0.43). Ignore that file; select externally.
#
# Usage:
#   sbatch train_localwind_gasfix.sh            # seed 1 (matched pair with control 25311)
#   SEED=2 sbatch train_localwind_gasfix.sh     # seed 2 (guards against a seed fluke)
#
#SBATCH --job-name=ppo_lw_gasfix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=72:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
VENV_PY="${VENV_PY:-/home/efe-mantaroglu/simenv/bin/python}"

SEED="${SEED:-1}"
RUN_NAME="ppo_lw_gasfix_s${SEED}_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}"
OUT_DIR="${SCRIPT_DIR}/reinforcement_learning/runs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

# Everything below EXCEPT the four --gas-* flags is byte-for-byte the job-25311 control
# (E=1024, rollout=256, 8000 updates, cfd-cases 3000, val-cases 600, lr 3e-4, flip ON, from scratch).
# Measured throughput with the corrected gas: ~16.5k steps/s => ~35 h for 2.097e9 steps (control: 22.4k/s).
"${VENV_PY}" -u train_localwind_gpu.py \
    --wind cfd \
    --cfd-dirs cfd_test/library_v4_4dir \
    --holdout-shards shard_38,shard_39 \
    --flip \
    --envs 1024 \
    --rollout 256 \
    --updates 8000 \
    --cfd-cases 3000 \
    --val-cases 600 \
    --eval-every 100 \
    --lr 3e-4 \
    --seed "${SEED}" \
    --gas-k 0.0005 \
    --gas-k-start 0.02 \
    --gas-anneal-frac 0.3 \
    --gas-sigma 0.10 \
    --gas-fps 5 \
    --out "${OUT_DIR}" \
    "$@"
