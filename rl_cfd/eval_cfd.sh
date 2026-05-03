#!/bin/bash
#SBATCH --job-name=eval_cfd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users
#SBATCH --output=/comp04-storage/efe-mantaroglu/osl/base/rl_cfd/runs/slurm-%j.out

set -euo pipefail
cd /comp04-storage/efe-mantaroglu/osl

VENV_PY=/home/efe-mantaroglu/simenv/bin/python

RUN_DIR="${1:-base/rl_cfd/runs/ppo_cfd_20260426_203843_job4497}"
shift || true

${VENV_PY} -u base/rl_cfd/test/eval_checkpoints_spatial.py \
    --run-dirs "${RUN_DIR}" "$@"
