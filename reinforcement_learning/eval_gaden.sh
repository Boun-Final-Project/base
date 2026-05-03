#!/bin/bash
#SBATCH --job-name=eval_gaden
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:0:0
#SBATCH --partition=batch
#SBATCH --qos=users
#SBATCH --account=users
#SBATCH --output=/comp04-storage/efe-mantaroglu/osl/base/reinforcement_learning/runs/slurm-%j.out

set -euo pipefail
cd /comp04-storage/efe-mantaroglu/osl

VENV_PY=/home/efe-mantaroglu/simenv/bin/python

RUN_DIR="${1:-base/reinforcement_learning/runs/<your-run-dir>}"
shift || true

${VENV_PY} -u -m base.reinforcement_learning.test.eval_checkpoints_gaden \
    --run-dirs "${RUN_DIR}" "$@"
