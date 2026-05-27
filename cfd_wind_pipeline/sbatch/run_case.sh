#!/bin/bash
# Run blockMesh + snappyHexMesh + simpleFoam on a single OpenFOAM case.
# Usage:  sbatch run_case.sh <abs-path-to-case-dir>
# Env vars:
#   CFD_OPENFOAM_SQSH  path to Pyxis squashfs container (OpenFOAM)
#   CFD_DATA_ROOT      where to write slurm-*.out (defaults inline below)
#SBATCH --job-name=cfd_case
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail
CASE_DIR=${1:?"usage: sbatch run_case.sh <abs-path-to-case-dir>"}
SQSH=${CFD_OPENFOAM_SQSH:-/comp04-storage/efe-mantaroglu/osl/opencfd+openfoam-default+latest.sqsh}

srun \
  --container-image=${SQSH} \
  --container-mounts=/comp04-storage:/comp04-storage \
  bash -c "
    set -euo pipefail
    cd ${CASE_DIR}
    echo '===blockMesh==='
    blockMesh 2>&1 | tail -10
    echo '===snappyHexMesh==='
    snappyHexMesh -overwrite 2>&1 | tail -20
    echo '===restore 0 from 0.orig==='
    rm -rf 0
    cp -r 0.orig 0
    echo '===simpleFoam==='
    simpleFoam 2>&1 | tail -30
    echo '===times produced==='
    ls -d [0-9]* 2>/dev/null
    LAST=\$(ls -d [0-9]* | sort -n | tail -1)
    echo last time: \$LAST
    echo '===cell count==='
    checkMesh -latestTime 2>&1 | grep -E 'cells:|points:' | head -5
"
