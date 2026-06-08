#!/bin/bash
# Smoke test: run the unmodified windAroundBuildings tutorial in the container.
# Use to verify the OpenFOAM Pyxis container + cluster setup is working before
# attempting any of our own cases.
# Requires: copy of windAroundBuildings tutorial at ${1}/test_unmodified
# Env vars:
#   CFD_OPENFOAM_SQSH  path to Pyxis squashfs container (OpenFOAM)
#SBATCH --job-name=cfd_smoke
#SBATCH --partition=batch
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail
CASE_DIR=${1:?"usage: sbatch run_smoke.sh <abs-path-to-test_unmodified-dir>"}
SQSH=${CFD_OPENFOAM_SQSH:-/comp04-storage/efe-mantaroglu/osl/opencfd+openfoam-default+latest.sqsh}

srun \
  --container-image=${SQSH} \
  --container-mounts=/comp04-storage:/comp04-storage \
  bash -c "
    set -euo pipefail
    cd ${CASE_DIR}
    echo '===Allrun.pre==='
    ./Allrun.pre
    echo '===restore0Dir==='
    cp -r 0.orig 0
    echo '===simpleFoam==='
    simpleFoam 2>&1 | tail -30
    echo '===check times produced==='
    ls -d [0-9]* 2>/dev/null
    LAST=\$(ls -d [0-9]* | sort -n | tail -1)
    echo last time: \$LAST
    head -50 \${LAST}/U
"
