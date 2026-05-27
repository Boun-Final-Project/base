#!/bin/bash
# Run blockMesh + snappyHexMesh + simpleFoam + postProcess + extract on ONE case
# from a library manifest, selected by SLURM_ARRAY_TASK_ID.
#
# Usage:
#   sbatch --array=0-199%24 run_array.sh <library-dir>
#
# Env vars:
#   CFD_OPENFOAM_SQSH  path to Pyxis squashfs container (OpenFOAM)
#   CFD_PYTHON_BIN     python interpreter with numpy + scipy (for extract step)
#SBATCH --job-name=cfd_arr
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=cfd_arr_%A_%a.out

set -euo pipefail

LIB_DIR=${1:?"usage: sbatch run_array.sh <library-dir>"}
SQSH=${CFD_OPENFOAM_SQSH:-/comp04-storage/efe-mantaroglu/osl/opencfd+openfoam-default+latest.sqsh}
PYBIN=${CFD_PYTHON_BIN:-/home/efe-mantaroglu/simenv/bin/python}
# Path to the cfd_wind_pipeline/ dir. Override with CFD_PIPELINE_DIR if you move
# the source tree. Cannot use BASH_SOURCE — SLURM stages this script to a
# scratch location, so ${BASH_SOURCE[0]} resolves to /var/spool/slurm/...
SCRIPT_DIR=${CFD_PIPELINE_DIR:-/comp04-storage/efe-mantaroglu/osl/base/cfd_wind_pipeline}

IDX=${SLURM_ARRAY_TASK_ID:?"must be run as a SLURM array job"}

# Pull the case_dir for this index out of the manifest using a tiny python
# inline call (avoids extra dep).
CASE_DIR=$(${PYBIN} - "$LIB_DIR" "$IDX" <<'PY'
import json, sys
lib, idx = sys.argv[1], int(sys.argv[2])
m = json.load(open(f"{lib}/manifest.json"))
print(m[idx]['case_dir'])
PY
)
echo "Array task $IDX → $CASE_DIR"

# Skip if already done (wind_field.npz present)
if [ -f "${CASE_DIR}/wind_field.npz" ]; then
    echo "Already complete: ${CASE_DIR}/wind_field.npz exists. Skipping."
    exit 0
fi

# Step 1: write the sample dict (needs to exist before postProcess is called).
${PYBIN} ${SCRIPT_DIR}/extract_wind.py --case-dir ${CASE_DIR} --write-dict-only

# Step 2: run mesh + solver + postProcess inside the container.
srun \
  --container-image=${SQSH} \
  --container-mounts=/comp04-storage:/comp04-storage \
  bash -c "
    set -euo pipefail
    cd ${CASE_DIR}
    echo '===blockMesh==='
    blockMesh 2>&1 | tail -5
    echo '===snappyHexMesh==='
    snappyHexMesh -overwrite 2>&1 | tail -5
    echo '===restore 0==='
    rm -rf 0
    cp -r 0.orig 0
    echo '===simpleFoam==='
    simpleFoam 2>&1 | tail -10
    echo '===postProcess sample==='
    postProcess -func sample -latestTime 2>&1 | tail -5
"

# Step 3: parse the raw output into wind_field.npz (outside container, needs scipy).
${PYBIN} ${SCRIPT_DIR}/extract_wind.py --case-dir ${CASE_DIR} --parse-only

# Step 4: prune CFD mesh artifacts unless CFD_KEEP_MESH=1. 132MB → ~400KB/case.
# Keeps wind_field.npz + grid.npz + meta.json (everything training needs).
if [ "${CFD_KEEP_MESH:-0}" != "1" ] && [ -f "${CASE_DIR}/wind_field.npz" ]; then
    rm -rf ${CASE_DIR}/constant ${CASE_DIR}/system ${CASE_DIR}/0 ${CASE_DIR}/0.orig
    rm -rf ${CASE_DIR}/postProcessing
    find ${CASE_DIR} -maxdepth 1 -type d -regex '.*/[0-9]+\(\.[0-9]+\)?$' -exec rm -rf {} +
fi

echo "DONE: $CASE_DIR"
