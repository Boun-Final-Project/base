"""Centralized configuration for the CFD wind pipeline.

Override any of these via environment variables of the same name.
"""
import os

# ---------------------------------------------------------------------------
# Paths — environment-specific. Override via env var if needed.
# ---------------------------------------------------------------------------

# Path to a Python package providing `reinforcement_learning.envs.map_generator`
# (used to generate procedural maps) and `.test.gaden_loader` (used to load
# GADEN maps + wind fields for reference visualizations and validation).
RL_PACKAGE_PATH = os.environ.get(
    'CFD_RL_PACKAGE_PATH',
    '/comp04-storage/efe-mantaroglu/osl/friend_base_loop_spatial',
)

# Path to the Pyxis-importable OpenFOAM squashfs container.
# Build with:  enroot import 'docker://opencfd/openfoam-default:latest'
OPENFOAM_SQSH = os.environ.get(
    'CFD_OPENFOAM_SQSH',
    '/comp04-storage/efe-mantaroglu/osl/opencfd+openfoam-default+latest.sqsh',
)

# Path to the GADEN maps directory (with recommended_configs.yaml at the root).
GADEN_MAPS_ROOT = os.environ.get(
    'CFD_GADEN_MAPS_ROOT',
    '/comp04-storage/efe-mantaroglu/osl/friend_base_loop_spatial/gaden_maps',
)

# Working directory for case dirs / wind library / artifacts.
# This is DATA, not code — should NOT be inside the git repo.
DATA_ROOT = os.environ.get(
    'CFD_DATA_ROOT',
    '/comp04-storage/efe-mantaroglu/osl/cfd_test',
)

# Python interpreter to use inside sbatch jobs.
PYTHON_BIN = os.environ.get(
    'CFD_PYTHON_BIN',
    '/home/efe-mantaroglu/simenv/bin/python',
)

# ---------------------------------------------------------------------------
# Defaults for the pipeline
# ---------------------------------------------------------------------------

# Wall + bg-mesh defaults
WALL_HEIGHT_M = 3.0
WALL_THICK_M = 0.6        # also the wall_thick_cells * cell_size
BG_PAD_M = 0.5            # padding between map edge and bg-mesh boundary
BG_CELLS_PER_M = 4.0

# Inlet defaults
DEFAULT_INLET_SPEED = 0.5   # m/s

# simpleFoam controls
SIMPLE_END_TIME = 200       # iterations to run (steady-state)

# Placement sampler defaults
PLACE_N_INLETS_PROBS = {1: 0.2, 2: 0.6, 3: 0.2}
PLACE_N_OUTLETS_PROBS = {1: 0.2, 2: 0.6, 3: 0.2}
PLACE_INLET_SIDE_PROBS = {'west': 0.7, 'south': 0.3}
PLACE_OPENING_WIDTH_RANGE = (1.0, 2.5)   # m
