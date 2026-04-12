"""
Hyperparameters and constants for RL Gas Source Localization.
Single source of truth — all other modules import from here.
"""

# =============================================================================
# Environment
# =============================================================================
MAX_STEPS = 600
STEP_SIZE = 0.5                 # meters per action
GRID_RESOLUTION = 0.1           # meters (occupancy grid cell size)
VISITED_CELL_RESOLUTION = 0.5   # meters (for new-cell reward tracking)
D_SUCCESS = 0.5                 # meters, source-found threshold
ROBOT_RADIUS = 0.25             # meters, collision checking radius (matches GADEN planners)
MIN_SOURCE_ROBOT_DIST = 3.0     # meters, minimum initial separation

# LiDAR
LIDAR_NUM_RAYS = 72
LIDAR_MAX_RANGE = 3.0           # meters

# Gas sensor
GAS_HISTORY_LENGTH = 10
GAS_FEATURES_PER_STEP = 3           # (rel_x, rel_y, binary) per timestep

# State dimensions (derived)
STATE_DIM = GAS_HISTORY_LENGTH * GAS_FEATURES_PER_STEP + LIDAR_NUM_RAYS + 2 + 2 + 1  # 107

# =============================================================================
# Rewards
# =============================================================================
R_SUCCESS = 200.0
R_DETECTION = 2.0
R_NEW_CELL = 0.2
R_STEP = -0.1
R_COLLISION = -2.0
R_MAX_STEPS = -20.0

# =============================================================================
# Wind
# =============================================================================
WIND_SPEED_RANGE = (0.1, 1.5)   # m/s, per-episode uniform sample
WIND_MAX_SPEED = 2.0            # for normalization
WIND_DISPERSION_FACTOR = 2.0    # how much wind shifts concentration peak

# =============================================================================
# Gas dispersion model selection
# =============================================================================
GAS_MODEL = "filament"          # "filament" or "igdm"

# =============================================================================
# Filament-based gas dispersion (Lagrangian model)
# =============================================================================
FILAMENTS_PER_STEP = 2          # new filaments released per env.step()
FILAMENT_DT = 0.5               # seconds per filament timestep
FILAMENT_K = 0.02               # atmospheric diffusivity (m^2/s)
FILAMENT_TURBULENCE_SCALE = 0.2 # turbulence as fraction of wind speed
FILAMENT_MAX_AGE = 120          # steps before filament is culled (~60 s)
FILAMENT_INITIAL_SIGMA = 0.05  # meters, initial filament size
FILAMENT_MIN_SIGMA = 0.01       # meters, clamp to prevent div-by-zero
FILAMENT_MASS = 1.0             # arbitrary, per-filament mass
FILAMENT_REFLECTION_ENERGY = 0.8 # velocity retention factor on wall bounce
FILAMENT_WARMUP_STEPS = 15      # plume steps before first obs (fresh-dispersion scenario)
FILAMENT_WALL_OCCLUSION = True  # zero contributions from filaments behind walls

# =============================================================================
# IGDM-based gas dispersion (Gaussian + Dijkstra, legacy)
# =============================================================================
SIGMA_M_BASE = 1.5              # dispersion parameter (matches ali_igdm)
DISPERSION_RATE = 0.12          # time-dependent dispersion (matches igdm_improved)
COARSE_RESOLUTION = 0.5         # coarse grid for Dijkstra
SENSOR_ALPHA = 0.1              # proportional noise coefficient
SENSOR_SIGMA_ENV = 0.1          # environmental noise std
SENSOR_THRESHOLD_WEIGHT = 0.5   # adaptive threshold weight
SOURCE_RELEASE_RATE = 1.0       # Q for gas source

# =============================================================================
# Map generation
# =============================================================================
WALL_THICKNESS = 0.2            # meters
MIN_GAP_SIZE = 1.2              # meters, minimum doorway/gap width
ROOM_WIDTH_RANGE = (8.0, 20.0)  # meters (full range)
ROOM_HEIGHT_RANGE = (6.0, 15.0) # meters (full range)

# Curriculum: room size grows linearly from small to full range
CURRICULUM_WIDTH_START = (8.0, 10.0)    # small rooms at start
CURRICULUM_HEIGHT_START = (6.0, 8.0)
CURRICULUM_FRACTION = 0.5               # fraction of training to reach full size

# Curriculum: template unlock schedule (progress → max template index)
# Templates: 0=empty, 1=single_wall, 2=u_shape, 3=three_walls, 4=complex_maze, 5=multi_room
TEMPLATE_CURRICULUM_STAGES = [
    (0.00, 1),   # 0-25%:  templates 0-1 (open rooms, learn gas-following)
    (0.25, 3),   # 25-50%: templates 0-3 (obstacles, learn wall navigation)
    (0.50, 5),   # 50%+:   templates 0-5 (full set, corridors and multi-room)
]

# =============================================================================
# PPO
# =============================================================================
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEFF = 0.02
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
NUM_ENVS = 8
ROLLOUT_LENGTH = 2048
NUM_MINIBATCHES = 32
UPDATE_EPOCHS = 10
TOTAL_TIMESTEPS = 100_000_000

# =============================================================================
# Network (shared by both architectures)
# =============================================================================
HIDDEN_DIM = 256
BACKBONE_LAYERS = 2
ACTOR_HEAD_DIM = 128
CRITIC_HEAD_DIM = 128

# Modular architecture
GAS_GRU_HIDDEN = 64
LIDAR_CONV_CHANNELS = 2
LIDAR_CONV_KERNEL = 5
