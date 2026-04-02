"""
Hyperparameters and constants for RL Gas Source Localization.
Single source of truth — all other modules import from here.
"""

# =============================================================================
# Environment
# =============================================================================
MAX_STEPS = 300
STEP_SIZE = 0.5                 # meters per action
GRID_RESOLUTION = 0.1           # meters (occupancy grid cell size)
VISITED_CELL_RESOLUTION = 0.5   # meters (for new-cell reward tracking)
D_SUCCESS = 0.5                 # meters, source-found threshold
ROBOT_RADIUS = 0.2              # meters, collision checking radius
MIN_SOURCE_ROBOT_DIST = 3.0     # meters, minimum initial separation

# LiDAR
LIDAR_NUM_RAYS = 24
LIDAR_MAX_RANGE = 3.0           # meters

# Gas sensor
GAS_HISTORY_LENGTH = 10

# State dimensions (derived)
STATE_DIM = GAS_HISTORY_LENGTH + LIDAR_NUM_RAYS + 2 + 2 + 1  # 39

# =============================================================================
# Rewards
# =============================================================================
R_SUCCESS = 100.0
R_DETECTION = 1.0
R_NEW_CELL = 0.2
R_STEP = -0.1
R_COLLISION = -2.0
R_MAX_STEPS = -10.0

# =============================================================================
# Wind
# =============================================================================
WIND_SPEED_RANGE = (0.1, 1.5)   # m/s, per-episode uniform sample
WIND_MAX_SPEED = 2.0            # for normalization
WIND_DISPERSION_FACTOR = 2.0    # how much wind shifts concentration peak

# =============================================================================
# Gas dispersion (IGDM)
# =============================================================================
SIGMA_M_BASE = 1.0              # base dispersion parameter
DISPERSION_RATE = 3.0           # temporal growth rate (alpha)
COARSE_RESOLUTION = 0.5         # coarse grid for Dijkstra
SENSOR_ALPHA = 0.1              # proportional noise coefficient
SENSOR_SIGMA_ENV = 0.1          # environmental noise std
SENSOR_THRESHOLD_WEIGHT = 0.5   # adaptive threshold weight
SOURCE_RELEASE_RATE = 1.0       # Q for gas source

# =============================================================================
# Map generation
# =============================================================================
WALL_THICKNESS = 0.2            # meters
MIN_GAP_SIZE = 0.8              # meters, minimum doorway/gap width
ROOM_WIDTH_RANGE = (8.0, 20.0)  # meters
ROOM_HEIGHT_RANGE = (6.0, 15.0) # meters

# =============================================================================
# PPO
# =============================================================================
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
NUM_ENVS = 8
ROLLOUT_LENGTH = 2048
NUM_MINIBATCHES = 32
UPDATE_EPOCHS = 10
TOTAL_TIMESTEPS = 10_000_000

# =============================================================================
# Network
# =============================================================================
HIDDEN_DIM = 256
BACKBONE_LAYERS = 2
ACTOR_HEAD_DIM = 128
CRITIC_HEAD_DIM = 128
