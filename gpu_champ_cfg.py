"""Single-source-of-truth config adapter for the faithful GPU champ port.

Loads champ_config.json (runs/lidar-007 snapshot) with fallback to
reinforcement_learning/config.py, and exposes the attribute names the GPU kernels
(gpu_filament/gpu_lidar/gpu_wind) and GpuChampEnv expect. champ_config.json OVERRIDES
config.py where they differ (e.g. CLIP_EPSILON 0.3 vs 0.2). See AUDIT.md.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import reinforcement_learning.config as C

_JSON = os.path.join(os.path.dirname(C.__file__), "champ_config.json")


def _load_json():
    try:
        with open(_JSON) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@dataclass
class ChampCfg:
    # --- geometry / sim ---
    res: float = C.GRID_RESOLUTION
    step_size: float = C.STEP_SIZE
    robot_radius: float = C.ROBOT_RADIUS
    d_success: float = C.D_SUCCESS
    max_steps: int = C.MAX_STEPS
    min_source_robot_dist: float = C.MIN_SOURCE_ROBOT_DIST

    # --- filament plume ---
    filaments_per_step: int = C.FILAMENTS_PER_STEP
    dt: float = C.FILAMENT_DT
    K: float = C.FILAMENT_K
    turb_scale: float = C.FILAMENT_TURBULENCE_SCALE
    max_age: int = C.FILAMENT_MAX_AGE
    initial_sigma: float = C.FILAMENT_INITIAL_SIGMA
    min_sigma: float = C.FILAMENT_MIN_SIGMA
    mass: float = C.FILAMENT_MASS
    reflection_energy: float = C.FILAMENT_REFLECTION_ENERGY
    warmup: int = C.FILAMENT_WARMUP_STEPS
    wall_occlusion: bool = C.FILAMENT_WALL_OCCLUSION

    # --- lidar ---
    lidar_rays: int = C.LIDAR_NUM_RAYS
    lidar_range: float = C.LIDAR_MAX_RANGE
    lidar_noise: float = C.LIDAR_NOISE_STD

    # --- sensor ---
    sensor_alpha: float = C.SENSOR_ALPHA
    sensor_sigma_env: float = C.SENSOR_SIGMA_ENV
    thr_weight: float = C.SENSOR_THRESHOLD_WEIGHT

    # --- wind ---
    wind_speed_range: Tuple[float, float] = tuple(C.WIND_SPEED_RANGE)
    max_speed: float = C.WIND_MAX_SPEED
    flip: bool = False  # champ TRAINING obs has NO anemometer flip (deploy-only)

    # --- obs / history ---
    gas_hist: int = C.GAS_HISTORY_LENGTH
    state_dim: int = C.STATE_DIM

    # --- rewards ---
    r_step: float = C.R_STEP
    r_coll: float = C.R_COLLISION
    r_det: float = C.R_DETECTION
    r_success: float = C.R_SUCCESS

    # --- map / curriculum ---
    room_width_range: Tuple[float, float] = tuple(C.ROOM_WIDTH_RANGE)
    room_height_range: Tuple[float, float] = tuple(C.ROOM_HEIGHT_RANGE)
    curric_width_start: Tuple[float, float] = tuple(C.CURRICULUM_WIDTH_START)
    curric_height_start: Tuple[float, float] = tuple(C.CURRICULUM_HEIGHT_START)
    curric_fraction: float = C.CURRICULUM_FRACTION
    template_stages: tuple = tuple(tuple(s) for s in C.TEMPLATE_CURRICULUM_STAGES)
    template_weights: tuple = tuple(C.TEMPLATE_SAMPLING_WEIGHTS)

    # padded grid extent (cells) — Hmax from max height, Wmax from max width
    Wmax: int = field(default=0)
    Hmax: int = field(default=0)

    def __post_init__(self):
        import math
        # Padded grid extent must cover the LARGEST map any template can produce, not just
        # ROOM_*_RANGE: _generate_multi_room sets height = base*aspect with base<=max(12,width_hi)
        # and aspect<=1.2, so height can reach 1.2*width_hi (~24m) — larger than height_hi (15m).
        # Scanned empirically: max 236x200 cells at full range. (+2 cell margin.)
        wmax_m = self.room_width_range[1]
        hmax_m = max(self.room_height_range[1], max(12.0, wmax_m) * 1.2)
        if self.Wmax == 0:
            self.Wmax = int(math.ceil(wmax_m / self.res)) + 2
        if self.Hmax == 0:
            self.Hmax = int(math.ceil(hmax_m / self.res)) + 2

    # F capacity per env (ring buffer): max_age*per_step*2 guarantees dead-before-reuse
    @property
    def F(self) -> int:
        return self.max_age * self.filaments_per_step * 2


def load_champ_cfg(**overrides) -> ChampCfg:
    """Build ChampCfg from config.py defaults, override with champ_config.json, then kwargs."""
    j = _load_json()
    cfg = ChampCfg()
    # json key -> cfg attr (only the ones that can differ from config.py)
    jmap = {
        "CLIP_EPSILON": None,  # PPO-side, not env; handled by train.py
        "FILAMENT_MAX_AGE": "max_age",
        "FILAMENTS_PER_STEP": "filaments_per_step",
        "FILAMENT_DT": "dt", "FILAMENT_K": "K",
        "FILAMENT_TURBULENCE_SCALE": "turb_scale",
        "FILAMENT_INITIAL_SIGMA": "initial_sigma",
        "FILAMENT_MIN_SIGMA": "min_sigma",
        "FILAMENT_REFLECTION_ENERGY": "reflection_energy",
        "FILAMENT_WARMUP_STEPS": "warmup",
        "LIDAR_NUM_RAYS": "lidar_rays", "LIDAR_MAX_RANGE": "lidar_range",
        "LIDAR_NOISE_STD": "lidar_noise",
        "SENSOR_ALPHA": "sensor_alpha", "SENSOR_SIGMA_ENV": "sensor_sigma_env",
        "SENSOR_THRESHOLD_WEIGHT": "thr_weight",
        "WIND_MAX_SPEED": "max_speed",
        "MAX_STEPS": "max_steps", "STEP_SIZE": "step_size",
        "ROBOT_RADIUS": "robot_radius", "D_SUCCESS": "d_success",
        "R_STEP": "r_step", "R_COLLISION": "r_coll",
        "R_DETECTION": "r_det", "R_SUCCESS": "r_success",
    }
    for jk, attr in jmap.items():
        if attr is not None and jk in j:
            setattr(cfg, attr, j[jk])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg
