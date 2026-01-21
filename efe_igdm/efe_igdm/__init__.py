"""
RRT-Infotaxis with IGDM for Gas Source Localization.

This package implements information-theoretic gas source localization
using RRT-based path planning with Indoor Gaussian Dispersion Model.
"""

# Main ROS2 node
from .igdm import RRTInfotaxisNode

# Planning algorithms
from .planning.rrt import RRT
from .planning.global_planner import GlobalPlanner
from .planning.dead_end_detector import DeadEndDetector

# Estimation
from .estimation.particle_filter import ParticleFilter
from .estimation.sensor_model import ContinuousGaussianSensorModel

# Gas dispersion models
from .models.igdm_gas_model import IndoorGaussianDispersionModel

# Mapping
from .mapping.occupancy_grid import (
    OccupancyGridMap,
    create_occupancy_map_from_service,
    create_empty_occupancy_map
)

# Visualization
from .visualization.text_visualizer import TextVisualizer

__all__ = [
    # Main node
    'RRTInfotaxisNode',

    # Planning
    'RRT',
    'GlobalPlanner',
    'DeadEndDetector',

    # Estimation
    'ParticleFilter',
    'ContinuousGaussianSensorModel',

    # Models
    'IndoorGaussianDispersionModel',

    # Mapping
    'OccupancyGridMap',
    'create_occupancy_map_from_service',
    'create_empty_occupancy_map',

    # Visualization
    'TextVisualizer',
]
