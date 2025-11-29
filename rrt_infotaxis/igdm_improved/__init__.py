"""
RRT-Infotaxis with IGDM (Indoor Gaussian Dispersion Model) Package.

This package contains the complete implementation of RRT-Infotaxis with IGDM
for gas source localization using adaptive sampling and information-theoretic planning.

Main Classes:
- RRTInfotaxisIGDM: Main algorithm orchestrator
- RRTInfotaxis: RRT planner with information gain evaluation
- ParticleFilter: Particle filter for source estimation
- IGDMModel: Indoor Gaussian Dispersion Model
- BinarySensorModel: Binary sensor model for measurements
- OccupancyGrid: Grid-based environment representation
- StepVisualizer: Visualization utility
"""

from .igdm_model import IGDMModel
from .sensor_model import BinarySensorModel
from .particle_filter import ParticleFilter
from .occupancy_grid import OccupancyGrid
from .rrt import RRTInfotaxis, Node
from .visualizer import StepVisualizer
from .rrt_infotaxis_igdm import RRTInfotaxisIGDM

__version__ = "1.0.0"
__author__ = "Ali Sonmez"

__all__ = [
    "RRTInfotaxisIGDM",
    "RRTInfotaxis",
    "ParticleFilter",
    "IGDMModel",
    "BinarySensorModel",
    "OccupancyGrid",
    "StepVisualizer",
    "Node",
]
