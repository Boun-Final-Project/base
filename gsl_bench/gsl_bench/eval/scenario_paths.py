"""Where every scenario datum lives. Pure functions, no ROS runtime needed.

All paths point into the INSTALL tree (install/benchmark_env/share/benchmark_env),
because that is what the launch files read and what the patches must edit.
"""
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


def install_share_dir() -> Path:
    prefix = os.environ.get('GSLB_BENCHMARK_ENV_PREFIX')
    if not prefix:
        prefix = subprocess.run(
            ['ros2', 'pkg', 'prefix', 'benchmark_env'],
            capture_output=True, text=True, check=True).stdout.strip()
    d = Path(prefix) / 'share' / 'benchmark_env'
    if not d.is_dir():
        raise RuntimeError(
            f'benchmark_env share dir not found at {d} — '
            'run: colcon build --packages-select benchmark_env')
    return d


def scenarios_dir() -> Path:
    return install_share_dir() / 'scenarios'


def scenario_dir(name: str) -> Path:
    return scenarios_dir() / name


def config_dir(name: str) -> Path:
    return scenario_dir(name) / 'environment_configurations' / 'config1'


def list_scenarios() -> List[str]:
    return sorted(p.name for p in scenarios_dir().iterdir() if p.is_dir())


def _load_yaml(path: Path) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def resolve_sim(name: str) -> str:
    """Which simulation folder scene1 plays back."""
    d = _load_yaml(config_dir(name) / 'scenes' / 'scene1.yaml')
    try:
        return str(d['simulations'][0]['sim'])
    except (KeyError, IndexError, TypeError):
        return 'sim1'


def source_xy(name: str, sim: str) -> Tuple[float, float]:
    d = _load_yaml(config_dir(name) / 'simulations' / sim / 'sim.yaml')
    pos = d['source']['position']
    return float(pos[0]), float(pos[1])


def start_xy(name: str) -> Tuple[float, float]:
    d = _load_yaml(config_dir(name) / 'config.yaml')
    pt = d['empty_point']
    return float(pt[0]), float(pt[1])


def start_yaw(name: str) -> float:
    """Robot start heading (rad) from BasicSimScene robots[0].angle. 0 for all
    current scenarios; used to align GADEN's world frame to the SLAM map frame."""
    d = _load_yaml(basic_sim_scene(name))
    try:
        return float(d['robots'][0].get('angle', 0.0) or 0.0)
    except (KeyError, IndexError, TypeError):
        return 0.0


def playback_initial_iteration(name: str) -> Optional[int]:
    d = _load_yaml(config_dir(name) / 'scenes' / 'scene1.yaml')
    v = d.get('playback_initial_iteration')
    return None if v is None else int(v)


def wind_csv(name: str) -> Optional[Path]:
    """Preferred: wind_simulations/1ms/. Fallback: the first one found."""
    preferred = scenario_dir(name) / 'wind_simulations' / '1ms' / 'wind_at_cell_centers_0.csv'
    if preferred.is_file():
        return preferred
    root = scenario_dir(name) / 'wind_simulations'
    if not root.is_dir():
        return None
    found = sorted(root.glob('*/wind_at_cell_centers_0.csv'))
    return found[0] if found else None


def result_dir(name: str, sim: str) -> Path:
    return config_dir(name) / 'simulations' / sim / 'result'


def basic_sim_scene(name: str) -> Path:
    return config_dir(name) / 'BasicSimScene.yaml'


def occupancy_yaml(name: str) -> Path:
    """The scenario's static ground-truth occupancy map (world frame)."""
    return config_dir(name) / 'occupancy.yaml'


def scene1_yaml(name: str) -> Path:
    return config_dir(name) / 'scenes' / 'scene1.yaml'
