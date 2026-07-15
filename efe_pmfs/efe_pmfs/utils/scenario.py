"""
Scenario loader for efe_pmfs.

Reads a test_env scenario directory and extracts the inputs efe_pmfs needs:
  - map bounding box + resolution  (from OccupancyGrid3D.csv header)
  - mean wind vector (u_mean, v_mean)  (from wind_simulations/ ... CSV)
  - true source position  (from simulations/<sim>/sim.yaml → source.position)

Scenario string format:
  "<scenario>"                          → config1, sim1, first wind file
  "<scenario>/<sim>"                    → e.g. "env_b/sim3"
  "<scenario>/<sim>@<wind_tag>"         → e.g. "10x6_central_obstacle/sim1@05ms"
                                          picks wind_simulations/<wind_tag>/...

Handles two known test_env wind layouts:
  A) flat:    wind_simulations/wind_0.csv
              (columns: Points:0, Points:1, Points:2, U:0, U:1, U:2)
  B) nested:  wind_simulations/<speed>/wind_at_cell_centers_0.csv
              (columns: U:0, U:1, U:2, Points:0, Points:1, Points:2)

Columns are resolved by CSV header name, not fixed index.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class ScenarioInfo:
    scenario: str
    sim: str
    wind_tag: str
    config_dir: Path
    wind_csv: Path
    # map
    origin_x: float
    origin_y: float
    width_m: float
    height_m: float
    cell_size_m: float
    # wind
    wind_u_mean: float
    wind_v_mean: float
    wind_speed_mean: float
    # source (may be None if sim.yaml missing)
    source_x: Optional[float]
    source_y: Optional[float]


def _read_3d_grid_header(csv_path: Path):
    """Parse the 4-line header of OccupancyGrid3D.csv. Example:
        #env_min(m) -0.5 -0.5 0
        #env_max(m) 24.5 16.5 3
        #num_cells 250 170 30
        #cell_size(m) 0.1
    """
    with open(csv_path, 'r') as f:
        lines = [f.readline() for _ in range(4)]
    vals = {}
    for line in lines:
        parts = line.strip().lstrip('#').split()
        key = parts[0]
        nums = [float(p) for p in parts[1:]]
        vals[key] = nums
    env_min = vals['env_min(m)']
    env_max = vals['env_max(m)']
    cell_size = vals['cell_size(m)'][0]
    return env_min, env_max, cell_size


def _resolve_wind_csv(wind_root: Path, wind_tag: Optional[str]):
    """
    Locate a single wind CSV under `wind_simulations/`. Returns (csv_path, tag).

    Search strategy:
      1. If `wind_tag` given → look under wind_root/<wind_tag>/.
      2. Otherwise: prefer flat files (wind_root/wind_*.csv), then any single
         nested speed subdir; error out if there are multiple nested dirs and
         no tag was given (user must choose).
    """
    if not wind_root.is_dir():
        raise FileNotFoundError(f'No wind_simulations dir: {wind_root}')

    if wind_tag:
        sub = wind_root / wind_tag
        if not sub.is_dir():
            raise FileNotFoundError(f'Wind tag "{wind_tag}" not found under {wind_root}')
        candidates = sorted(list(sub.glob('wind_*.csv'))
                            + list(sub.glob('wind_at_cell_centers_*.csv')))
        if not candidates:
            raise FileNotFoundError(f'No wind CSV under {sub}')
        return candidates[0], wind_tag

    # No tag: prefer flat layout
    flat = sorted(list(wind_root.glob('wind_*.csv')))
    if flat:
        return flat[0], ''

    # Nested layout: require unambiguity
    subdirs = [p for p in sorted(wind_root.iterdir()) if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f'No wind CSV or subdirs under {wind_root}')
    if len(subdirs) > 1:
        options = ', '.join(p.name for p in subdirs)
        raise ValueError(
            f'Multiple wind subdirs in {wind_root}: [{options}]. '
            f'Specify one via scenario string "<scenario>/<sim>@<wind_tag>".'
        )
    sub = subdirs[0]
    candidates = sorted(list(sub.glob('wind_*.csv'))
                        + list(sub.glob('wind_at_cell_centers_*.csv')))
    if not candidates:
        raise FileNotFoundError(f'No wind CSV under {sub}')
    return candidates[0], sub.name


def _mean_wind_from_csv(csv_path: Path, z_low: float = 0.2,
                       z_high: float = 1.0) -> Tuple[float, float, float]:
    """
    Load a ParaView-style wind CSV, resolve U/V/Z columns by header name,
    and compute mean (u, v) inside the horizontal slab [z_low, z_high]. Falls
    back to the full volume if that slab is empty.
    """
    # Parse header to find column indices
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
    cols = [c.strip().strip('"') for c in header.split(',')]
    try:
        u_idx = cols.index('U:0')
        v_idx = cols.index('U:1')
        z_idx = cols.index('Points:2')
    except ValueError as e:
        raise ValueError(f'Unexpected CSV header in {csv_path}: {header}') from e

    data = np.genfromtxt(str(csv_path), delimiter=',', skip_header=1)
    if data.ndim != 2 or data.shape[1] < max(u_idx, v_idx, z_idx) + 1:
        raise ValueError(f'Unexpected wind CSV shape in {csv_path}: {data.shape}')

    z = data[:, z_idx]
    mask = (z >= z_low) & (z <= z_high)
    if not np.any(mask):
        mask = np.ones_like(z, dtype=bool)
    u = float(data[mask, u_idx].mean())
    v = float(data[mask, v_idx].mean())
    speed = float(np.hypot(u, v))
    return u, v, speed


def _source_from_sim_yaml(sim_yaml: Path):
    """Minimal inline yaml reader — avoid PyYAML dep. Extract source.position [x,y,z]."""
    if not sim_yaml.exists():
        return None, None
    text = sim_yaml.read_text()
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('position:') and '[' in s:
            try:
                inside = s.split('[', 1)[1].rsplit(']', 1)[0]
                parts = [p.strip() for p in inside.split(',')]
                return float(parts[0]), float(parts[1])
            except Exception:
                return None, None
    return None, None


def _parse_scenario_string(s: str):
    """Split "<scenario>[/<sim>][@<wind_tag>]" → (scenario, sim, wind_tag)."""
    wind_tag = ''
    if '@' in s:
        s, wind_tag = s.rsplit('@', 1)
    if '/' in s:
        scen, sim = s.split('/', 1)
    else:
        scen, sim = s, 'sim1'
    return scen, sim, wind_tag


def load_scenario(scenario_string: str,
                  test_env_share: Optional[str] = None) -> ScenarioInfo:
    """
    Parameters
    ----------
    scenario_string : str
        "<scenario>", "<scenario>/<sim>", or "<scenario>/<sim>@<wind_tag>".
    test_env_share : Optional[str]
        Override path to the test_env share dir. If None, resolved via
        ament_index_python.
    """
    scen, sim, wind_tag = _parse_scenario_string(scenario_string)

    if test_env_share is None:
        from ament_index_python.packages import get_package_share_directory
        test_env_share = get_package_share_directory('test_env')
    base = Path(test_env_share) / 'scenarios' / scen
    if not base.is_dir():
        raise FileNotFoundError(f'No scenario directory: {base}')

    config_dir = base / 'environment_configurations' / 'config1'
    if not config_dir.is_dir():
        raise FileNotFoundError(f'No config1 under: {base}')

    grid_csv = config_dir / 'OccupancyGrid3D.csv'
    env_min, env_max, cell_size = _read_3d_grid_header(grid_csv)
    origin_x, origin_y = env_min[0], env_min[1]
    width_m = env_max[0] - env_min[0]
    height_m = env_max[1] - env_min[1]

    wind_csv, resolved_tag = _resolve_wind_csv(base / 'wind_simulations', wind_tag or None)
    u_mean, v_mean, speed_mean = _mean_wind_from_csv(wind_csv)

    sim_yaml = config_dir / 'simulations' / sim / 'sim.yaml'
    sx, sy = _source_from_sim_yaml(sim_yaml)

    return ScenarioInfo(
        scenario=scen, sim=sim, wind_tag=resolved_tag,
        config_dir=config_dir, wind_csv=wind_csv,
        origin_x=origin_x, origin_y=origin_y,
        width_m=width_m, height_m=height_m, cell_size_m=cell_size,
        wind_u_mean=u_mean, wind_v_mean=v_mean, wind_speed_mean=speed_mean,
        source_x=sx, source_y=sy,
    )
