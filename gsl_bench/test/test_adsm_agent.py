import numpy as np
import pytest

from gsl_bench.agent import MapInfo, Observation, ScenarioInfo, Waypoint

adsm_core_py = pytest.importorskip('adsm_core_py')
from gsl_bench.agents.adsm_agent import AdsmAgent  # noqa: E402


def _map():
    grid = np.full((80, 80), -1, dtype=np.int8)
    grid[20:60, 20:60] = 0
    grid[35:40, 35:40] = 100
    return MapInfo(grid, 0.1, 0.0, 0.0, 8.0, 8.0)


def _obs(step, sim_time):
    return Observation(
        x=2.5, y=2.5, theta=0.0, gas_ppm=0.0,
        wind_speed=0.3, wind_direction=0.0,
        lidar=np.full(72, 3.0), lidar_max_range=3.0,
        step=step, sim_time=sim_time, map=_map())


def test_adsm_capabilities_and_deterministic_trace():
    scenario = ScenarioInfo('test', _map(), 2.5, 2.5, 20, 0.5)
    agents = [AdsmAgent({'seed': 42, 'rrt_max_iter': 30}) for _ in range(2)]
    traces = []
    for agent in agents:
        agent.reset(scenario)
        trace = []
        for step in range(4):
            agent.observe(_obs(step, float(step + 1)))
            wp = agent.act()
            assert isinstance(wp, Waypoint)
            assert wp.theta == 0.0
            trace.append((wp.x, wp.y, agent.metadata()['goal_type']))
        traces.append(trace)
    assert traces[0] == traces[1]
    assert agents[0].motion_mode == 'continuous'
    assert agents[0].decision_rate_hz == 1.0
    assert agents[0].max_goal_distance == 3.0


def test_cpp_binding_rejects_non_grid_input():
    engine = adsm_core_py.Engine(adsm_core_py.Config())
    engine.reset(1)
    with pytest.raises(ValueError):
        engine.step(0, 0, 0, 0, 0, 0, 0, np.zeros(5, dtype=np.int8), 0.1, 0, 0)
