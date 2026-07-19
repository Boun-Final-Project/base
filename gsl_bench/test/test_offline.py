"""Offline checks: no ROS, no simulator. Run with: pytest src/gsl_bench/test"""
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gsl_bench.agent import MapInfo, Observation, ScenarioInfo, Waypoint
from gsl_bench.eval import metrics
from gsl_bench.registry import available_agents, load_agent_class


def _map():
    grid = np.zeros((60, 140), np.uint8)
    grid[:, 100:102] = 1                      # a wall at x ~ 9.8-10.0 m
    return MapInfo(grid=grid, resolution=0.1, origin_x=-0.2, origin_y=-0.2,
                   width_m=14.0, height_m=6.0)


def _obs(m, x=5.0, y=3.0, gas=0.0, wind_dir=0.0, wind_speed=0.0, step=0):
    return Observation(x=x, y=y, theta=0.0, gas_ppm=gas, wind_speed=wind_speed,
                       wind_direction=wind_dir, lidar=np.full(72, 2.5, np.float32),
                       lidar_max_range=3.0, step=step, sim_time=float(step), map=m)


def test_map_is_free():
    m = _map()
    assert m.is_free(5.0, 3.0)
    assert not m.is_free(9.9, 3.0)            # inside the wall
    assert not m.is_free(-5.0, 3.0)           # outside the map


def test_registry_lists_shipped_agents():
    assert {'random_walk', 'upwind_greedy', 'gpulaika',
            'zigzag', 'surge_cast'} <= set(available_agents())


@pytest.mark.parametrize('spec', ['random_walk', 'upwind_greedy', 'zigzag', 'surge_cast'])
def test_baseline_agents_step_into_free_space(spec):
    m = _map()
    agent = load_agent_class(spec)({'seed': 3, 'gas_threshold': 0.1})
    agent.initialize()
    agent.reset(ScenarioInfo('t', m, 5.0, 3.0, 600, 0.5))
    x, y = 5.0, 3.0
    for step in range(10):
        agent.observe(_obs(m, x, y, wind_dir=0.0, wind_speed=0.5, step=step))
        wp = agent.act()
        assert isinstance(wp, Waypoint)
        assert math.hypot(wp.x - x, wp.y - y) <= 0.51
        assert m.is_free(wp.x, wp.y)
        x, y = wp.x, wp.y


def test_upwind_greedy_goes_upwind_on_a_hit():
    m = _map()
    agent = load_agent_class('upwind_greedy')({'gas_threshold': 0.1})
    agent.initialize()
    agent.reset(ScenarioInfo('t', m, 5.0, 3.0, 600, 0.5))
    # wind blowing toward +x means the source is toward -x
    agent.observe(_obs(m, gas=1.0, wind_dir=0.0, wind_speed=0.5))
    wp = agent.act()
    assert wp.x < 5.0, 'should surge upwind (-x) on a gas hit'


def test_zigzag_alternates_sides():
    m = _map()
    agent = load_agent_class('zigzag')({'leg_length': 0.5, 'step_size': 0.5,
                                        'amplitude_growth': 0.0, 'upwind_drift': 0.0})
    agent.initialize()
    agent.reset(ScenarioInfo('t', m, 5.0, 3.0, 600, 0.5))
    x, y = 5.0, 3.0
    sides = []
    for step in range(4):
        agent.observe(_obs(m, x, y, wind_dir=0.0, wind_speed=0.5, step=step))
        wp = agent.act()
        sides.append(agent.side)
        x, y = wp.x, wp.y
    assert sides == [-1, 1, -1, 1], 'a full leg every step should flip side each call'


def test_zigzag_falls_back_to_random_walk_with_no_wind():
    m = _map()
    agent = load_agent_class('zigzag')({'seed': 1})
    agent.initialize()
    agent.reset(ScenarioInfo('t', m, 5.0, 3.0, 600, 0.5))
    agent.observe(_obs(m, wind_dir=0.0, wind_speed=0.0))
    wp = agent.act()
    assert isinstance(wp, Waypoint) and m.is_free(wp.x, wp.y)


def test_surge_cast_surges_upwind_on_hit():
    m = _map()
    agent = load_agent_class('surge_cast')({'gas_threshold': 0.1})
    agent.initialize()
    agent.reset(ScenarioInfo('t', m, 5.0, 3.0, 600, 0.5))
    agent.observe(_obs(m, gas=1.0, wind_dir=0.0, wind_speed=0.5))
    wp = agent.act()
    assert wp.x < 5.0, 'should surge upwind (-x) on a gas hit'
    assert agent.state == 'SURGE'


def test_surge_cast_casts_then_spirals_after_sustained_loss():
    m = _map()
    agent = load_agent_class('surge_cast')(
        {'gas_threshold': 0.1, 'lost_steps_to_spiral': 3})
    agent.initialize()
    agent.reset(ScenarioInfo('t', m, 5.0, 3.0, 600, 0.5))

    # one hit -> SURGE
    agent.observe(_obs(m, gas=1.0, wind_dir=0.0, wind_speed=0.5, step=0))
    agent.act()
    assert agent.state == 'SURGE'

    # lose the plume -> CAST
    agent.observe(_obs(m, gas=0.0, wind_dir=0.0, wind_speed=0.5, step=1))
    agent.act()
    assert agent.state == 'CAST'

    # keep losing it past lost_steps_to_spiral -> SPIRAL
    for step in range(2, 6):
        agent.observe(_obs(m, gas=0.0, wind_dir=0.0, wind_speed=0.5, step=step))
        agent.act()
    assert agent.state == 'SPIRAL'

    # a fresh hit drops straight back to SURGE
    agent.observe(_obs(m, gas=1.0, wind_dir=0.0, wind_speed=0.5, step=6))
    agent.act()
    assert agent.state == 'SURGE'


def test_aggregate_and_fairness_guard():
    runs = [
        {'scenario': 's1', 'success': True, 'status': 'success', 'steps': 10,
         'sim_time_s': 20.0, 'travel_distance_m': 5.0, 'final_distance_m': 0.4,
         'harness': {'max_steps': 600, 'escape': False}},
        {'scenario': 's1', 'success': False, 'status': 'max_steps', 'steps': 600,
         'sim_time_s': 900.0, 'travel_distance_m': 90.0, 'final_distance_m': 7.0,
         'harness': {'max_steps': 600, 'escape': False}},
    ]
    agg = metrics.aggregate(runs)
    s = agg['per_scenario']['s1']
    assert (s['n'], s['n_success'], s['success_rate']) == (2, 1, 0.5)
    assert s['mean_time_s'] == 20.0        # TT over successes only
    assert s['mean_distance_m'] == 5.0     # TD over successes only
    assert s['mean_final_distance_m_failures'] == 7.0
    assert agg['harness_mismatch'] == []

    runs[1]['harness']['escape'] = True    # not comparable any more
    assert metrics.aggregate(runs)['harness_mismatch'] == ['escape']


def test_load_runs_reads_result_json():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'run_1'
        p.mkdir()
        (p / 'result.json').write_text(json.dumps({'scenario': 's', 'success': True}))
        runs = metrics.load_runs(Path(d))
        assert len(runs) == 1 and runs[0]['scenario'] == 's'
