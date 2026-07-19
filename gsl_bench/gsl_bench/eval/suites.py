"""Named scenario suites."""
from typing import List

from gsl_bench.eval import scenario_paths

# One representative of each map family — the 7-map suite every published number
# in this project is quoted against (7 scenarios x 5 runs = 35).
NAV7 = [
    '4_rooms_start_a',
    '10x6_u_left_1',
    '10x6_u_right_1',
    'curved_labrinth_left_1',
    'curved_labrinth_right_1',
    'many_rooms_1',
    'ultimate_1',
]

# The full published benchmark: every scenario family, every variant, House09
# deliberately excluded (its occupancy voxelization is unreliable). These are
# exactly the keys of Results/oracle_budgets_nav2_reliable.json — the oracle has
# a budget for each, so all 29 can be scored against a 5x/10x/20x envelope.
BENCHMARK29 = [
    '10x6_u_left_1', '10x6_u_left_2', '10x6_u_left_3', '10x6_u_left_4',
    '10x6_u_right_1', '10x6_u_right_2', '10x6_u_right_3', '10x6_u_right_4',
    '4_rooms_start_a', '4_rooms_start_b', '4_rooms_start_c', '4_rooms_start_d',
    '4_rooms_start_e',
    'curved_labrinth_left_1', 'curved_labrinth_left_2', 'curved_labrinth_left_3',
    'curved_labrinth_right_1', 'curved_labrinth_right_2', 'curved_labrinth_right_3',
    'many_rooms_1', 'many_rooms_2', 'many_rooms_3', 'many_rooms_4',
    'ultimate_1', 'ultimate_2', 'ultimate_3', 'ultimate_4', 'ultimate_5',
    'ultimate_6',
]


def resolve_suite(name: str) -> List[str]:
    if name == 'nav7':
        return list(NAV7)
    if name == 'benchmark29':
        return list(BENCHMARK29)
    if name == 'all':
        return scenario_paths.list_scenarios()
    raise KeyError(f"Unknown suite '{name}'. Known: nav7, benchmark29, all.")
