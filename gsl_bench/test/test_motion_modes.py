import math
from types import SimpleNamespace

from builtin_interfaces.msg import Time
from gsl_bench.harness.runner_node import RunnerNode


class _Now:
    nanoseconds = 1_000_000_000

    def to_msg(self):
        return Time(sec=1, nanosec=0)


class _Clock:
    def now(self):
        return _Now()


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _Navigator:
    def __init__(self):
        self.goals = []

    def send_goal(self, *args, **kwargs):
        self.goals.append((args, kwargs))


def _runner(mode, moving=False):
    return SimpleNamespace(
        _motion_mode=mode, _is_moving=moving, _drive_canceling=False,
        _drive_goal_time=None, _drive_goal_wall=None, _goal_update_pub=_Publisher(),
        _navigator=_Navigator(), _nav_goal_tolerance=0.1,
        _n_goal_updates=0, _n_action_starts=0, get_clock=lambda: _Clock())


def test_continuous_drive_updates_active_goal_without_new_action():
    r = _runner('continuous', moving=True)
    RunnerNode._drive(r, 2.0, 3.0, 0.4)
    assert not r._navigator.goals
    assert r._n_goal_updates == 1
    msg = r._goal_update_pub.messages[0]
    assert (msg.pose.position.x, msg.pose.position.y) == (2.0, 3.0)
    assert math.isclose(msg.pose.orientation.z, math.sin(0.2))


def test_stop_go_and_first_continuous_goal_start_action():
    for mode in ('stop_go', 'continuous'):
        r = _runner(mode, moving=False)
        RunnerNode._drive(r, 2.0, 3.0, 0.0)
        assert len(r._navigator.goals) == 1
        assert r._n_action_starts == 1
        assert r._is_moving


def test_per_agent_goal_reach_controls_cap():
    r = SimpleNamespace(_effective_max_hop=3.0)
    assert RunnerNode._cap_hop(r, 0.0, 0.0, 2.5, 0.0) == (2.5, 0.0)
    x, y = RunnerNode._cap_hop(r, 0.0, 0.0, 6.0, 0.0)
    assert (x, y) == (3.0, 0.0)
