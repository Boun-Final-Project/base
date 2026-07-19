"""gsl_bench eval — the batch runner. One Python port of the bash drivers.

    ros2 run gsl_bench eval --agent gpulaika --suite nav7 --runs 5 --escape

Owns: process-group isolation, headless display, bake-if-needed, the install-tree
patches, env launch + ground-truth wait, the runner node, wall timeouts, cleanup,
and the aggregate report. Everything that used to be copy-pasted between three
bash scripts and drifted.
"""
import argparse
import atexit
import filecmp
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from gsl_bench.eval import metrics, report, scenario_paths, suites

TUNED_NAV2_PARAMS = Path(
    '/home/efe/ros2_ws/src/base/gaden_config/navigation_config/nav2_params.yaml')

BOLD, DIM, CYAN, GREEN, YELLOW, RED, RESET = (
    '\033[1m', '\033[2m', '\033[36m', '\033[32m', '\033[33m', '\033[31m', '\033[0m')

_XVFB_PROC: Optional[subprocess.Popen] = None
_LIVE_PROCS: List[subprocess.Popen] = []   # launch processes whose trees we must reap
_CLEANED = False


# ---------------------------------------------------------------------------
# Process hygiene
# ---------------------------------------------------------------------------

def reexec_under_setsid():
    """Re-exec in a fresh session so cleanup is pgid-scoped.

    Everything we spawn (ros2 launch, the node, Xvfb and their whole subtrees)
    inherits this pgid, so kill_ros can kill by pgid instead of by name — which is
    load-bearing when several jobs are packed on one host: a name-based
    `pkill -f "ros2 run"` would take out a sibling job's processes too.

    `setsid -w` is required: plain `setsid cmd` forks and exits immediately, so the
    parent would think the sweep finished in a second while it ran on detached.
    """
    if os.environ.get('GSLB_SETSID_DONE'):
        return
    os.environ['GSLB_SETSID_DONE'] = '1'
    os.execvp('setsid', ['setsid', '-w', sys.executable, '-u'] + sys.argv)


# A ROS node binary we are allowed to reap when it is on OUR domain. Deliberately
# an allowlist of executable paths, not a loose name match: `pkill -f ros` on a
# shared host kills sibling jobs.
_ROS_BIN = re.compile(
    r'/opt/ros/[^/]+/lib/|/install/(gaden|basic_sim|nav2|olfaction|gsl_bench/lib/gsl_bench/runner_node)'
    r'|basic_sim|gaden_|ros2cli\.daemon')


def _descendants(root_pid: int) -> List[int]:
    """Every live descendant of root_pid, deepest first.

    Must be called while root_pid is still ALIVE. `ros2 launch` starts each node with
    its own session (start_new_session), so those nodes are NOT in our process group
    and a pgid-scoped kill structurally cannot reach them — kill the launch parent
    first and they simply reparent to init and keep publishing. That is the zombie
    cross-talk that poisoned whole sweeps: an orphan from run N republishing on a
    reused ROS_DOMAIN_ID in run N+1. So walk the tree before touching the parent.
    """
    ps = subprocess.run(['ps', '-eo', 'pid=,ppid='],
                        capture_output=True, text=True).stdout
    children = {}
    for line in ps.splitlines():
        try:
            pid, ppid = (int(v) for v in line.split())
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)

    out, stack = [], [root_pid]
    while stack:
        pid = stack.pop()
        for c in children.get(pid, []):
            out.append(c)
            stack.append(c)
    return list(reversed(out))


def kill_tree(proc: Optional[subprocess.Popen]):
    """Kill a launch process and its whole (multi-session) descendant tree."""
    if proc is None or proc.poll() is not None:
        pids = []
    else:
        pids = _descendants(proc.pid)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def reap_domain(domain: Optional[str]) -> int:
    """Backstop: kill leftover ROS nodes that are on OUR ROS_DOMAIN_ID.

    Catches orphans a previous crashed sweep left behind. Domain-scoped by reading
    each process's own environ, so a co-resident job on a different domain is never
    touched — unlike a name-based pkill.
    """
    if not domain:
        return 0
    me, killed = os.getpid(), 0
    needle = f'ROS_DOMAIN_ID={domain}'.encode()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == me:
            continue
        try:
            if needle not in (p / 'environ').read_bytes().split(b'\0'):
                continue
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='ignore')
        except OSError:
            continue
        if not _ROS_BIN.search(cmd):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except OSError:
            pass
    return killed


def kill_ros(domain_id: Optional[str] = None):
    """Kill our own process group, reap any stragglers on our domain, stop the daemon."""
    domain = domain_id or os.environ.get('ROS_DOMAIN_ID')

    try:
        my_pgid = os.getpgid(0)
    except OSError:
        my_pgid = None

    if my_pgid is not None:
        out = subprocess.run(['pgrep', '-g', str(my_pgid)],
                             capture_output=True, text=True).stdout.split()
        me = os.getpid()
        for pid_s in out:
            pid = int(pid_s)
            if pid == me:
                continue
            comm = subprocess.run(['ps', '-o', 'comm=', '-p', pid_s],
                                  capture_output=True, text=True).stdout.strip()
            if comm in ('Xvfb', 'tee'):     # our own long-lived infrastructure
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    reap_domain(domain)

    # The ros2 CLI daemon deliberately daemonizes OUT of our process group, so the
    # pgid kill above cannot reach it. Left alive it holds FastRTPS shared-memory
    # ports open and poisons the next job that reuses this ROS_DOMAIN_ID.
    subprocess.run(['ros2', 'daemon', 'stop'],
                   capture_output=True, timeout=30, check=False)
    time.sleep(2)


def _cleanup():
    global _CLEANED
    if _CLEANED:
        return
    _CLEANED = True
    for proc in list(_LIVE_PROCS):
        kill_tree(proc)
    kill_ros()
    if _XVFB_PROC is not None and _XVFB_PROC.poll() is None:
        try:
            _XVFB_PROC.kill()
        except OSError:
            pass


def start_xvfb() -> int:
    """Headless display. benchmark_env's stock launch hardcodes xterm + rviz.

    The display number must be unique per packed task: a bare `pkill -9 Xvfb` (or two
    tasks both on :99) silently kills a co-resident task's display mid-run, and the
    launch then dies with no error. Key it off ROS_DOMAIN_ID when we have one.
    """
    global _XVFB_PROC
    num = int(os.environ.get('ROS_DOMAIN_ID') or (99 + os.getpid() % 512))
    subprocess.run(['pkill', '-9', '-f', f'Xvfb :{num} '], check=False)
    time.sleep(1)
    log = open(f'/tmp/gslb_xvfb_{num}.log', 'wb')
    _XVFB_PROC = subprocess.Popen(
        ['Xvfb', f':{num}', '-screen', '0', '1024x768x24'], stdout=log, stderr=log)
    os.environ['DISPLAY'] = f':{num}'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    time.sleep(3)
    return num


# ---------------------------------------------------------------------------
# Install-tree patches
# ---------------------------------------------------------------------------

def patch_no_rviz(share: Path) -> bool:
    """Stop the stock launch from starting rviz2 and xterm.

    rviz2 is not exposed as a launch arg, and even under Xvfb it eats enough CPU on a
    loaded host that gaden_environment's occupancyMap3D response misses the node's
    startup timeout — a silent hang, no error. Patches the INSTALLED copy only.
    """
    f = share / 'launch' / 'main_simbot_launch.py'
    if not f.is_file():
        return False
    text = f.read_text()
    orig = text
    text = text.replace('"use_rviz": "True"', '"use_rviz": "False"')
    text = re.sub(r'\s*prefix\s*=\s*[\'"]xterm -hold -e[\'"]\s*,', '', text)
    if text == orig:
        return False
    try:
        f.write_text(text)     # read-only install tree (enroot/Pyxis) -> skip, Xvfb covers us
        return True
    except OSError:
        return False


def patch_nav2_params(share: Path) -> bool:
    """Install the tuned nav2 params (no RotateToGoal critic).

    Stock benchmark_env ships the untuned default, which wedges the robot at walls
    under Nav2 drive.
    """
    target = share / 'navigation_config' / 'nav2_params.yaml'
    if not TUNED_NAV2_PARAMS.is_file() or not target.parent.is_dir():
        return False
    if target.is_file() and filecmp.cmp(TUNED_NAV2_PARAMS, target, shallow=False):
        return False           # already identical (e.g. baked into a container image)
    try:
        shutil.copy(TUNED_NAV2_PARAMS, target)
        return True
    except OSError:
        return False


def patch_lidar_config(scenario: str) -> bool:
    """360 deg / 72 rays / 3 m — what every checkpoint here was trained against.

    MUST run AFTER preproc: preproc regenerates BasicSimScene.yaml from empty_point.
    """
    f = scenario_paths.basic_sim_scene(scenario)
    if not f.is_file():
        return False
    text = f.read_text()
    if re.search(r'maxAngleRad:\s*6\.2832', text):
        return False
    for pat, sub in [
        (r'minAngleRad:\s*\S+', 'minAngleRad: 0.0'),
        (r'maxAngleRad:\s*\S+', 'maxAngleRad: 6.2832  # 2pi — full 360 deg'),
        (r'angleResolutionRad:\s*\S+', 'angleResolutionRad: 0.08727  # 2pi / 72'),
        (r'maxDistance:\s*\S+', 'maxDistance: 3.0  # matches LIDAR_MAX_RANGE'),
    ]:
        text = re.sub(pat, sub, text, count=1)
    try:
        f.write_text(text)
        return True
    except OSError:
        return False


def patch_playback_loop(scenario: str, sim: str) -> bool:
    """Make gas playback loop instead of running off the end of the bake.

    Every scenario ships loop:false, so a long episode outlasts the baked iterations
    and spams "File ... does not exist!" forever. Recomputed from the ACTUAL result
    dir each run so it self-heals after a re-bake. Loops back to
    playback_initial_iteration (already mature gas), not 0, so there is no
    gas-buildup discontinuity.
    """
    scene = scenario_paths.scene1_yaml(scenario)
    rdir = scenario_paths.result_dir(scenario, sim)
    if not scene.is_file() or not rdir.is_dir():
        return False
    text = scene.read_text()
    m = re.search(r'playback_initial_iteration:\s*(\d+)', text)
    if not m:
        return False
    initial = int(m.group(1))
    iters = [int(mm.group(1)) for f in rdir.iterdir()
             if (mm := re.match(r'iteration_(\d+)$', f.name))]
    if not iters:
        return False
    max_iter = max(iters)
    if max_iter <= initial:
        return False
    block = f'playback_loop:\n  loop: true\n  from: {initial}\n  to: {max_iter}'
    new_text, n = re.subn(
        r'playback_loop:\s*\n\s*loop:\s*\S+\s*\n\s*from:\s*\d+\s*\n\s*to:\s*\d+',
        block, text)
    if not n:
        return False
    try:
        scene.write_text(new_text)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Baking
# ---------------------------------------------------------------------------

def is_baked(scenario: str, sim: str) -> bool:
    """Baked iff result/ is non-empty AND newer than sim.yaml and config.yaml."""
    rdir = scenario_paths.result_dir(scenario, sim)
    if not rdir.is_dir():
        return False
    files = list(rdir.iterdir())
    if not files:
        return False
    newest = max(f.stat().st_mtime for f in files)
    cdir = scenario_paths.config_dir(scenario)
    for cfg in (cdir / 'simulations' / sim / 'sim.yaml', cdir / 'config.yaml'):
        if cfg.is_file() and cfg.stat().st_mtime > newest:
            return False
    return True


def bake_if_needed(scenario: str, sim: str, skip: bool) -> bool:
    if skip or is_baked(scenario, sim):
        return True
    print(f'  {YELLOW}Baking gas{RESET} (missing or stale) — this is slow...')
    for launch in ('gaden_preproc_launch.py', 'gaden_sim_launch.py'):
        r = subprocess.run(
            ['ros2', 'launch', 'benchmark_env', launch,
             f'scenario:={scenario}', f'simulation:={sim}'],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f'  {RED}{launch} failed (rc={r.returncode}){RESET}')
            return False
    return True


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def wait_for_pose(ns: str, timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ['ros2', 'topic', 'echo', f'/{ns}/ground_truth', '--once'],
                capture_output=True, text=True, timeout=10, check=False)
        except subprocess.TimeoutExpired:
            # `ros2 topic echo` itself can hang past its own timeout when the
            # graph is mid-bringup — treat like an empty response and keep
            # retrying instead of letting this blow up the whole sweep.
            continue
        if 'header' in r.stdout:
            return True
        time.sleep(2)
    return False


def _fmt(v) -> str:
    """ROS rejects '3' for a DOUBLE parameter — whole numbers must carry a decimal."""
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return repr(float(v))
    return str(v)


def write_stub_result(path: Path, scenario: str, sim: str, agent: str, status: str,
                      source, start, wall_s: float, harness: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'schema': 1, 'scenario': scenario, 'sim': sim, 'agent': agent,
            'agent_config': {}, 'status': status, 'success': False, 'steps': 0,
            'sim_time_s': None, 'wall_time_s': round(wall_s, 1),
            'travel_distance_m': None, 'final_distance_m': None,
            'source': list(source), 'start': list(start),
            'escapes': 0, 'clamped_steps': 0, 'drive_timeouts': 0,
            'harness': harness,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }, f, indent=2)


def realistic_share() -> Optional[Path]:
    try:
        prefix = subprocess.run(
            ['ros2', 'pkg', 'prefix', 'benchmark_env_realistic'],
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    d = Path(prefix) / 'share' / 'benchmark_env_realistic'
    return d if d.is_dir() else None


def start_visual(scenario: str, start, log_dir: Path) -> List[subprocess.Popen]:
    """Launch the world-align relay + rviz2 for watch mode. Returns their procs."""
    procs: List[subprocess.Popen] = []
    yaw = scenario_paths.start_yaw(scenario)
    align = subprocess.Popen(
        ['ros2', 'run', 'gsl_bench', 'world_align', '--ros-args',
         '-p', f'start_x:={_fmt(float(start[0]))}',
         '-p', f'start_y:={_fmt(float(start[1]))}',
         '-p', f'yaw:={_fmt(float(yaw))}'],
        stdout=open(log_dir / 'world_align.log', 'wb'), stderr=subprocess.STDOUT)
    procs.append(align)

    share = realistic_share()
    rviz_cfg = share / 'navigation_config' / 'gsl_watch.rviz' if share else None
    if rviz_cfg and rviz_cfg.is_file():
        rviz = subprocess.Popen(
            ['rviz2', '-d', str(rviz_cfg)],
            stdout=open(log_dir / 'rviz.log', 'wb'), stderr=subprocess.STDOUT)
        procs.append(rviz)
        print(f'  {BOLD}rviz{RESET}: {rviz_cfg.name}  (align start=({start[0]:.1f},'
              f'{start[1]:.1f}) yaw={yaw:.2f})')
    else:
        print(f'  {YELLOW}gsl_watch.rviz not found — align node only{RESET}')
    for p in procs:
        _LIVE_PROCS.append(p)
    return procs


def run_one(args, scenario: str, sim: str, run_dir: Path, agent_cfg_path: Optional[Path],
            source, start, harness: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_dir / 'result.json'
    t0 = time.time()

    env_log = open(run_dir / 'env.log', 'wb')
    launch_pkg = args.launch
    launch_file = ('realistic_launch.py' if launch_pkg == 'benchmark_env_realistic'
                   else 'main_simbot_launch.py')
    launch_cmd = ['ros2', 'launch', launch_pkg, launch_file,
                  f'scenario:={scenario}', 'configuration:=config1', f'simulation:={sim}',
                  f'namespace:={args.namespace}']
    if launch_pkg == 'benchmark_env_realistic':
        launch_cmd.append(f'motion:={args.motion}')
        launch_cmd.append(f'nav_profile:={args.nav_profile}')
        launch_cmd.append(f'lidar_max_range:={_fmt(float(args.lidar_max_range))}')
        launch_cmd.append(f'wall_outlets:={args.wall_outlets}')
        # DirectDrive + ground-truth isolation: when the policy runs on
        # ground-truth pose/service map, DirectDrive must navigate in the same
        # (world) frame — ground_truth pose topic + the scenario's static
        # world-frame occupancy.yaml — or the world-frame drive targets would be
        # misinterpreted in the SLAM frame.
        if (args.motion == 'directdrive' and args.pose_source == 'ground_truth'):
            occ_yaml = scenario_paths.occupancy_yaml(scenario)
            launch_cmd.append('drive_pose_source:=topic')
            launch_cmd.append(f'drive_map_yaml:={occ_yaml}')
    env_proc = subprocess.Popen(
        launch_cmd, stdout=env_log, stderr=subprocess.STDOUT)
    _LIVE_PROCS.append(env_proc)

    print('  waiting for ground_truth...')
    if not wait_for_pose(args.namespace):
        # Never start the agent against a dead env: it would burn the whole
        # wall-timeout budget before failing, turning a 2-minute abort into a
        # 30-minute one.
        print(f'  {RED}env never published ground_truth — aborting run{RESET}')
        write_stub_result(result_file, scenario, sim, args.agent, 'env_dead',
                          source, start, time.time() - t0, harness)
        kill_tree(env_proc)
        _LIVE_PROCS.remove(env_proc)
        kill_ros()
        time.sleep(3)
        return json.loads(result_file.read_text())

    print('  pose online, starting agent.')
    time.sleep(3)

    vis_procs: List[subprocess.Popen] = []
    if getattr(args, 'visual', False):
        vis_procs = start_visual(scenario, start, run_dir)

    cmd = ['ros2', 'run', 'gsl_bench', 'runner_node', '--ros-args',
           '-p', f'agent:={args.agent}',
           '-p', f'scenario_name:={scenario}',
           '-p', f'sim_name:={sim}',
           '-p', f'true_source_x:={_fmt(float(source[0]))}',
           '-p', f'true_source_y:={_fmt(float(source[1]))}',
           '-p', f'start_x:={_fmt(float(start[0]))}',
           '-p', f'start_y:={_fmt(float(start[1]))}',
           '-p', f'max_steps:={int(args.max_steps)}',
           '-p', f'max_travel_distance_m:={_fmt(float(harness.get("max_travel_distance_m", 0.0)))}',
           '-p', f'max_sim_time_s:={_fmt(float(harness.get("max_sim_time_s", 0.0)))}',
           '-p', f'step_delay:={_fmt(float(args.step_delay))}',
           '-p', f'success_radius:={_fmt(float(args.success_radius))}',
           '-p', f'nav_goal_tolerance:={_fmt(float(args.nav_goal_tolerance))}',
           '-p', f'drive_timeout:={_fmt(float(args.drive_timeout))}',
           '-p', f'max_hop:={_fmt(float(args.max_hop))}',
           '-p', f'robot_radius:={_fmt(float(args.robot_radius))}',
            '-p', f'escape:={_fmt(bool(args.escape))}',
            '-p', f'namespace:={args.namespace}',
            '-p', f'result_file:={result_file}',
            '-p', f'pose_source:={args.pose_source}',
           '-p', f'map_source:={args.map_source}',
            '-p', f'motion:={args.motion}',
            '-p', f'nav_profile:={args.nav_profile}',
            # The whole sim stack (basic_sim /clock, cartographer, Nav2 — see
            # main_simbot_launch.py and nav2_realistic_launch.py) runs on sim time.
            # runner_node MUST match, or its goal stamps (get_clock().now()) carry
            # wall-epoch time while TF is at sim time: the controller then errors
            # "Transform data too old when converting from map to odom", produces no
            # motion, and the harness counts the step anyway (0 travel, 600 fake
            # Nav2 actions). _sim_time()/max_sim_time budgeting is also wall-based
            # without this. Runner starts after /clock is already flowing.
            '-p', 'use_sim_time:=true']
    if agent_cfg_path is not None:
        cmd += ['-p', f'agent_config:={agent_cfg_path}']

    node_log = open(run_dir / 'node.log', 'wb')
    node_proc = subprocess.Popen(cmd, stdout=node_log, stderr=subprocess.STDOUT)

    status = 'done'
    while node_proc.poll() is None:
        if time.time() - t0 > args.wall_timeout:
            status = 'wall_timeout'
            print(f'  {YELLOW}WALL TIMEOUT{RESET} after {time.time() - t0:.0f}s — killing.')
            node_proc.terminate()
            time.sleep(2)
            node_proc.kill()
            break
        time.sleep(2)

    wall_s = time.time() - t0
    if not result_file.is_file():
        write_stub_result(
            result_file, scenario, sim, args.agent,
            'wall_timeout' if status == 'wall_timeout' else 'crashed',
            source, start, wall_s, harness)
    else:
        res = json.loads(result_file.read_text())
        res['wall_time_s'] = round(wall_s, 1)
        result_file.write_text(json.dumps(res, indent=2))

    res = json.loads(result_file.read_text())
    if res.get('success'):
        print(f'  {GREEN}SUCCESS{RESET}  steps={res["steps"]}  '
              f'd={res["final_distance_m"]}m  ({wall_s:.0f}s wall)')
    else:
        print(f'  {YELLOW}not solved{RESET}  status={res["status"]}  '
              f'steps={res.get("steps")}  ({wall_s:.0f}s wall)')

    for p in vis_procs:
        kill_tree(p)
        if p in _LIVE_PROCS:
            _LIVE_PROCS.remove(p)
    # Tree-kill the env FIRST, while the launch parent is still alive and its
    # (separately-sessioned) nodes are still reachable through it.
    kill_tree(env_proc)
    _LIVE_PROCS.remove(env_proc)
    kill_ros()
    time.sleep(3)
    return res


def label_success_within_budget(result_file: Path, budget: Optional[dict]) -> None:
    """Post-hoc relabel a run against the 5x-oracle success budget.

    The runner terminates a run the instant it reaches the source (0.5 m), so its
    raw `success` really means "reached the source at all" — the robot may have
    wandered up to 10x the oracle distance getting there. The published metric is
    stricter: a reach only counts if it was efficient (travel <= 5x oracle). We
    preserve the raw reach as `reached_source` and overwrite `success` with the
    budget-gated verdict so metrics/report headline the stricter number.

    Runs with no oracle record (budget is None) keep success == reached_source.
    """
    if not result_file.is_file():
        return
    try:
        res = json.loads(result_file.read_text())
    except (json.JSONDecodeError, OSError):
        return
    reached = bool(res.get('success'))
    res['reached_source'] = reached
    if budget is not None:
        res.update(budget)
        travel = res.get('travel_distance_m')
        within = (travel is not None
                  and travel <= budget['success_budget_distance_m'])
        res['success_within_budget'] = bool(reached and within)
        res['success'] = res['success_within_budget']
    else:
        res['success_within_budget'] = reached
    result_file.write_text(json.dumps(res, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]):
    ap = argparse.ArgumentParser(prog='gsl_bench eval')
    ap.add_argument('--agent', required=True,
                    help="registered name (random_walk, upwind_greedy, gpulaika) "
                         "or 'pkg.module:ClassName'")
    ap.add_argument('--agent-config', type=Path, default=None,
                    help='YAML handed to the agent constructor')
    ap.add_argument('--suite', default=None, help='nav7 | all')
    ap.add_argument('--scenario', default=None,
                    help='scenario name, or a comma-separated list')
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--out', type=Path,
                    default=Path('/home/efe/ros2_ws/Results'))
    ap.add_argument('--wall-timeout', type=float, default=1800.0)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--oracle-budgets', type=Path, default=None,
                    help='JSON manifest from `ros2 run gsl_bench oracle` '
                         '({"<scenario>_<sim>": {"travel_distance_m":.., "sim_time_s":..}}). '
                         'When set, each scenario is scored against a 5x/10x/20x envelope '
                         'of its oracle record (see the three --*-budget-multiplier flags), '
                         'on top of --max-steps.')
    ap.add_argument('--budget-multiplier', type=float, default=10.0,
                    help='DISTANCE termination multiplier: the episode is terminated as a '
                         'failure once travel exceeds this x oracle distance '
                         '(only with --oracle-budgets)')
    ap.add_argument('--time-budget-multiplier', type=float, default=20.0,
                    help='TIME termination multiplier: the episode is terminated as a '
                         'failure once sim time exceeds this x oracle time '
                         '(only with --oracle-budgets)')
    ap.add_argument('--success-budget-multiplier', type=float, default=5.0,
                    help='SUCCESS labelling multiplier: reaching the source only counts as '
                         'success_within_budget if travel <= this x oracle distance. The '
                         'robot keeps running past this up to the termination budget, so '
                         'reaches between 5x and 10x are recorded but not counted '
                         '(only with --oracle-budgets)')
    ap.add_argument('--step-delay', type=float, default=0.5)
    ap.add_argument('--success-radius', type=float, default=0.5)
    ap.add_argument('--nav-goal-tolerance', type=float, default=0.10)
    ap.add_argument('--drive-timeout', type=float, default=30.0)
    ap.add_argument('--max-hop', type=float, default=1.0)
    ap.add_argument('--robot-radius', type=float, default=0.25)
    ap.add_argument('--namespace', default='PioneerP3DX')
    ap.add_argument('--escape', action='store_true',
                    help='harness stuck-escape recovery (recorded in every result)')
    ap.add_argument('--skip-bake', action='store_true')
    ap.add_argument('--domain-id', type=int, default=None)
    ap.add_argument('--pose-source', default='ground_truth',
                    choices=['ground_truth', 'tf'],
                    help='ground_truth: perfect pose from /ground_truth topic; '
                         'tf: pose from TF lookup (map -> base_link)')
    ap.add_argument('--map-source', default='service',
                    choices=['service', 'slam'],
                    help='service: ground-truth map from Gaden; '
                         'slam: incremental map from Cartographer topic')
    ap.add_argument('--realistic', action='store_true',
                    help='shortcut for --pose-source tf --map-source slam '
                         '--launch benchmark_env_realistic')
    ap.add_argument('--visual', action='store_true',
                    help='open a live RViz (SLAM map + gas + robot + goal) on the '
                         'shared VNC display :99 for each run, with the GADEN '
                         'world<->SLAM map origin fix. Realistic mode only; intended '
                         'for watching one scenario (--scenario X --runs 1), not '
                         'headless sweeps')
    ap.add_argument('--launch', default='benchmark_env',
                    help='launch package for the environment '
                         '(benchmark_env or benchmark_env_realistic)')
    ap.add_argument('--motion', default='nav2', choices=['nav2', 'directdrive'],
                    help='drive mechanism, benchmark_env_realistic only: nav2 (full '
                         'Nav2 stack) or directdrive (rotate-then-go A* on the live '
                         'SLAM map, no Nav2 — bypasses Nav2 planning failures)')
    ap.add_argument('--nav-profile', default='faithful',
                    choices=['standard', 'faithful'],
                    help='benchmark_env_realistic Nav2 parameters. faithful restores '
                         'the original ROS1 DWA reverse and rotation behavior; ignored '
                         'when --motion directdrive is selected')
    ap.add_argument('--lidar-max-range', type=float, default=3.0,
                    help='benchmark_env_realistic only: BasicSim sensor range in '
                         'meters. The agent\'s own lidar features stay clipped to '
                         '3.0 in obs_cache regardless of this value')
    ap.add_argument('--wall-outlets', default='true', choices=['true', 'false'],
                    help='benchmark_env_realistic only: mark GADEN outlet cells as '
                         'occupied in the map the realistic lidar sees, so they map '
                         'as walls like the ground-truth path (default true)')
    args = ap.parse_args(argv)
    if not args.suite and not args.scenario:
        args.suite = 'nav7'
    if args.realistic:
        args.pose_source = 'tf'
        args.map_source = 'slam'
        args.launch = 'benchmark_env_realistic'
    if args.visual and args.launch != 'benchmark_env_realistic':
        ap.error('--visual needs the SLAM map — pass --realistic')
    if args.agent == 'adsm':
        if not args.realistic:
            ap.error('ADSM currently supports realistic perception only; pass --realistic')
        if args.escape:
            ap.error('canonical ADSM requires harness escape to remain disabled')
        if args.motion != 'nav2':
            ap.error('ADSM continuous goal updates require --motion nav2')
        args.nav_profile = 'faithful'
        args.max_hop = max(args.max_hop, 3.0)
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)

    if args.domain_id is not None:
        os.environ['ROS_DOMAIN_ID'] = str(args.domain_id)
    elif not os.environ.get('ROS_DOMAIN_ID'):
        print(f'{YELLOW}Warning: ROS_DOMAIN_ID unset. Never run two sweeps at once on '
              f'one host without distinct --domain-id.{RESET}')

    reexec_under_setsid()   # everything below runs in our own session

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(1))

    scenario_list = ([s.strip() for s in args.scenario.split(',') if s.strip()]
                     if args.scenario else suites.resolve_suite(args.suite))
    share = scenario_paths.install_share_dir()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    agent_tag = re.sub(r'[^A-Za-z0-9_.-]', '_', args.agent)
    out_root = args.out / f'{agent_tag}_{stamp}'
    out_root.mkdir(parents=True, exist_ok=True)

    harness = {
        'success_radius': args.success_radius, 'max_steps': args.max_steps,
        'step_delay': args.step_delay, 'nav_goal_tolerance': args.nav_goal_tolerance,
        'drive_timeout': args.drive_timeout, 'escape': bool(args.escape),
        'max_hop': args.max_hop, 'robot_radius': args.robot_radius,
        'lidar': '360deg/72ray/3m', 'motion': args.motion,
        'nav_profile': args.nav_profile,
        'pose_source': args.pose_source, 'map_source': args.map_source,
        'launch': args.launch,
        'lidar_max_range': args.lidar_max_range,
        'wall_outlets': args.wall_outlets,
    }

    print(f'\n{BOLD}{CYAN} gsl_bench eval{RESET}')
    print(f'  agent     : {BOLD}{args.agent}{RESET}')
    print(f'  scenarios : {len(scenario_list)} x {args.runs} runs = '
          f'{len(scenario_list) * args.runs}')
    print(f'  escape    : {"ON" if args.escape else "OFF"}')
    print(f'  pose      : {BOLD}{args.pose_source}{RESET}  '
          f'map: {BOLD}{args.map_source}{RESET}  '
          f'launch: {args.launch}')
    print(f'  output    : {BOLD}{out_root}{RESET}\n')

    if args.visual:
        # Watch mode: use the shared VNC display (:99) instead of a private
        # headless Xvfb, so rviz2 is viewable. watch.sh brings up Xvfb + x11vnc.
        os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':99')
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
        os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
        print(f'  {BOLD}visual mode{RESET}: rviz on DISPLAY={os.environ["DISPLAY"]}'
              f'  (view over VNC :5900)')
    else:
        display = start_xvfb()
        print(f'  {DIM}headless display :{display}{RESET}')

    kill_ros()
    if patch_no_rviz(share):
        print(f'  {DIM}patched install launch: no rviz, no xterm{RESET}')
    if patch_nav2_params(share):
        print(f'  {DIM}patched install nav2_params: tuned (no RotateToGoal){RESET}')
    # Start from a clean daemon: a stale graph cache makes `ros2 topic echo --once`
    # return nothing while the graph underneath is healthy, so every run would look
    # like "env never started".
    subprocess.run(['ros2', 'daemon', 'start'], capture_output=True, check=False)
    time.sleep(2)

    base_cfg = {}
    if args.agent_config:
        base_cfg = yaml.safe_load(args.agent_config.read_text()) or {}

    oracle_budgets = {}
    if args.oracle_budgets:
        oracle_budgets = json.loads(args.oracle_budgets.read_text())
        print(f'  {DIM}oracle budgets: {args.oracle_budgets} '
              f'(x{args.budget_multiplier}){RESET}')

    total = len(scenario_list) * args.runs
    done = 0
    for scenario in scenario_list:
        if not scenario_paths.scenario_dir(scenario).is_dir():
            print(f'{RED}Unknown scenario: {scenario} — skipping{RESET}')
            continue
        sim = scenario_paths.resolve_sim(scenario)
        print(f'\n{DIM}{"=" * 50}{RESET}\n{BOLD} {scenario}{RESET} ({sim})')

        if not bake_if_needed(scenario, sim, args.skip_bake):
            print(f'{RED}bake failed — skipping {scenario}{RESET}')
            continue
        patch_playback_loop(scenario, sim)   # after the bake: keyed off the result dir
        patch_lidar_config(scenario)         # after preproc: it regenerates the scene
        patch_nav2_params(share)

        source = scenario_paths.source_xy(scenario, sim)
        start = scenario_paths.start_xy(scenario)
        wind = scenario_paths.wind_csv(scenario)
        print(f'  source ({source[0]:.2f}, {source[1]:.2f})  '
              f'start ({start[0]:.2f}, {start[1]:.2f})')

        scen_harness = dict(harness)
        scen_budget = None
        if oracle_budgets:
            rec = oracle_budgets.get(f'{scenario}_{sim}')
            if rec is None:
                print(f'  {YELLOW}no oracle record for {scenario}_{sim} — '
                      f'falling back to max_steps only{RESET}')
            else:
                oracle_dist = float(rec['travel_distance_m'])
                oracle_time = float(rec['sim_time_s'])
                scen_harness['max_travel_distance_m'] = args.budget_multiplier * oracle_dist
                scen_harness['max_sim_time_s'] = args.time_budget_multiplier * oracle_time
                scen_budget = {
                    'oracle_travel_distance_m': round(oracle_dist, 3),
                    'oracle_sim_time_s': round(oracle_time, 3),
                    'success_budget_distance_m': round(
                        args.success_budget_multiplier * oracle_dist, 3),
                    'success_budget_multiplier': args.success_budget_multiplier,
                }
                print(f'  budget    : success<={scen_budget["success_budget_distance_m"]:.1f}m'
                      f'  terminate dist<={scen_harness["max_travel_distance_m"]:.1f}m'
                      f'  time<={scen_harness["max_sim_time_s"]:.1f}s')

        # Per-scenario agent config: the wind CSV is a property of the scenario, not
        # of the agent, so the runner injects it rather than the user hardcoding it.
        scen_dir = out_root / f'{scenario}_{sim}'
        scen_dir.mkdir(parents=True, exist_ok=True)
        agent_cfg_path = None
        if base_cfg or wind:
            cfg = dict(base_cfg)
            if wind and not cfg.get('wind_file'):
                cfg['wind_file'] = str(wind)
            agent_cfg_path = scen_dir / 'agent_config.yaml'
            agent_cfg_path.write_text(yaml.safe_dump(cfg))

        for i in range(1, args.runs + 1):
            done += 1
            print(f'\n{BOLD} [{done}/{total}] {scenario} run {i}/{args.runs}{RESET} '
                  f'{DIM}{datetime.now():%H:%M:%S}{RESET}')
            run_dir = scen_dir / f'run_{i}'
            run_one(args, scenario, sim, run_dir, agent_cfg_path,
                    source, start, scen_harness)
            label_success_within_budget(run_dir / 'result.json', scen_budget)

    text = report.write_all(out_root)
    print('\n' + text)
    print(f'Results: {BOLD}{out_root}{RESET}')

    runs = metrics.load_runs(out_root)
    agg = metrics.aggregate(runs)
    o = agg['overall']
    print(f"{BOLD}{GREEN}Overall: {o['n_success']}/{o['n']} "
          f"({o['success_rate'] * 100:.0f}%){RESET}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
