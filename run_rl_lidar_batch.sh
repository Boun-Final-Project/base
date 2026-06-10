#!/bin/bash
# ============================================================
#  gaden_rl_node_lidar Batch Runner
#
#  Modern replacement for run_rl_batch.sh. Drives the lidar-only
#  deployment node that lives in the gaden_transfer package
#  (src/base/gaden_transfer/gaden_transfer_lidar/). That node is a strict
#  superset of the old reinforcement_learning gaden_rl_node:
#    - same arch options: mlp | modular | dual
#    - additionally supports arch=spatial (CNN + FiLM wind)
#    - fresh-scan gate: blocks the first post-teleport step until a
#      LaserScan with stamp > teleport stamp arrives (fixes pre-teleport
#      lidar splatter into the post-teleport spatial obs)
#    - optional per-step PNG dumps (RL_DEBUG_DUMP=1) for the spatial arch
#
#  The reinforcement_learning package no longer ships a console_script
#  for gaden_rl_node — keep using run_rl_batch.sh only if you have a
#  reason to hit a stale egg-link build.
#
#  Per-run flow (unchanged from run_rl_batch.sh):
#    1. patches scenario BasicSimScene.yaml so the simulated laser matches
#       training (360°, 72 rays, 3 m). Backed up to BasicSimScene_old.yaml.
#    2. launches GADEN env (method:=none) at recommended initial_iteration.
#    3. launches gaden_rl_node_lidar with num_episodes:=1 so it exits cleanly.
#    4. harvests node.log → summary.txt (steps, final distance, status).
#    5. kills every ROS process before the next run.
#
#  Usage:
#    tmux new-session -d -s rl 'bash ~/ros2_ws/run_rl_lidar_batch.sh'
#    tmux attach -t rl
#
#  Overrides:
#    RL_NUM_RUNS      trials per scenario (default 1)
#    RL_WALL_TIMEOUT  wall-clock cap per run in seconds (default 1000)
#    RL_MAX_STEPS     agent step cap (default 600 — matches cfg.MAX_STEPS)
#    RL_SPEED         GADEN playback speed multiplier (default 5.0)
#    RL_CHECKPOINT    path to .pt (REQUIRED — no sane default)
#    RL_ARCH          mlp | modular | dual | spatial (default mlp)
#    RL_DEVICE        cpu | cuda (default cpu)
#    RL_STEP_DELAY    seconds between policy steps (default 0.5)
#    RL_TARGETS       space-separated "scenario::sim" overrides
#    RL_DEBUG_DUMP    "1" to enable per-step PNG dumps (spatial arch only)
#    RL_DEBUG_EVERY   step interval for dumps (default 10)
# ============================================================

NUM_RUNS="${RL_NUM_RUNS:-1}"
WALL_TIMEOUT="${RL_WALL_TIMEOUT:-1000}"
MAX_STEPS="${RL_MAX_STEPS:-600}"
SPEED="${RL_SPEED:-5.0}"
CHECKPOINT="${RL_CHECKPOINT:-}"
ARCH="${RL_ARCH:-mlp}"
DEVICE="${RL_DEVICE:-cpu}"
STEP_DELAY="${RL_STEP_DELAY:-0.5}"
DEBUG_DUMP="${RL_DEBUG_DUMP:-0}"
DEBUG_EVERY="${RL_DEBUG_EVERY:-10}"
LIDAR_FRAME="${RL_LIDAR_FRAME:-world}"
# Motion mode: false = teleport between steps (default); true = drive each step
# via Nav2 (stop-go). Driving needs the sim's Nav2 stack (already in the launch)
# and is best run at SPEED=1.0 so the controller isn't starved.
USE_NAV2="${RL_USE_NAV2:-false}"
NAV_GOAL_TOL="${RL_NAV_GOAL_TOL:-0.1}"
DRIVE_TIMEOUT="${RL_DRIVE_TIMEOUT:-8.0}"   # per-step drive timeout (s); 0=off
PACKAGE="test_env"
LAUNCH_FILE="main_simbot_launch.py"

# --- Optional RViz video recording (RL_RECORD=1) ---------------------
# Records the live RViz window (top-down map + gas plume + robot) to
# <run_dir>/capture.mp4 via a headless Xvfb display + the imageio-ffmpeg
# bundled binary (x11grab). Off by default so the normal sweep is untouched.
RECORD="${RL_RECORD:-0}"
REC_DISPLAY="${RL_DISPLAY:-:99}"
REC_RES="${RL_REC_RES:-1600x900}"
REC_FPS="${RL_REC_FPS:-10}"
# RViz TopDownOrtho camera angle (rad). Positive rotates the view; tuned so the
# top-down map sits upright + ~5 deg clockwise per request. Override via RL_REC_ANGLE.
# SLAM-map videos look best perfectly upright, so default to 0 there.
if [[ "${RL_SLAM_VIZ:-0}" == "1" ]]; then REC_ANGLE="${RL_REC_ANGLE:-0}"; else REC_ANGLE="${RL_REC_ANGLE:-0.0423}"; fi
RVIZ_ARG=False
if [[ "$RECORD" == "1" ]]; then
    REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"   # defined here too; the canonical def is later but the RViz paths below need it now
    RVIZ_ARG=True
    FFMPEG=$(python3 -c 'import imageio_ffmpeg,sys;sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null)
    # RL_SLAM_VIZ=1 shows the robot's online SLAM map (/rlaika/slam_map) instead of the
    # ground-truth map (needs RL_ESCAPE=1 so the node builds+publishes it). The launch reads
    # GADEN_RVIZ_CONFIG, so we point it at gaden_slam.rviz and frame the camera into that file.
    RVIZ_NAME=gaden.rviz
    [[ "${RL_SLAM_VIZ:-0}" == "1" ]] && RVIZ_NAME=gaden_slam.rviz
    RVIZ_REAL=$(readlink -f "$REPO_ROOT/install/test_env/share/test_env/launch/$RVIZ_NAME")
    export GADEN_RVIZ_CONFIG="$RVIZ_REAL"
    [[ "${RL_SLAM_VIZ:-0}" == "1" ]] && echo "RL_SLAM_VIZ=1: RViz will load $RVIZ_NAME (SLAM map)."
    export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe DISPLAY="$REC_DISPLAY"
    pkill -9 Xvfb 2>/dev/null; sleep 1
    Xvfb "$REC_DISPLAY" -screen 0 "${REC_RES}x24" >/tmp/xvfb_rec.log 2>&1 &
    XVFB_PID=$!; sleep 3
    echo "RECORD mode: Xvfb on $REC_DISPLAY (${REC_RES}), ffmpeg=$FFMPEG, rviz=$RVIZ_REAL"
fi

# Auto-fit the RViz TopDownOrtho view to a scenario's map (center + scale) so each
# recorded map is framed correctly. Writes X/Y/Scale/Angle into the live gaden.rviz.
set_rviz_view() {
    [[ "$RECORD" == "1" ]] || return 0
    local scenario="$1"
    local occ="$INSTALL_SCENARIOS_DIR/$scenario/environment_configurations/config1/occupancy.yaml"
    [ -f "$occ" ] || { echo "  (set_rviz_view: no occupancy.yaml for $scenario)"; return 0; }
    local rviz="${RVIZ_REAL:-$(readlink -f "$REPO_ROOT/install/test_env/share/test_env/launch/gaden.rviz")}"
    python3 - "$occ" "$rviz" "$REC_ANGLE" <<'PY'
import sys, os, re, yaml, imageio.v2 as iio
occ_path, rviz_path, angle = sys.argv[1], sys.argv[2], sys.argv[3]
o = yaml.safe_load(open(occ_path)); res=float(o['resolution'])
ox, oy = float(o['origin'][0]), float(o['origin'][1])
img = iio.imread(os.path.join(os.path.dirname(occ_path), o['image']))
H, W = img.shape[:2]
Wm, Hm = W*res, H*res
cx, cy = ox + Wm/2.0, oy + Hm/2.0
scale = min(916.0/Wm, 786.0/Hm) * 0.9   # fit the 916x786 crop with margin
t = open(rviz_path).read()
t = re.sub(r'^      Angle: .*$', f'      Angle: {angle}', t, count=1, flags=re.M)
t = re.sub(r'^      Scale: .*$', f'      Scale: {scale:.2f}', t, count=1, flags=re.M)
t = re.sub(r'^      X: .*$', f'      X: {cx:.2f}', t, count=1, flags=re.M)
t = re.sub(r'^      Y: .*$', f'      Y: {cy:.2f}', t, count=1, flags=re.M)
open(rviz_path, 'w').write(t)
print(f'  rviz view [{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(occ_path))))}]: '
      f'center=({cx:.1f},{cy:.1f}) scale={scale:.1f} angle={angle}')
PY
}

# Wind-observation flags (read by obs_builder/wind_model via the environment).
# Local-wind policies (e.g. champ_far02, trained LOCAL_WIND_OBS=1) MUST run with
# OSL_LOCAL_WIND_OBS=1, else obs_builder feeds a frozen episode-mean polar wind
# instead of the per-step live local Cartesian point-wind the policy expects.
# OSL_DEPLOY_ANEMO/GADEN_ANEMO_FRAME pin the (-u,+v) anemometer-frame flip used in
# training. OSL_WIND_ZSNAP stays off (z=0.5 reads real wind on these maps).
# Override per-run via RL_LOCAL_WIND_OBS / RL_DEPLOY_ANEMO / RL_WIND_ZSNAP.
export OSL_LOCAL_WIND_OBS="${RL_LOCAL_WIND_OBS:-1}"
export OSL_DEPLOY_ANEMO="${RL_DEPLOY_ANEMO:-1}"
export GADEN_ANEMO_FRAME="${RL_DEPLOY_ANEMO:-1}"
export OSL_WIND_ZSNAP="${RL_WIND_ZSNAP:-0}"

# SLAM-based circling-escape (RL_ESCAPE=1 to enable; default off = baseline).
# Builds an online occupancy map from LiDAR and, when stuck circling, drives to
# the largest unexplored frontier. Tunables match escape_planner.CirclingEscape.
export OSL_ESCAPE="${RL_ESCAPE:-0}"
export OSL_ESCAPE_STREAK="${RL_ESCAPE_STREAK:-35}"     # efficiency-streak (circling)
export OSL_ESCAPE_WIN="${RL_ESCAPE_WIN:-25}"
export OSL_ESCAPE_RATIO="${RL_ESCAPE_RATIO:-0.2}"
export OSL_ESCAPE_COV_STREAK="${RL_ESCAPE_COV_STREAK:-25}"  # coverage-stagnation (loitering)
export OSL_ESCAPE_COV_WIN="${RL_ESCAPE_COV_WIN:-40}"
export OSL_ESCAPE_COV_K="${RL_ESCAPE_COV_K:-80}"   # min NEW SLAM cells / cov_win to count as exploring
export OSL_ESCAPE_COV_RES="${RL_ESCAPE_COV_RES:-0.5}"
export OSL_ESCAPE_COOLDOWN="${RL_ESCAPE_COOLDOWN:-40}"
export OSL_ESCAPE_MINDIST="${RL_ESCAPE_MINDIST:-3.0}"
export OSL_ESCAPE_TARGET="${RL_ESCAPE_TARGET:-largest}"   # largest | nearest

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCENARIOS_DIR="$REPO_ROOT/src/gaden/test_env/scenarios"
INSTALL_SCENARIOS_DIR="$REPO_ROOT/install/test_env/share/test_env/scenarios"
RECOMMENDED_CONFIGS="$REPO_ROOT/src/base/gaden_maps/recommended_configs.yaml"

if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi
if [ -f "$REPO_ROOT/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/install/setup.bash"
fi

BOLD='\033[1m'; DIM='\033[2m'; CYAN='\033[36m'; GREEN='\033[32m'
YELLOW='\033[33m'; RED='\033[31m'; RESET='\033[0m'

if [ -z "$CHECKPOINT" ]; then
    echo -e "${RED}RL_CHECKPOINT not set. Example:${RESET}"
    echo "  RL_CHECKPOINT=/abs/path/agent_X.pt RL_ARCH=dual bash $0"
    exit 1
fi

RUN_NAME="rl_lidar_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$REPO_ROOT/Results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

LOG_FILE="$RESULTS_DIR/batch.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ---- Targets --------------------------------------------------------
TARGETS_DEFAULT=(
    "curved_labrinth_left::sim1"
    "curved_labrinth_right::sim1"
    "10x6_u_left::sim1"
    "10x6_u_right::sim1"
    "4_rooms::sim1"
    "many_rooms::sim1"
    "ultimate::sim1"
)
if [[ -n "${RL_TARGETS:-}" ]]; then
    read -r -a TARGETS <<< "$RL_TARGETS"
else
    TARGETS=("${TARGETS_DEFAULT[@]}")
fi

TOTAL_TARGETS=${#TARGETS[@]}
TOTAL_RUNS=$(( TOTAL_TARGETS * NUM_RUNS ))
GLOBAL_RUN=0
TIMEOUTS=()
FAILURES=()
SUCCESSES=()

# ---- Helpers --------------------------------------------------------
rec_value() {
    local scenario="$1" sim="$2" field="$3"
    python3 -c "
import yaml
d = yaml.safe_load(open('$RECOMMENDED_CONFIGS'))
v = d.get('$scenario', {}).get('$sim', {}).get('$field')
if isinstance(v, list): print(','.join(str(x) for x in v))
elif v is not None: print(v)
" 2>/dev/null
}

save_delta_time() {
    local scenario="$1" sim="$2"
    local sim_yaml="$SCENARIOS_DIR/$scenario/environment_configurations/config1/simulations/$sim/sim.yaml"
    python3 -c "
import yaml
d = yaml.safe_load(open('$sim_yaml'))
print(d.get('saveDeltaTime', 0.5))
" 2>/dev/null
}

# Patch BasicSimScene.yaml so the simulated laser matches training. Idempotent.
patch_lidar_config() {
    local scenario="$1"
    local bases=(
        "$SCENARIOS_DIR/$scenario/environment_configurations/config1"
        "$INSTALL_SCENARIOS_DIR/$scenario/environment_configurations/config1"
    )
    for base in "${bases[@]}"; do
        local scene="$base/BasicSimScene.yaml"
        local backup="$base/BasicSimScene_old.yaml"
        [ -f "$scene" ] || continue
        if grep -q "maxAngleRad:[[:space:]]*6\.2832" "$scene" 2>/dev/null; then
            continue
        fi
        [ -f "$backup" ] || cp "$scene" "$backup"
        python3 - "$scene" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
subs = [
    (r"minAngleRad:\s*\S+",          "minAngleRad: 0.0"),
    (r"maxAngleRad:\s*\S+",          "maxAngleRad: 6.2832  # 2π — full 360°"),
    (r"angleResolutionRad:\s*\S+",   "angleResolutionRad: 0.08727  # 2π / 72"),
    (r"maxDistance:\s*\S+",          "maxDistance: 3.0  # matches LIDAR_MAX_RANGE"),
]
for pat, sub in subs:
    text = re.sub(pat, sub, text, count=1)
p.write_text(text)
PY
    done
}

# Locate the wind CSV a scenario is using.
wind_file_of() {
    local scenario="$1"
    local wind_root="$SCENARIOS_DIR/$scenario/wind_simulations"
    [ -d "$wind_root" ] || return 1
    local nested="$wind_root/1ms/wind_at_cell_centers_0.csv"
    if [ -f "$nested" ]; then
        echo "$nested"; return
    fi
    local first_nested
    first_nested=$(find "$wind_root" -mindepth 2 -name "wind_at_cell_centers_0.csv" 2>/dev/null | sort | head -1)
    if [ -n "$first_nested" ]; then
        echo "$first_nested"; return
    fi
    if [ -f "$wind_root/wind_0.csv" ]; then
        echo "$wind_root/wind_0.csv"; return
    fi
    return 1
}

kill_ros() {
    pkill -9 -f gaden_rl_node        2>/dev/null
    pkill -9 -f "ros2 launch"        2>/dev/null
    pkill -9 -f "ros2 run"           2>/dev/null
    pkill -9 -f rviz2                2>/dev/null
    pkill -9 -f gaden_player         2>/dev/null
    pkill -9 -f filament_simulator   2>/dev/null
    pkill -9 -f gaden_environment    2>/dev/null
    pkill -9 -f "gaden_"             2>/dev/null  # gaden_player/_environment/_rl_node — NOT scripts named *gaden*
    pkill -9 -f basic_sim            2>/dev/null
    pkill -9 -f simulated_gas_sensor 2>/dev/null
    pkill -9 -f simulated_anemometer 2>/dev/null
    pkill -9 -f simulated_tdlas      2>/dev/null
    pkill -9 -f fake_pid             2>/dev/null
    pkill -9 -f fake_anemometer      2>/dev/null
    pkill -9 -f bt_navigator         2>/dev/null
    pkill -9 -f planner_server       2>/dev/null
    pkill -9 -f controller_server    2>/dev/null
    pkill -9 -f behavior_server      2>/dev/null
    pkill -9 -f map_server           2>/dev/null
    pkill -9 -f lifecycle_manager    2>/dev/null
    pkill -9 -f nav2                 2>/dev/null
    pkill -9 -f static_transform_publisher 2>/dev/null
    pkill -9 -f robot_state_publisher 2>/dev/null
    pkill -9 -f joint_state_broadcaster 2>/dev/null
    sleep 2
    rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
    rm -f /tmp/fastrtps_* 2>/dev/null
    sleep 3
}

# Pre-flight
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Checkpoint not found: $CHECKPOINT${RESET}"
    exit 1
fi

echo ""
echo -e "${BOLD}${CYAN} ╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN} ║   gaden_rl_node_lidar Batch Runner       ║${RESET}"
echo -e "${BOLD}${CYAN} ╚══════════════════════════════════════════╝${RESET}"
echo -e "  Checkpoint  : ${BOLD}$(basename "$CHECKPOINT")${RESET}"
echo -e "  Architecture: ${BOLD}$ARCH${RESET}"
echo -e "  Lidar frame : ${BOLD}$LIDAR_FRAME${RESET}"
echo -e "  Runs/sim    : ${BOLD}$NUM_RUNS${RESET}"
echo -e "  Max steps   : ${BOLD}$MAX_STEPS${RESET}"
echo -e "  Wall t/o    : ${BOLD}${WALL_TIMEOUT}s${RESET}"
echo -e "  Speed       : ${BOLD}${SPEED}x${RESET}"
echo -e "  Total runs  : ${BOLD}$TOTAL_RUNS${RESET}"
echo -e "  Output      : ${BOLD}$RESULTS_DIR${RESET}"
echo ""

kill_ros

for TARGET in "${TARGETS[@]}"; do
    SCENARIO="${TARGET%%::*}"
    SIM="${TARGET##*::}"

    echo ""
    echo -e "${DIM} ══════════════════════════════════════${RESET}"
    echo -e "${BOLD} Scenario: ${CYAN}$SCENARIO/$SIM${RESET}"

    patch_lidar_config "$SCENARIO"
    echo -e "  ${DIM}lidar patched to 360°/72-ray/3m${RESET}"
    set_rviz_view "$SCENARIO"

    WIND_FILE=$(wind_file_of "$SCENARIO")
    if [ -z "$WIND_FILE" ]; then
        echo -e "  ${RED}No wind CSV found — skipping${RESET}"
        FAILURES+=("$SCENARIO/$SIM — no wind file")
        continue
    fi
    echo -e "  ${DIM}wind: ${WIND_FILE#$REPO_ROOT/}${RESET}"

    SRC_POS=$(rec_value "$SCENARIO" "$SIM" "source")
    START_POS=$(rec_value "$SCENARIO" "$SIM" "robot_start")
    REC_START_TIME=$(rec_value "$SCENARIO" "$SIM" "start_time")
    if [ -z "$SRC_POS" ] || [ -z "$START_POS" ] \
       || [ "$START_POS" = "None" ] || [ "$SRC_POS" = "None" ]; then
        echo -e "  ${RED}Missing source/start in recommended_configs — skipping${RESET}"
        FAILURES+=("$SCENARIO/$SIM — missing recommended config")
        continue
    fi
    SRC_X="${SRC_POS%%,*}";   SRC_Y="${SRC_POS##*,}"
    START_X="${START_POS%%,*}"; START_Y="${START_POS##*,}"

    SAVE_DT=$(save_delta_time "$SCENARIO" "$SIM")
    INITIAL_ITERATION=0
    if [[ -n "$REC_START_TIME" ]] && [[ -n "$SAVE_DT" ]]; then
        INITIAL_ITERATION=$(python3 -c "print(int(float('$REC_START_TIME') / float('$SAVE_DT')))")
    fi

    SIM_RESULTS_DIR="$RESULTS_DIR/${SCENARIO}_${SIM}"
    mkdir -p "$SIM_RESULTS_DIR"

    echo -e "  Source      : ${DIM}($SRC_X, $SRC_Y)${RESET}"
    echo -e "  Robot start : ${DIM}($START_X, $START_Y)${RESET}"
    echo -e "  Start time  : ${DIM}${REC_START_TIME:-0}s (iter $INITIAL_ITERATION)${RESET}"

    SCENES_DIR="$SCENARIOS_DIR/$SCENARIO/environment_configurations/config1/scenes"
    AUTO_SCENE="$SCENES_DIR/$SIM.yaml"
    if [[ ! -f "$AUTO_SCENE" ]]; then
        mkdir -p "$SCENES_DIR"
        cat > "$AUTO_SCENE" <<EOF
playback_initial_iteration: 0
playback_loop:
  loop: false
  from: 0
  to: 0
simulations:
  - sim: $SIM
    gas_color: [0.29, 1.0, 0.0]
EOF
    fi

    for (( run=1; run<=NUM_RUNS; run++ )); do
        (( GLOBAL_RUN++ ))

        echo ""
        echo -e "${BOLD} [$GLOBAL_RUN/$TOTAL_RUNS]  ${SCENARIO}/${SIM}  run ${run}/${NUM_RUNS}${RESET}  ${DIM}$(date '+%H:%M:%S')${RESET}"
        RUN_DIR="$SIM_RESULTS_DIR/run_${run}"
        mkdir -p "$RUN_DIR"

        run_start=$(date +%s)

        ros2 launch "$PACKAGE" "$LAUNCH_FILE" \
            scenario:="$SCENARIO" \
            playback:="$SIM" \
            method:=none \
            speed:="$SPEED" \
            initial_iteration:="$INITIAL_ITERATION" \
            headless:=True \
            use_rviz:=$RVIZ_ARG \
            > "$RUN_DIR/env.log" 2>&1 &
        env_pid=$!

        echo "  Waiting for /PioneerP3DX/ground_truth..."
        deadline=$(( $(date +%s) + 120 ))
        until timeout 2 ros2 topic echo /PioneerP3DX/ground_truth --once 2>/dev/null \
              | head -1 | grep -q header; do
            if [[ $(date +%s) -gt $deadline ]]; then
                echo -e "  ${RED}env never published ground_truth — aborting run${RESET}"
                break
            fi
            sleep 2
        done
        echo "  pose online, starting agent."
        sleep 3

        CAP_PID=""
        if [[ "$RECORD" == "1" ]]; then
            sleep 6   # let RViz draw the map + initial plume before grabbing
            "$FFMPEG" -hide_banner -loglevel error -y -f x11grab -draw_mouse 0 \
                -framerate "$REC_FPS" -video_size "$REC_RES" -i "$REC_DISPLAY" \
                -pix_fmt yuv420p -c:v libx264 -preset ultrafast \
                "$RUN_DIR/capture.mp4" >/tmp/ffmpeg_rec.log 2>&1 &
            CAP_PID=$!
            echo "  recording RViz -> $RUN_DIR/capture.mp4 (cap pid $CAP_PID)"
        fi

        DUMP_ARGS=()
        if [ "$DEBUG_DUMP" = "1" ]; then
            DUMP_ARGS=(-p debug_dump_dir:="$RUN_DIR/dumps" -p debug_dump_every:="$DEBUG_EVERY")
        fi
        ros2 run gaden_transfer gaden_rl_node_lidar --ros-args \
            -p checkpoint:="$CHECKPOINT" \
            -p arch:="$ARCH" \
            -p device:="$DEVICE" \
            -p wind_file:="$WIND_FILE" \
            -p true_source_x:="$SRC_X" \
            -p true_source_y:="$SRC_Y" \
            -p start_x:="$START_X" \
            -p start_y:="$START_Y" \
            -p max_steps:="$MAX_STEPS" \
            -p num_episodes:=1 \
            -p step_delay:="$STEP_DELAY" \
            -p lidar_frame:="$LIDAR_FRAME" \
            -p use_nav2:="$USE_NAV2" \
            -p nav_goal_tolerance:="$NAV_GOAL_TOL" \
            -p drive_timeout:="$DRIVE_TIMEOUT" \
            "${DUMP_ARGS[@]}" \
            > "$RUN_DIR/node.log" 2>&1 &
        node_pid=$!

        status="done"
        elapsed=0
        while kill -0 "$node_pid" 2>/dev/null; do
            elapsed=$(( $(date +%s) - run_start ))
            if (( elapsed > WALL_TIMEOUT )); then
                status="timeout"
                echo -e "  ${YELLOW}TIMEOUT${RESET} after ${elapsed}s — killing."
                kill -TERM "$node_pid" 2>/dev/null
                sleep 2
                kill -KILL "$node_pid" 2>/dev/null
                break
            fi
            sleep 2
        done
        wait "$node_pid" 2>/dev/null
        node_rc=$?

        if [[ "$RECORD" == "1" && -n "$CAP_PID" ]]; then
            sleep 2
            kill -INT "$CAP_PID" 2>/dev/null; sleep 2; kill -9 "$CAP_PID" 2>/dev/null
            cap_sz=$(stat -c%s "$RUN_DIR/capture.mp4" 2>/dev/null || echo 0)
            echo "  capture stopped: ${cap_sz} bytes -> $RUN_DIR/capture.mp4"
        fi

        python3 - \
            "$RUN_DIR/node.log" "$RUN_DIR/summary.txt" \
            "$SCENARIO" "$SIM" "$SRC_X" "$SRC_Y" "$START_X" "$START_Y" \
            "$status" "$elapsed" <<'PY'
import re, sys
log_path, out_path, scen, sim, sx, sy, rx, ry, status, elapsed = sys.argv[1:11]
steps = final_dist = None
success = False
timed_out_internally = False
try:
    for line in open(log_path):
        m = re.search(r"Source found at step (\d+)!\s+Distance:\s+([\d.]+)\s*m", line)
        if m:
            steps = int(m.group(1)); final_dist = float(m.group(2)); success = True
            continue
        m2 = re.search(r"Max steps\s*\((\d+)\)\s*reached", line)
        if m2:
            steps = int(m2.group(1)); timed_out_internally = True
except FileNotFoundError:
    pass

if success:
    result = "success"
elif status == "timeout":
    result = "wall_timeout"
elif timed_out_internally:
    result = "max_steps"
else:
    result = "crashed_or_unknown"

with open(out_path, "w") as f:
    f.write(f"scenario: {scen}/{sim}\n")
    f.write(f"status: {result}\n")
    f.write(f"steps: {steps if steps is not None else 'n/a'}\n")
    f.write(f"final_distance: {final_dist if final_dist is not None else 'n/a'}\n")
    f.write(f"wall_time_s: {elapsed}\n")
    f.write(f"source: ({sx}, {sy})\n")
    f.write(f"start:  ({rx}, {ry})\n")
PY

        if [[ "$status" == "timeout" ]]; then
            TIMEOUTS+=("${SCENARIO}/${SIM} run ${run}")
        fi
        if grep -q "status: success" "$RUN_DIR/summary.txt" 2>/dev/null; then
            steps_line=$(grep "^steps" "$RUN_DIR/summary.txt")
            dist_line=$(grep "^final_distance" "$RUN_DIR/summary.txt")
            echo -e "  ${GREEN}SUCCESS${RESET}  $steps_line  $dist_line  (${elapsed}s wall)"
            SUCCESSES+=("${SCENARIO}/${SIM} run ${run}")
        else
            st=$(grep "^status" "$RUN_DIR/summary.txt" 2>/dev/null)
            echo -e "  ${YELLOW}not solved${RESET}  $st"
        fi

        kill_ros

        if [[ $run -lt $NUM_RUNS ]] || [[ $GLOBAL_RUN -lt $TOTAL_RUNS ]]; then
            sleep 3
        fi
    done
done

# ============================================================
if [[ "$RECORD" == "1" && -n "${XVFB_PID:-}" ]]; then
    kill -9 "$XVFB_PID" 2>/dev/null
fi

echo ""
echo -e "${BOLD}${GREEN} ╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN} ║   gaden_rl_node_lidar batch complete     ║${RESET}"
echo -e "${BOLD}${GREEN} ╚══════════════════════════════════════════╝${RESET}"
echo -e "  Results   : ${BOLD}$RESULTS_DIR${RESET}"
echo -e "  Total     : $TOTAL_RUNS runs"
echo -e "  Successes : ${#SUCCESSES[@]}"
echo -e "  Timeouts  : ${#TIMEOUTS[@]}"
echo -e "  Failures  : ${#FAILURES[@]}"

if [[ ${#TIMEOUTS[@]} -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}Timed out:${RESET}"
    for t in "${TIMEOUTS[@]}"; do echo "  - $t"; done
fi
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed:${RESET}"
    for f in "${FAILURES[@]}"; do echo "  - $f"; done
fi

AGG="$RESULTS_DIR/aggregate.txt"
{
    echo "================= RL lidar batch aggregate ================="
    echo "Date       : $(date)"
    echo "Checkpoint : $CHECKPOINT"
    echo "Arch       : $ARCH"
    echo ""
    for d in "$RESULTS_DIR"/*/run_*; do
        [ -d "$d" ] || continue
        [ -f "$d/summary.txt" ] || continue
        echo "=== $(basename "$(dirname "$d")")/$(basename "$d") ==="
        cat "$d/summary.txt"
        echo ""
    done
} > "$AGG"
echo ""
echo -e "Aggregate : ${BOLD}$AGG${RESET}"
