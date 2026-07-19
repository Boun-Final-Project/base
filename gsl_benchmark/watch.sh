#!/usr/bin/env bash
# gsl_benchmark — watch ONE scenario live in RViz (realistic/SLAM mode).
#
# Opens an RViz window (over VNC display :99) showing the SLAM map, the gas plume,
# the robot, and the goal it is driving to, with GADEN's world frame aligned to the
# SLAM map frame so everything lines up.
#
# Usage:
#   ./watch.sh <scenario> [agent=gpulaika] [extra gsl_bench eval flags...]
#
# Examples:
#   ./watch.sh 4_rooms_start_a
#   ./watch.sh ultimate_1 gpulaika
#   ./watch.sh 10x6_u_left_1 surge_cast --max-steps 400
#
# View it from your Mac by tunnelling port 5900, then open  vnc://localhost:5900
# (see REMOTE_VIEWING_README.md). This runs a SINGLE run — for batch stats use
# run_benchmark.sh (headless).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../.." && pwd)"

SCENARIO="${1:?usage: watch.sh <scenario> [agent] [extra flags...]}"
AGENT="${2:-gpulaika}"
shift || true; shift 2>/dev/null || true

ORACLE="$HERE/oracle_budgets_nav2_reliable.json"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export DISPLAY=":99"

# Bring up the virtual display + VNC server if they aren't already running.
if ! pgrep -f "Xvfb :99" >/dev/null 2>&1; then
  echo "starting Xvfb :99"
  Xvfb :99 -screen 0 1600x1000x24 +extension GLX +render >/tmp/gslb_watch_xvfb.log 2>&1 &
  sleep 2
fi
if ! pgrep -f "x11vnc.*:99" >/dev/null 2>&1; then
  echo "starting x11vnc on :5900"
  x11vnc -display :99 -rfbport 5900 -forever -shared -nopw -bg >/tmp/gslb_watch_x11vnc.log 2>&1 || true
  sleep 1
fi
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe

AGENT_CONFIG="$WS/src/gsl_bench/configs/agents/${AGENT}.yaml"
CONFIG_FLAG=()
[[ -f "$AGENT_CONFIG" ]] && CONFIG_FLAG=(--agent-config "$AGENT_CONFIG")

# shellcheck disable=SC1091
set +u; source "$WS/install/setup.bash"; set -u   # setup.bash uses unbound vars

echo "watch: scenario=$SCENARIO  agent=$AGENT  (realistic/SLAM, 1 run)"
echo "  view: tunnel 5900 -> open vnc://localhost:5900"

exec ros2 run gsl_bench eval \
  --agent "$AGENT" \
  "${CONFIG_FLAG[@]}" \
  --scenario "$SCENARIO" \
  --runs 1 \
  --realistic --visual \
  --oracle-budgets "$ORACLE" \
  --budget-multiplier 10 \
  --time-budget-multiplier 20 \
  --success-budget-multiplier 5 \
  --success-radius 0.5 \
  "$@"
