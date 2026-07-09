#!/usr/bin/env bash
# killros — terminate all running ROS 2 processes cleanly, then forcefully.
# Targets ros2 CLI, launch, nodes, the daemon, DDS discovery, gzserver/gzclient,
# rviz and colcon-spawned processes tied to a ROS session.

set -u

# Patterns that identify ROS / simulation processes. Word-ish matches on the
# full command line so we don't accidentally hit unrelated binaries.
PATTERNS=(
  "ros2"
  "_ros2_daemon"
  "ros2 launch"
  "rclpy"
  "rclcpp"
  "robot_state_publisher"
  "rviz2"
  "gzserver"
  "gzclient"
  "gazebo"
  "ign gazebo"
  "gaden"
  "rmw_"
  "fastdds"
  "discovery"
)

collect_pids() {
  local pids=""
  for pat in "${PATTERNS[@]}"; do
    # -f full cmdline match; exclude this script and the grep itself.
    local found
    found=$(pgrep -f "$pat" 2>/dev/null)
    pids="$pids $found"
  done
  # De-dupe, drop our own PID and parent shell.
  echo "$pids" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u \
    | grep -vx "$$" | grep -vx "$PPID"
}

kill_with() {
  local sig="$1"; shift
  local pids="$*"
  [ -z "$pids" ] && return 0
  # shellcheck disable=SC2086
  kill "-$sig" $pids 2>/dev/null
}

main() {
  # Let the daemon try to shut down gracefully first.
  ros2 daemon stop >/dev/null 2>&1 || true

  local pids
  pids=$(collect_pids)

  if [ -z "$pids" ]; then
    echo "killros: no ROS processes found."
    return 0
  fi

  echo "killros: sending SIGTERM to:"
  ps -o pid=,cmd= -p $(echo "$pids" | tr '\n' ' ') 2>/dev/null | sed 's/^/  /'
  kill_with TERM $pids

  # Give them a moment to exit on their own.
  sleep 2

  # Anything still alive gets SIGKILL.
  local remaining
  remaining=$(collect_pids)
  if [ -n "$remaining" ]; then
    echo "killros: force-killing survivors:"
    ps -o pid=,cmd= -p $(echo "$remaining" | tr '\n' ' ') 2>/dev/null | sed 's/^/  /'
    kill_with KILL $remaining
  fi

  echo "killros: done."
}

main "$@"
