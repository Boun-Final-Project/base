#!/bin/bash

# GADEN Simulation Launcher
# Interactive menu for running ROS2 GADEN simulations (robot + environment)
#
# Usage: ./runsim.sh
# Or add an alias: alias runsim='~/ros2_ws/src/base/scripts/runsim.sh'

# Auto-detect workspace root (two levels up from this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SCENARIOS_DIR="$WS_ROOT/src/gaden/test_env/scenarios"
INSTALL_SCENARIOS_DIR="$WS_ROOT/install/pmfs_env/share/pmfs_env/scenarios"

if [ ! -d "$SCENARIOS_DIR" ]; then
    echo "ERROR: Scenarios directory not found: $SCENARIOS_DIR"
    echo "Make sure GADEN is installed in your workspace."
    exit 1
fi

# Find all scenarios and simulations
declare -A SIMS
while IFS= read -r yaml_file; do
    scenario=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f1)
    sim_name=$(basename "$(dirname "$yaml_file")")
    SIMS["$scenario::$sim_name"]="$yaml_file"
done < <(find "$SCENARIOS_DIR" -name "*.yaml" -path "*/simulations/*" | sort)

if [ ${#SIMS[@]} -eq 0 ]; then
    echo "No simulations found in $SCENARIOS_DIR"
    exit 1
fi

# Display menu
echo "======================================"
echo "  GADEN Simulation Launcher"
echo "======================================"
echo ""

i=1
declare -a menu_items
for key in "${!SIMS[@]}"; do
    scenario=$(echo "$key" | cut -d':' -f1)
    sim=$(echo "$key" | cut -d':' -f3)
    menu_items[$i]="$scenario::$sim"
    printf "%2d) %-30s %s\n" "$i" "$scenario" "$sim"
    ((i++))
done

echo ""
echo "======================================"
read -p "Select simulation number (or 'q' to quit): " choice

if [[ "$choice" == "q" ]] || [[ "$choice" == "Q" ]]; then
    echo "Exiting..."
    exit 0
fi

if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -ge "$i" ]; then
    echo "Invalid selection!"
    exit 1
fi

selected="${menu_items[$choice]}"
scenario=$(echo "$selected" | cut -d':' -f1)
sim=$(echo "$selected" | cut -d':' -f3)

# ── Infer max available iteration ──────────────────────────────────────────────
gas_result_dir="$INSTALL_SCENARIOS_DIR/$scenario/gas_simulations/$sim/result"
max_iteration=0
if [ -d "$gas_result_dir" ]; then
    max_iteration=$(ls "$gas_result_dir" | grep -oP '(?<=iteration_)\d+' | sort -n | tail -1)
    max_iteration=${max_iteration:-0}
fi

# ── Read saveDeltaTime from sim YAML ───────────────────────────────────────────
sim_yaml="$SCENARIOS_DIR/$scenario/environment_configurations/config1/simulations/$sim/sim.yaml"
save_delta_time=0.5  # default seconds per iteration
if [ -f "$sim_yaml" ]; then
    parsed=$(grep -oP 'saveDeltaTime:\s*\K[\d.]+' "$sim_yaml" | head -1)
    if [ -n "$parsed" ]; then
        save_delta_time="$parsed"
    fi
fi

# Compute max time in seconds (with integer arithmetic via awk)
max_time_s=$(awk "BEGIN { printf \"%.1f\", $max_iteration * $save_delta_time }")

# ── Start time prompt ───────────────────────────────────────────────────────────
echo ""
echo "Scenario: $scenario  |  Simulation: $sim"
if [ "$max_iteration" -gt 0 ]; then
    echo "Gas data:  $max_iteration iterations available  (~${max_time_s}s at ${save_delta_time}s/iter)"
fi
echo ""
read -p "Start time in seconds [0]: " start_time_input
start_time_input="${start_time_input:-0}"

# Convert seconds → iteration index
initial_iteration=$(awk "BEGIN { printf \"%d\", $start_time_input / $save_delta_time }")

# Clamp to max
if [ "$max_iteration" -gt 0 ] && [ "$initial_iteration" -gt "$max_iteration" ]; then
    echo "Warning: requested iteration $initial_iteration exceeds max $max_iteration — clamping."
    initial_iteration=$max_iteration
fi

# ── Launch ──────────────────────────────────────────────────────────────────────
echo ""
echo "Launching: $scenario / $sim  (start iteration: $initial_iteration)"
echo "Command: ros2 launch test_env main_simbot_launch.py scenario:=$scenario simulation:=$sim initial_iteration:=$initial_iteration"
echo ""

ros2 launch test_env main_simbot_launch.py scenario:="$scenario" simulation:="$sim" initial_iteration:="$initial_iteration"
