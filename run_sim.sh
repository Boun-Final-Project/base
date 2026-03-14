#!/bin/bash

# GADEN Simulation Launcher
# Interactive menu for running ROS2 GADEN simulations

SCENARIOS_DIR="/home/efe/ros2_ws/src/gaden/test_env/scenarios"
INSTALL_SCENARIOS_DIR="/home/efe/ros2_ws/install/test_env/share/test_env/scenarios"

# ── Discovery ────────────────────────────────────────────────────────────────────
declare -A SIMS
while IFS= read -r yaml_file; do
    scenario=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f1)
    sim_name=$(basename "$(dirname "$yaml_file")")
    SIMS["$scenario::$sim_name"]="$yaml_file"
done < <(find "$SCENARIOS_DIR" -name "*.yaml" -path "*/simulations/*" | sort)

declare -A SCENES
while IFS= read -r yaml_file; do
    scenario=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f1)
    scene_name=$(basename "$yaml_file" .yaml)
    SCENES["$scenario::$scene_name"]="$yaml_file"
done < <(find "$SCENARIOS_DIR" -name "*.yaml" -path "*/scenes/*" | sort)

# ── 1. Selection ──────────────────────────────────────────────────────────────────
echo "======================================"
echo "  GADEN Simulation Launcher"
echo "======================================"
echo ""

i=1
declare -a menu_items
declare -a menu_types   # "scene" or "sim"

# Scenes first
for key in $(echo "${!SCENES[@]}" | tr ' ' '\n' | sort); do
    scenario=$(echo "$key" | cut -d':' -f1)
    scene=$(echo "$key" | cut -d':' -f3)
    menu_items[$i]="$scenario::$scene"
    menu_types[$i]="scene"
    printf "%2d) %-32s [SCENE] %s\n" "$i" "$scenario" "$scene"
    ((i++))
done

echo ""

# Individual sims second
for key in $(echo "${!SIMS[@]}" | tr ' ' '\n' | sort); do
    scenario=$(echo "$key" | cut -d':' -f1)
    sim=$(echo "$key" | cut -d':' -f3)
    menu_items[$i]="$scenario::$sim"
    menu_types[$i]="sim"
    printf "%2d) %-32s [SIM]   %s\n" "$i" "$scenario" "$sim"
    ((i++))
done

echo ""
read -p "Select (or 'q' to quit): " choice

if [[ "$choice" == "q" ]] || [[ "$choice" == "Q" ]]; then
    echo "Exiting..."
    exit 0
fi

if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -ge "$i" ]; then
    echo "Invalid selection!"
    exit 1
fi

selected="${menu_items[$choice]}"
selected_type="${menu_types[$choice]}"
scenario=$(echo "$selected" | cut -d':' -f1)
selected_id=$(echo "$selected" | cut -d':' -f3)

# ── Resolve playback_id & primary sim ─────────────────────────────────────────────
if [ "$selected_type" == "scene" ]; then
    playback_id="$selected_id"
    scene_yaml="$SCENARIOS_DIR/$scenario/environment_configurations/config1/scenes/$selected_id.yaml"
    # Get the first sim listed in the scene for iteration-count lookup
    primary_sim=$(python3 -c "
import yaml, sys
d = yaml.safe_load(open('$scene_yaml'))
sims = d.get('simulations', [])
print(sims[0]['sim'] if sims else 'sim1')
" 2>/dev/null)
    primary_sim="${primary_sim:-sim1}"
else
    # Individual sim: auto-create a single-sim scene file so gaden_player can load it
    scenes_dir="$SCENARIOS_DIR/$scenario/environment_configurations/config1/scenes"
    auto_scene="$scenes_dir/$selected_id.yaml"
    if [ ! -f "$auto_scene" ]; then
        mkdir -p "$scenes_dir"
        cat > "$auto_scene" << SCENE_EOF
playback_initial_iteration: 0
playback_loop:
  loop: false
  from: 0
  to: 0
simulations:
  - sim: $selected_id
    gas_color: [0.29, 1.0, 0.0]
SCENE_EOF
        echo "Auto-created scene file: $auto_scene"
    fi
    playback_id="$selected_id"
    primary_sim="$selected_id"
fi

# ── 2. Agent Selection ───────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "  Select Agent"
echo "======================================"
echo "  1) None (environment only)"
echo "  2) RRT-Infotaxis basic (no wind)"
echo "  3) RRT-Infotaxis advanced (with wind)"
echo "  4) PMFS"
echo ""
read -p "Select agent [1]: " agent_choice
agent_choice="${agent_choice:-1}"

case $agent_choice in
    1) method="none" ;;
    2) method="efe_igdm" ;;
    3) method="efe_igdm_wind" ;;
    4) method="PMFS" ;;
    *) echo "Invalid selection!"; exit 1 ;;
esac

# ── 3. Speed Selection ───────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "  Simulation Speed"
echo "======================================"
echo "  1) 1x   (realtime)"
echo "  2) 3x"
echo "  3) 5x   (default)"
echo "  4) 10x  (fast)"
echo ""
read -p "Select speed [3]: " speed_choice
speed_choice="${speed_choice:-3}"

case $speed_choice in
    1) speed=1.0 ;;
    2) speed=3.0 ;;
    3) speed=5.0 ;;
    4) speed=10.0 ;;
    *) speed=5.0 ;;
esac

# ── 4. Start Time ────────────────────────────────────────────────────────────────
gas_result_dir="$INSTALL_SCENARIOS_DIR/$scenario/environment_configurations/config1/simulations/$primary_sim/result"
max_iteration=0
if [ -d "$gas_result_dir" ]; then
    max_iteration=$(ls "$gas_result_dir" | grep -oP '(?<=iteration_)\d+' | sort -n | tail -1)
    max_iteration=${max_iteration:-0}
fi

sim_yaml="$SCENARIOS_DIR/$scenario/environment_configurations/config1/simulations/$primary_sim/sim.yaml"
save_delta_time=0.5
if [ -f "$sim_yaml" ]; then
    parsed=$(grep -oP 'saveDeltaTime:\s*\K[\d.]+' "$sim_yaml" | head -1)
    if [ -n "$parsed" ]; then
        save_delta_time="$parsed"
    fi
fi

max_time_s=$(awk "BEGIN { printf \"%.1f\", $max_iteration * $save_delta_time }")

echo ""
echo "Scenario: $scenario  |  Playback: $playback_id ($selected_type)"
if [ "$max_iteration" -gt 0 ]; then
    echo "Gas data:  $max_iteration iterations available  (~${max_time_s}s at ${save_delta_time}s/iter)"
fi
echo ""
read -p "Start time in seconds [0]: " start_time_input
start_time_input="${start_time_input:-0}"

initial_iteration=$(awk "BEGIN { printf \"%d\", $start_time_input / $save_delta_time }")

if [ "$max_iteration" -gt 0 ] && [ "$initial_iteration" -gt "$max_iteration" ]; then
    echo "Warning: requested iteration $initial_iteration exceeds max $max_iteration — clamping."
    initial_iteration=$max_iteration
fi

# ── 5. Launch ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "  Launching"
echo "======================================"
echo "  Scenario:  $scenario"
echo "  Playback:  $playback_id  ($selected_type)"
echo "  Agent:     $method"
echo "  Speed:     ${speed}x"
echo "  Start:     iteration $initial_iteration"
echo "======================================"
echo ""

ros2 launch test_env main_simbot_launch.py \
    scenario:="$scenario" \
    playback:="$playback_id" \
    method:="$method" \
    speed:="$speed" \
    initial_iteration:="$initial_iteration"
