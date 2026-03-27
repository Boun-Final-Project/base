#!/bin/bash

# GADEN Simulation Launcher
# Interactive menu for running ROS2 GADEN simulations (robot + environment)
#
# Usage: ./runsim.sh
# Or add an alias: alias runsim='~/ros2_ws/src/base/scripts/runsim.sh'

# ── Colors ───────────────────────────────────────────────────────────────────────
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
MAGENTA='\033[35m'
WHITE='\033[97m'
RESET='\033[0m'

# Auto-detect workspace root (two levels up from this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SCENARIOS_DIR="$WS_ROOT/src/gaden/test_env/scenarios"
INSTALL_SCENARIOS_DIR="$WS_ROOT/install/test_env/share/test_env/scenarios"

if [ ! -d "$SCENARIOS_DIR" ]; then
    echo -e "${BOLD}ERROR:${RESET} Scenarios directory not found: $SCENARIOS_DIR"
    exit 1
fi

# ── Discovery ────────────────────────────────────────────────────────────────────
declare -A SIMS
while IFS= read -r yaml_file; do
    scenario=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f1)
    sim_name=$(basename "$(dirname "$yaml_file")")
    SIMS["$scenario::$sim_name"]="$yaml_file"
done < <(find "$SCENARIOS_DIR" -name "*.yaml" -path "*/simulations/*" | sort)

declare -A SCENES
while IFS= read -r yaml_file; do
    sim_count=$(python3 -c "
import yaml
d = yaml.safe_load(open('$yaml_file'))
print(len(d.get('simulations', [])))
" 2>/dev/null)
    if [ "$sim_count" -gt 1 ]; then
        scenario=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f1)
        scene_name=$(basename "$yaml_file" .yaml)
        SCENES["$scenario::$scene_name"]="$yaml_file"
    fi
done < <(find "$SCENARIOS_DIR" -name "*.yaml" -path "*/scenes/*" | sort)

if [ ${#SIMS[@]} -eq 0 ]; then
    echo "No simulations found in $SCENARIOS_DIR"
    exit 1
fi

# ── Collect unique scenarios ─────────────────────────────────────────────────────
declare -A scenario_set
for key in "${!SIMS[@]}"; do
    scenario=$(echo "$key" | cut -d':' -f1)
    scenario_set["$scenario"]=1
done
IFS=$'\n' sorted_scenarios=($(printf '%s\n' "${!scenario_set[@]}" | sort)); unset IFS

# ── Step 1: Select Scenario ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN} ╔══════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN} ║     GADEN Simulation Launcher       ║${RESET}"
echo -e "${BOLD}${CYAN} ╚══════════════════════════════════════╝${RESET}"
echo ""
echo -e "${BOLD} Scenarios${RESET}"
echo -e "${DIM} ─────────────────────────────────────${RESET}"

for idx in "${!sorted_scenarios[@]}"; do
    num=$((idx + 1))
    s="${sorted_scenarios[$idx]}"
    # Count sims and scenes for this scenario
    sim_count=0
    scene_count=0
    for key in "${!SIMS[@]}"; do
        [[ "$key" == "$s::"* ]] && ((sim_count++))
    done
    for key in "${!SCENES[@]}"; do
        [[ "$key" == "$s::"* ]] && ((scene_count++))
    done
    detail="${sim_count} sim"
    [ "$sim_count" -gt 1 ] && detail="${sim_count} sims"
    [ "$scene_count" -gt 0 ] && detail="$detail, ${scene_count} scene"
    printf "  ${WHITE}%2d${RESET})  %-30s ${DIM}%s${RESET}\n" "$num" "$s" "$detail"
done

echo ""
echo -ne "${BOLD} >>${RESET} Select scenario ${DIM}[q to quit]${RESET}: "
read choice

[[ "$choice" == "q" || "$choice" == "Q" ]] && exit 0
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#sorted_scenarios[@]}" ]; then
    echo -e "${BOLD}Invalid selection!${RESET}"; exit 1
fi

scenario="${sorted_scenarios[$((choice - 1))]}"

# ── Step 2: Select Sim/Scene within scenario ─────────────────────────────────────
# Collect sims for this scenario
declare -a sub_items
declare -a sub_types
j=0

# Sims
for key in $(echo "${!SIMS[@]}" | tr ' ' '\n' | sort); do
    [[ "$key" != "$scenario::"* ]] && continue
    sim=$(echo "$key" | cut -d':' -f3)
    sub_items[$j]="$sim"
    sub_types[$j]="sim"
    ((j++))
done

# Scenes
for key in $(echo "${!SCENES[@]}" | tr ' ' '\n' | sort); do
    [[ "$key" != "$scenario::"* ]] && continue
    scene=$(echo "$key" | cut -d':' -f3)
    sub_items[$j]="$scene"
    sub_types[$j]="scene"
    ((j++))
done

if [ "$j" -eq 1 ]; then
    # Only one option, auto-select
    selected_id="${sub_items[0]}"
    selected_type="${sub_types[0]}"
    echo -e "\n  ${DIM}Auto-selected:${RESET} $selected_id"
else
    echo ""
    echo -e "${BOLD} ${scenario}${RESET}"
    echo -e "${DIM} ─────────────────────────────────────${RESET}"
    for idx in "${!sub_items[@]}"; do
        num=$((idx + 1))
        if [ "${sub_types[$idx]}" == "scene" ]; then
            printf "  ${WHITE}%2d${RESET})  %-20s ${MAGENTA}[SCENE]${RESET}\n" "$num" "${sub_items[$idx]}"
        else
            printf "  ${WHITE}%2d${RESET})  %s\n" "$num" "${sub_items[$idx]}"
        fi
    done

    echo ""
    echo -ne "${BOLD} >>${RESET} Select ${DIM}[q to quit]${RESET}: "
    read choice2

    [[ "$choice2" == "q" || "$choice2" == "Q" ]] && exit 0
    if ! [[ "$choice2" =~ ^[0-9]+$ ]] || [ "$choice2" -lt 1 ] || [ "$choice2" -gt "$j" ]; then
        echo -e "${BOLD}Invalid selection!${RESET}"; exit 1
    fi

    selected_id="${sub_items[$((choice2 - 1))]}"
    selected_type="${sub_types[$((choice2 - 1))]}"
fi

# ── Resolve playback_id & primary sim ─────────────────────────────────────────────
if [ "$selected_type" == "scene" ]; then
    playback_id="$selected_id"
    scene_yaml="$SCENARIOS_DIR/$scenario/environment_configurations/config1/scenes/$selected_id.yaml"
    primary_sim=$(python3 -c "
import yaml
d = yaml.safe_load(open('$scene_yaml'))
sims = d.get('simulations', [])
print(sims[0]['sim'] if sims else 'sim1')
" 2>/dev/null)
    primary_sim="${primary_sim:-sim1}"
else
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
        echo -e "  ${DIM}Auto-created scene file${RESET}"
    fi
    playback_id="$selected_id"
    primary_sim="$selected_id"
fi

# ── Recommended config ───────────────────────────────────────────────────────────
RECOMMENDED_CONFIGS="$SCRIPT_DIR/../gaden_maps/recommended_configs.yaml"
REC_ROBOT_START=""
REC_START_TIME=""
REC_NOTES=""

if [ -f "$RECOMMENDED_CONFIGS" ]; then
    REC_ROBOT_START=$(python3 -c "
import yaml
d = yaml.safe_load(open('$RECOMMENDED_CONFIGS'))
sc = d.get('$scenario', {}).get('$primary_sim', {})
robot = sc.get('robot_start')
print(f'{robot[0]},{robot[1]}' if robot else '')
" 2>/dev/null)
    REC_START_TIME=$(python3 -c "
import yaml
d = yaml.safe_load(open('$RECOMMENDED_CONFIGS'))
sc = d.get('$scenario', {}).get('$primary_sim', {})
t = sc.get('start_time')
print(t if t is not None else '')
" 2>/dev/null)
    REC_NOTES=$(python3 -c "
import yaml
d = yaml.safe_load(open('$RECOMMENDED_CONFIGS'))
sc = d.get('$scenario', {}).get('$primary_sim', {})
print(sc.get('notes', ''))
" 2>/dev/null)
fi

USE_REC_ROBOT_START=""
if [ -n "$REC_ROBOT_START" ] || [ -n "$REC_START_TIME" ] || [ -n "$REC_NOTES" ]; then
    echo ""
    echo -e "${BOLD} Recommended config${RESET}"
    echo -e "${DIM} ─────────────────────────────────────${RESET}"
    [ -n "$REC_ROBOT_START" ] && echo -e "  Robot start : ${CYAN}($REC_ROBOT_START)${RESET}"
    [ -n "$REC_START_TIME"  ] && echo -e "  Start time  : ${CYAN}${REC_START_TIME}s${RESET}"
    [ -n "$REC_NOTES"       ] && echo -e "  Notes       : ${DIM}${REC_NOTES}${RESET}"

    if [ -n "$REC_ROBOT_START" ]; then
        echo ""
        echo -ne "${BOLD} >>${RESET} Use recommended robot start position? ${DIM}[Y/n]${RESET}: "
        read use_rec
        use_rec="${use_rec:-Y}"
        if [[ "$use_rec" =~ ^[Yy]$ ]]; then
            USE_REC_ROBOT_START="$REC_ROBOT_START"
        fi
    fi
fi

# ── Step 3: Agent Selection ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD} Agent${RESET}"
echo -e "${DIM} ─────────────────────────────────────${RESET}"
echo -e "  ${WHITE} 1${RESET})  None ${DIM}(environment only)${RESET}"
echo -e "  ${WHITE} 2${RESET})  RRT-Infotaxis ${DIM}(no wind)${RESET}"
echo -e "  ${WHITE} 3${RESET})  RRT-Infotaxis ${DIM}(with wind)${RESET}"
echo -e "  ${WHITE} 4${RESET})  PMFS"
echo -e "  ${WHITE} 5${RESET})  IGDM Multiple ${DIM}(multi-source)${RESET}"
echo ""
echo -ne "${BOLD} >>${RESET} Select agent ${DIM}[1]${RESET}: "
read agent_choice
agent_choice="${agent_choice:-1}"

case $agent_choice in
    1) method="none" ;;
    2) method="efe_igdm" ;;
    3) method="efe_igdm_wind" ;;
    4) method="PMFS" ;;
    5) method="igdm_multiple" ;;
    *) echo -e "${BOLD}Invalid selection!${RESET}"; exit 1 ;;
esac

# ── Step 4: Speed Selection ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD} Speed${RESET}"
echo -e "${DIM} ─────────────────────────────────────${RESET}"
echo -e "  ${WHITE} 1${RESET})  1x   ${DIM}realtime${RESET}"
echo -e "  ${WHITE} 2${RESET})  3x"
echo -e "  ${WHITE} 3${RESET})  5x   ${DIM}default${RESET}"
echo -e "  ${WHITE} 4${RESET})  10x  ${DIM}fast${RESET}"
echo ""
echo -ne "${BOLD} >>${RESET} Select speed ${DIM}[3]${RESET}: "
read speed_choice
speed_choice="${speed_choice:-3}"

case $speed_choice in
    1) speed=1.0 ;;
    2) speed=3.0 ;;
    3) speed=5.0 ;;
    4) speed=10.0 ;;
    *) speed=5.0 ;;
esac

# ── Step 5: Start Time ──────────────────────────────────────────────────────────
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
echo -e "${BOLD} Start Time${RESET}"
echo -e "${DIM} ─────────────────────────────────────${RESET}"
if [ "$max_iteration" -gt 0 ]; then
    echo -e "  Available: ${GREEN}$max_iteration${RESET} iterations ${DIM}(~${max_time_s}s at ${save_delta_time}s/iter)${RESET}"
fi
default_start="${REC_START_TIME:-0}"
echo -ne "${BOLD} >>${RESET} Start time in seconds ${DIM}[${default_start}]${RESET}: "
read start_time_input
start_time_input="${start_time_input:-$default_start}"

initial_iteration=$(awk "BEGIN { printf \"%d\", $start_time_input / $save_delta_time }")

if [ "$max_iteration" -gt 0 ] && [ "$initial_iteration" -gt "$max_iteration" ]; then
    echo -e "  ${YELLOW}Clamped to max iteration $max_iteration${RESET}"
    initial_iteration=$max_iteration
fi

# ── Launch ───────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN} ╔══════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN} ║            Launching                 ║${RESET}"
echo -e "${BOLD}${GREEN} ╚══════════════════════════════════════╝${RESET}"
echo -e "  Scenario  ${BOLD}$scenario${RESET}"
echo -e "  Playback  ${BOLD}$playback_id${RESET}  ${DIM}($selected_type)${RESET}"
echo -e "  Agent     ${BOLD}$method${RESET}"
echo -e "  Speed     ${BOLD}${speed}x${RESET}"
echo -e "  Start     ${BOLD}iteration $initial_iteration${RESET}"
echo ""

LAUNCH_ARGS="scenario:=$scenario playback:=$playback_id method:=$method speed:=$speed initial_iteration:=$initial_iteration"
if [ -n "$USE_REC_ROBOT_START" ]; then
    rx=$(echo "$USE_REC_ROBOT_START" | cut -d',' -f1)
    ry=$(echo "$USE_REC_ROBOT_START" | cut -d',' -f2)
    LAUNCH_ARGS+=" robot_x:=$rx robot_y:=$ry"
fi

ros2 launch test_env main_simbot_launch.py $LAUNCH_ARGS
