#!/bin/bash

# GADEN Preprocessing + Simulation Launcher
# Interactive menu for running ROS2 GADEN preprocessing and simulation
#
# Usage: ./presim.sh
# Or add an alias: alias presim='~/ros2_ws/src/base/scripts/presim.sh'

# ── Colors ───────────────────────────────────────────────────────────────────────
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
MAGENTA='\033[35m'
RED='\033[31m'
WHITE='\033[97m'
RESET='\033[0m'

# Auto-detect workspace root (two levels up from this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SCENARIOS_DIR="$WS_ROOT/src/gaden/test_env/scenarios"

if [ ! -d "$SCENARIOS_DIR" ]; then
    echo -e "${BOLD}ERROR:${RESET} Scenarios directory not found: $SCENARIOS_DIR"
    exit 1
fi

# ── Discovery ────────────────────────────────────────────────────────────────────
declare -A SIMS
while IFS= read -r yaml_file; do
    scenario=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f1)
    config=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f3)
    sim_name=$(basename "$(dirname "$yaml_file")")
    SIMS["$scenario::$config::$sim_name"]="$yaml_file"
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
        config=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f3)
        scene_name=$(basename "$yaml_file" .yaml)
        SCENES["$scenario::$config::$scene_name"]="$yaml_file"
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
echo -e "${BOLD}${CYAN} ║   GADEN Preprocessing + Simulation  ║${RESET}"
echo -e "${BOLD}${CYAN} ╚══════════════════════════════════════╝${RESET}"
echo ""
echo -e "${BOLD} Scenarios${RESET}"
echo -e "${DIM} ─────────────────────────────────────${RESET}"

for idx in "${!sorted_scenarios[@]}"; do
    num=$((idx + 1))
    s="${sorted_scenarios[$idx]}"
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

# ── Step 2: Select Sim/Scene ─────────────────────────────────────────────────────
declare -a sub_items
declare -a sub_types
declare -a sub_configs
j=0

for key in $(echo "${!SIMS[@]}" | tr ' ' '\n' | sort); do
    [[ "$key" != "$scenario::"* ]] && continue
    config=$(echo "$key" | cut -d':' -f3)
    sim=$(echo "$key" | cut -d':' -f5)
    sub_items[$j]="$sim"
    sub_types[$j]="sim"
    sub_configs[$j]="$config"
    ((j++))
done

for key in $(echo "${!SCENES[@]}" | tr ' ' '\n' | sort); do
    [[ "$key" != "$scenario::"* ]] && continue
    config=$(echo "$key" | cut -d':' -f3)
    scene=$(echo "$key" | cut -d':' -f5)
    sub_items[$j]="$scene"
    sub_types[$j]="scene"
    sub_configs[$j]="$config"
    ((j++))
done

if [ "$j" -eq 1 ]; then
    selected_id="${sub_items[0]}"
    selected_type="${sub_types[0]}"
    config="${sub_configs[0]}"
    echo -e "\n  ${DIM}Auto-selected:${RESET} $selected_id"
else
    echo ""
    echo -e "${BOLD} ${scenario}${RESET}"
    echo -e "${DIM} ─────────────────────────────────────${RESET}"
    for idx in "${!sub_items[@]}"; do
        num=$((idx + 1))
        config_label=""
        [[ "${sub_configs[$idx]}" != "config1" ]] && config_label=" ${DIM}[${sub_configs[$idx]}]${RESET}"
        if [ "${sub_types[$idx]}" == "scene" ]; then
            printf "  ${WHITE}%2d${RESET})  %-20s ${MAGENTA}[SCENE]${RESET}%s\n" "$num" "${sub_items[$idx]}" "$config_label"
        else
            printf "  ${WHITE}%2d${RESET})  %s%s\n" "$num" "${sub_items[$idx]}" "$config_label"
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
    config="${sub_configs[$((choice2 - 1))]}"
fi

# ── Resolve which sims to preprocess ──────────────────────────────────────────────
declare -a sim_list

if [ "$selected_type" == "scene" ]; then
    scene_yaml="$SCENARIOS_DIR/$scenario/environment_configurations/$config/scenes/$selected_id.yaml"
    while IFS= read -r sim_id; do
        sim_list+=("$sim_id")
    done < <(python3 -c "
import yaml
d = yaml.safe_load(open('$scene_yaml'))
for s in d.get('simulations', []):
    print(s['sim'])
" 2>/dev/null)
    if [ ${#sim_list[@]} -eq 0 ]; then
        echo -e "${BOLD}ERROR:${RESET} No simulations found in scene: $scene_yaml"
        exit 1
    fi
else
    sim_list=("$selected_id")
fi

# ── Run preprocessing + simulation ───────────────────────────────────────────────
total=${#sim_list[@]}
current=0

echo ""
echo -e "${BOLD}${GREEN} ╔══════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN} ║          Starting GADEN              ║${RESET}"
echo -e "${BOLD}${GREEN} ╚══════════════════════════════════════╝${RESET}"
echo -e "  Scenario  ${BOLD}$scenario${RESET}"
echo -e "  Config    ${BOLD}$config${RESET}"
echo -e "  Target    ${BOLD}${selected_id}${RESET}  ${DIM}($selected_type, $total sim(s))${RESET}"
echo ""

for sim in "${sim_list[@]}"; do
    ((current++))
    echo -e "${DIM} ─────────────────────────────────────${RESET}"
    echo -e "${BOLD} [$current/$total] ${CYAN}$sim${RESET}"
    echo -e "${DIM} ─────────────────────────────────────${RESET}"

    # Step 1: Preprocessing
    echo -e "\n  ${YELLOW}[1/2]${RESET} Preprocessing..."
    ros2 launch test_env gaden_preproc_launch.py scenario:="$scenario" configuration:="$config" simulation:="$sim"
    preproc_status=$?

    if [ $preproc_status -ne 0 ]; then
        echo -e "\n  ${RED}${BOLD}ERROR:${RESET} Preprocessing failed for $sim (exit $preproc_status)"
        exit $preproc_status
    fi

    # Step 2: Simulation
    echo -e "\n  ${YELLOW}[2/2]${RESET} Simulating..."
    ros2 launch test_env gaden_sim_launch.py scenario:="$scenario" configuration:="$config" simulation:="$sim"
    sim_status=$?

    if [ $sim_status -ne 0 ]; then
        echo -e "\n  ${RED}${BOLD}ERROR:${RESET} Simulation failed for $sim (exit $sim_status)"
        exit $sim_status
    fi

    echo -e "  ${GREEN}Done${RESET}"
done

echo ""
echo -e "${BOLD}${GREEN} ╔══════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN} ║       Completed successfully!        ║${RESET}"
echo -e "${BOLD}${GREEN} ╚══════════════════════════════════════╝${RESET}"
echo ""
