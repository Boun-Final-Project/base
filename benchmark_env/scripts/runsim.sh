#!/bin/bash
# runsim — launch a benchmark_env scenario (robot + gas playback + sensors).
# Menu: pick a scenario (and simulation), then runs main_simbot_launch.py.
#
# Robot start, source, and start-time all come from the scenario's config files
# (config.yaml empty_point, simulations/<sim>/sim.yaml, scenes/scene1.yaml) — this
# script does NOT pass any of them; it just selects the scenario and launches.
#
# Usage:  ./runsim.sh                        (interactive)
#         ./runsim.sh <scenario> [simulation]   (non-interactive)
# Alias:  alias runsim='~/ros2_ws/src/benchmark_env/scripts/runsim.sh'

set -o pipefail
BOLD='\033[1m'; DIM='\033[2m'; CYAN='\033[36m'; GREEN='\033[32m'; WHITE='\033[97m'; RED='\033[31m'; YELLOW='\033[33m'; RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # src/benchmark_env
SCENARIOS_DIR="$PKG_DIR/scenarios"

[ -d "$SCENARIOS_DIR" ] || { echo -e "${RED}${BOLD}ERROR:${RESET} scenarios dir not found: $SCENARIOS_DIR"; exit 1; }

mapfile -t SCENARIOS < <(find "$SCENARIOS_DIR" -maxdepth 2 -name gaden.gproj -printf '%h\n' | xargs -n1 basename | sort)
[ ${#SCENARIOS[@]} -gt 0 ] || { echo "No scenarios found in $SCENARIOS_DIR"; exit 1; }

# --- pick scenario ---
scenario="$1"
if [ -z "$scenario" ]; then
    echo -e "\n${BOLD}${CYAN} Run benchmark scenario${RESET}"
    echo -e "${BOLD} Scenarios${RESET}\n${DIM} ─────────────────────────────${RESET}"
    for i in "${!SCENARIOS[@]}"; do printf "  ${WHITE}%2d${RESET})  %s\n" "$((i+1))" "${SCENARIOS[$i]}"; done
    echo -ne "\n${BOLD} >>${RESET} Select scenario ${DIM}[q]${RESET}: "; read c
    [[ "$c" == q ]] && exit 0
    [[ "$c" =~ ^[0-9]+$ ]] && [ "$c" -ge 1 ] && [ "$c" -le ${#SCENARIOS[@]} ] || { echo "Invalid selection"; exit 1; }
    scenario="${SCENARIOS[$((c-1))]}"
fi
[ -d "$SCENARIOS_DIR/$scenario" ] || { echo -e "${RED}Unknown scenario:${RESET} $scenario"; exit 1; }

# --- discover simulations (for playback selection) ---
SIMS_DIR="$SCENARIOS_DIR/$scenario/environment_configurations/config1/simulations"
mapfile -t SIMS < <(find "$SIMS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort)

simulation="$2"
if [ -z "$simulation" ]; then
    if [ ${#SIMS[@]} -le 1 ]; then
        simulation="${SIMS[0]:-sim1}"
    else
        echo -e "\n${BOLD} ${scenario}${RESET}\n${DIM} ─────────────────────────────${RESET}"
        for i in "${!SIMS[@]}"; do printf "  ${WHITE}%2d${RESET})  %s\n" "$((i+1))" "${SIMS[$i]}"; done
        echo -ne "\n${BOLD} >>${RESET} Select simulation ${DIM}[q]${RESET}: "; read c
        [[ "$c" == q ]] && exit 0
        [[ "$c" =~ ^[0-9]+$ ]] && [ "$c" -ge 1 ] && [ "$c" -le ${#SIMS[@]} ] || { echo "Invalid selection"; exit 1; }
        simulation="${SIMS[$((c-1))]}"
    fi
fi

# --- bake gas first if it's missing OR stale (offer to run presim inline) ---
# The gas source position is frozen into the baked log files at bake time (the
# player reads it from there, NOT from sim.yaml), so editing the config has no
# effect on playback until a re-bake. Detect that drift by comparing config
# mtimes against the newest baked result file.
INSTALL_CFG_DIR="$(ros2 pkg prefix benchmark_env 2>/dev/null)/share/benchmark_env/scenarios/$scenario/environment_configurations/config1"
RESULT_DIR="$INSTALL_CFG_DIR/simulations/$simulation/result"
bake_reason=""
if [ ! -d "$RESULT_DIR" ] || [ -z "$(ls -A "$RESULT_DIR" 2>/dev/null)" ]; then
    bake_reason="No baked gas"
else
    newest_bake="$(find "$RESULT_DIR" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
    for cfg in "$INSTALL_CFG_DIR/simulations/$simulation/sim.yaml" "$INSTALL_CFG_DIR/config.yaml"; do
        [ -f "$cfg" ] || continue
        cfg_mtime="$(stat -c '%Y' "$cfg" 2>/dev/null)"
        if [ -n "$cfg_mtime" ] && [ -n "$newest_bake" ] \
           && awk "BEGIN{exit !($cfg_mtime > $newest_bake)}"; then
            bake_reason="Stale gas (config changed since last bake)"
            break
        fi
    done
fi
if [ -n "$bake_reason" ]; then
    echo -e "\n${YELLOW}${BOLD} $bake_reason${RESET} for $scenario/$simulation."
    echo -ne "${BOLD} >>${RESET} Preprocess + bake gas now? ${DIM}[Y/n]${RESET}: "; read bake_ans
    if [[ ! "$bake_ans" =~ ^[Nn]$ ]]; then
        echo -e "\n  ${YELLOW}[bake 1/2]${RESET} Preprocessing..."
        ros2 launch benchmark_env gaden_preproc_launch.py scenario:="$scenario" simulation:="$simulation" \
            || { echo -e "${RED}${BOLD}Preprocessing failed${RESET}"; exit 1; }
        echo -e "\n  ${YELLOW}[bake 2/2]${RESET} Baking gas ${DIM}(slow — let it finish)${RESET}..."
        ros2 launch benchmark_env gaden_sim_launch.py scenario:="$scenario" simulation:="$simulation" \
            || { echo -e "${RED}${BOLD}Gas bake failed${RESET}"; exit 1; }
    else
        echo -e "${DIM} continuing without gas — playback will be empty${RESET}"
    fi
fi

echo -e "\n${BOLD}${GREEN} Launching${RESET}  scenario=${BOLD}$scenario${RESET}  simulation=${BOLD}$simulation${RESET}\n"
ros2 launch benchmark_env main_simbot_launch.py scenario:="$scenario" simulation:="$simulation"
