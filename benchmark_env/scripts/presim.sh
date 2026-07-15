#!/bin/bash
# presim — GADEN preprocessing + gas bake for a benchmark_env scenario.
# Menu: pick a scenario (and simulation), then runs:
#   gaden_preproc_launch.py  ->  gaden_sim_launch.py
#
# Usage:  ./presim.sh            (interactive)
#         ./presim.sh <scenario> [simulation]   (non-interactive)
# Alias:  alias presim='~/ros2_ws/src/benchmark_env/scripts/presim.sh'

set -o pipefail
BOLD='\033[1m'; DIM='\033[2m'; CYAN='\033[36m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; WHITE='\033[97m'; RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # src/benchmark_env
SCENARIOS_DIR="$PKG_DIR/scenarios"

[ -d "$SCENARIOS_DIR" ] || { echo -e "${RED}${BOLD}ERROR:${RESET} scenarios dir not found: $SCENARIOS_DIR"; exit 1; }

# --- discover scenarios (dirs with a gaden.gproj) ---
mapfile -t SCENARIOS < <(find "$SCENARIOS_DIR" -maxdepth 2 -name gaden.gproj -printf '%h\n' | xargs -n1 basename | sort)
[ ${#SCENARIOS[@]} -gt 0 ] || { echo "No scenarios found in $SCENARIOS_DIR"; exit 1; }

# --- pick scenario (arg or menu) ---
scenario="$1"
if [ -z "$scenario" ]; then
    echo -e "\n${BOLD}${CYAN} GADEN preprocessing + gas bake${RESET}"
    echo -e "${BOLD} Scenarios${RESET}\n${DIM} ─────────────────────────────${RESET}"
    for i in "${!SCENARIOS[@]}"; do printf "  ${WHITE}%2d${RESET})  %s\n" "$((i+1))" "${SCENARIOS[$i]}"; done
    echo -ne "\n${BOLD} >>${RESET} Select scenario ${DIM}[q]${RESET}: "; read c
    [[ "$c" == q ]] && exit 0
    [[ "$c" =~ ^[0-9]+$ ]] && [ "$c" -ge 1 ] && [ "$c" -le ${#SCENARIOS[@]} ] || { echo "Invalid selection"; exit 1; }
    scenario="${SCENARIOS[$((c-1))]}"
fi
[ -d "$SCENARIOS_DIR/$scenario" ] || { echo -e "${RED}Unknown scenario:${RESET} $scenario"; exit 1; }

# --- discover simulations under config1 ---
SIMS_DIR="$SCENARIOS_DIR/$scenario/environment_configurations/config1/simulations"
mapfile -t SIMS < <(find "$SIMS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort)
[ ${#SIMS[@]} -gt 0 ] || { echo -e "${RED}No simulations in${RESET} $SIMS_DIR"; exit 1; }

# --- pick simulation (arg / auto / menu) ---
simulation="$2"
if [ -z "$simulation" ]; then
    if [ ${#SIMS[@]} -eq 1 ]; then
        simulation="${SIMS[0]}"; echo -e "  ${DIM}Auto-selected sim:${RESET} $simulation"
    else
        echo -e "\n${BOLD} ${scenario}${RESET}\n${DIM} ─────────────────────────────${RESET}"
        for i in "${!SIMS[@]}"; do printf "  ${WHITE}%2d${RESET})  %s\n" "$((i+1))" "${SIMS[$i]}"; done
        echo -ne "\n${BOLD} >>${RESET} Select simulation ${DIM}[q]${RESET}: "; read c
        [[ "$c" == q ]] && exit 0
        [[ "$c" =~ ^[0-9]+$ ]] && [ "$c" -ge 1 ] && [ "$c" -le ${#SIMS[@]} ] || { echo "Invalid selection"; exit 1; }
        simulation="${SIMS[$((c-1))]}"
    fi
fi

echo -e "\n${BOLD}${GREEN} Baking${RESET}  scenario=${BOLD}$scenario${RESET}  simulation=${BOLD}$simulation${RESET}\n"

echo -e "  ${YELLOW}[1/2]${RESET} Preprocessing..."
ros2 launch benchmark_env gaden_preproc_launch.py scenario:="$scenario" simulation:="$simulation" \
    || { echo -e "${RED}${BOLD}Preprocessing failed${RESET}"; exit 1; }

echo -e "\n  ${YELLOW}[2/2]${RESET} Simulating (gas bake)..."
ros2 launch benchmark_env gaden_sim_launch.py scenario:="$scenario" simulation:="$simulation" \
    || { echo -e "${RED}${BOLD}Simulation failed${RESET}"; exit 1; }

echo -e "\n${BOLD}${GREEN} Done.${RESET}  Run it with:  runsim $scenario\n"
