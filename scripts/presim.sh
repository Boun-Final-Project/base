#!/bin/bash

# GADEN Preprocessing + Simulation Launcher
# Interactive menu for running ROS2 GADEN preprocessing and simulation
#
# Usage: ./presim.sh
# Or add an alias: alias presim='~/ros2_ws/src/base/scripts/presim.sh'

# Auto-detect workspace root (two levels up from this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SCENARIOS_DIR="$WS_ROOT/src/gaden/test_env/scenarios"

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
echo "  GADEN Preprocessing + Simulation"
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

echo ""
echo "======================================"
echo "  Running GADEN for: $scenario / $sim"
echo "======================================"
echo ""

# Step 1: Run preprocessing
echo "[1/2] Running GADEN preprocessing..."
echo "Command: ros2 launch test_env gaden_preproc_launch.py scenario:=$scenario simulation:=$sim"
echo ""

ros2 launch test_env gaden_preproc_launch.py scenario:="$scenario" simulation:="$sim"
preproc_status=$?

if [ $preproc_status -ne 0 ]; then
    echo ""
    echo "ERROR: Preprocessing failed with exit code $preproc_status"
    exit $preproc_status
fi

echo ""
echo "======================================"
echo "[2/2] Running GADEN simulation..."
echo "Command: ros2 launch test_env gaden_sim_launch.py scenario:=$scenario simulation:=$sim"
echo ""

ros2 launch test_env gaden_sim_launch.py scenario:="$scenario" simulation:="$sim"
sim_status=$?

if [ $sim_status -ne 0 ]; then
    echo ""
    echo "ERROR: Simulation failed with exit code $sim_status"
    exit $sim_status
fi

echo ""
echo "======================================"
echo "  GADEN completed successfully!"
echo "======================================"
