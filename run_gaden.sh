#!/bin/bash

# GADEN Preprocessing + Simulation Launcher
# Interactive menu for running ROS2 GADEN preprocessing and simulation

SCENARIOS_DIR="/home/efe/ros2_ws/src/gaden/test_env/scenarios"

# Find all scenarios and simulations
declare -A SIMS
while IFS= read -r yaml_file; do
    scenario=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f1)
    config=$(echo "$yaml_file" | sed "s|$SCENARIOS_DIR/||" | cut -d'/' -f3)
    sim_name=$(basename "$(dirname "$yaml_file")")
    SIMS["$scenario::$config::$sim_name"]="$yaml_file"
done < <(find "$SCENARIOS_DIR" -name "*.yaml" -path "*/simulations/*" | sort)

# Display menu
echo "======================================"
echo "  GADEN Preprocessing + Simulation"
echo "======================================"
echo ""

i=1
declare -a menu_items
for key in "${!SIMS[@]}"; do
    scenario=$(echo "$key" | cut -d':' -f1)
    config=$(echo "$key" | cut -d':' -f3)
    sim=$(echo "$key" | cut -d':' -f5)
    menu_items[$i]="$scenario::$config::$sim"
    if [[ "$config" == "config1" ]]; then
        printf "%2d) %-30s %s\n" "$i" "$scenario" "$sim"
    else
        printf "%2d) %-30s %s  [%s]\n" "$i" "$scenario" "$sim" "$config"
    fi
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
config=$(echo "$selected" | cut -d':' -f3)
sim=$(echo "$selected" | cut -d':' -f5)

echo ""
echo "======================================"
echo "  Running GADEN for: $scenario / $config / $sim"
echo "======================================"
echo ""

# Step 1: Run preprocessing
echo "[1/2] Running GADEN preprocessing..."
echo "Command: ros2 launch test_env gaden_preproc_launch.py scenario:=$scenario configuration:=$config simulation:=$sim"
echo ""

ros2 launch test_env gaden_preproc_launch.py scenario:="$scenario" configuration:="$config" simulation:="$sim"
preproc_status=$?

if [ $preproc_status -ne 0 ]; then
    echo ""
    echo "ERROR: Preprocessing failed with exit code $preproc_status"
    exit $preproc_status
fi

echo ""
echo "======================================"
echo "[2/2] Running GADEN simulation..."
echo "Command: ros2 launch test_env gaden_sim_launch.py scenario:=$scenario configuration:=$config simulation:=$sim"
echo ""

ros2 launch test_env gaden_sim_launch.py scenario:="$scenario" configuration:="$config" simulation:="$sim"
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
