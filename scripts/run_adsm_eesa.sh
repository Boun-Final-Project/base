#!/bin/bash

# ADSM / EESA Algorithm Launcher
# Interactive menu for launching ADSM or EESA with the correct source position
# for each GADEN map.
#
# Usage: ./run_adsm_eesa.sh
# Or add an alias: alias run_adsm_eesa='~/ros2_ws/src/base/scripts/run_adsm_eesa.sh'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPS_DIR="$SCRIPT_DIR/../gaden_maps"

if [ ! -d "$MAPS_DIR" ]; then
    echo "ERROR: gaden_maps directory not found: $MAPS_DIR"
    exit 1
fi

# ── Step 1: Choose algorithm ─────────────────────────────────────────────────
echo "======================================"
echo "  ADSM / EESA Launcher"
echo "======================================"
echo ""
echo "  1) adsm   - Adaptive Duty Sampling Method"
echo "  2) eesa   - Exploration-Enhanced Search Algorithm"
echo ""
read -p "Select algorithm [1/2]: " algo_choice

case "$algo_choice" in
    1) ALGORITHM="adsm"; LAUNCH_FILE="adsm_launch.py" ;;
    2) ALGORITHM="eesa"; LAUNCH_FILE="eesa_launch.py" ;;
    *)
        echo "Invalid selection!"
        exit 1
        ;;
esac

# ── Step 2: Discover maps and simulations ────────────────────────────────────
echo ""
declare -a menu_labels
declare -a menu_source_x
declare -a menu_source_y

i=1
while IFS= read -r yaml_file; do
    position_line=$(grep 'position:' "$yaml_file" | head -1)
    if [ -z "$position_line" ]; then
        continue
    fi

    src_x=$(echo "$position_line" | sed 's/.*\[//;s/\].*//' | cut -d',' -f1 | tr -d ' ')
    src_y=$(echo "$position_line" | sed 's/.*\[//;s/\].*//' | cut -d',' -f2 | tr -d ' ')

    # Ensure values have a decimal point (ROS 2 needs double, not integer)
    [[ "$src_x" != *.* ]] && src_x="${src_x}.0"
    [[ "$src_y" != *.* ]] && src_y="${src_y}.0"

    rel_path="${yaml_file#$MAPS_DIR/}"
    map_name=$(echo "$rel_path" | cut -d'/' -f1)
    sim_name=$(basename "$(dirname "$yaml_file")")

    menu_labels[$i]="$map_name / $sim_name"
    menu_source_x[$i]="$src_x"
    menu_source_y[$i]="$src_y"

    printf "  %2d) %-40s source=(%s, %s)\n" "$i" "$map_name / $sim_name" "$src_x" "$src_y"
    ((i++))
done < <(find "$MAPS_DIR" -name "sim.yaml" -path "*/simulations/*" | sort)

if [ "$i" -eq 1 ]; then
    echo "No simulations found in $MAPS_DIR"
    exit 1
fi

echo ""
read -p "Select map [1-$((i-1))]: " map_choice

if ! [[ "$map_choice" =~ ^[0-9]+$ ]] || [ "$map_choice" -lt 1 ] || [ "$map_choice" -ge "$i" ]; then
    echo "Invalid selection!"
    exit 1
fi

SRC_X="${menu_source_x[$map_choice]}"
SRC_Y="${menu_source_y[$map_choice]}"
SELECTED="${menu_labels[$map_choice]}"

# ── Step 3: Optional extra launch arguments ──────────────────────────────────
echo ""
read -p "Extra launch arguments (or press Enter for none): " extra_args

# ── Step 4: Launch ───────────────────────────────────────────────────────────
CMD="ros2 launch $ALGORITHM $LAUNCH_FILE source_x:=$SRC_X source_y:=$SRC_Y $extra_args"

echo ""
echo "======================================"
echo "  Algorithm : $ALGORITHM"
echo "  Map       : $SELECTED"
echo "  Source    : ($SRC_X, $SRC_Y)"
echo "======================================"
echo ""
echo "Command: $CMD"
echo ""

exec $CMD
