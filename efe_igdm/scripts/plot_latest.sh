#!/bin/bash
# Quick script to plot the most recent IGDM log

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR=~/igdm_logs

# Find latest log
LATEST_LOG=$(ls -t $LOG_DIR/igdm_log_*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log files found in $LOG_DIR"
    exit 1
fi

echo "Plotting latest log: $LATEST_LOG"

# Check for map image argument
if [ -n "$1" ]; then
    echo "Using map: $1"
    python3 "$SCRIPT_DIR/plot_search_trajectory.py" "$LATEST_LOG" --map "$1" --all
else
    echo "No map specified, plotting without background"
    python3 "$SCRIPT_DIR/plot_search_trajectory.py" "$LATEST_LOG" --all
fi

echo ""
echo "Plots saved to: $LOG_DIR/"
