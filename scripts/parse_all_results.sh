#!/bin/bash
# Script to parse execution results for all device-app pairs

# Define devices and applications
DEVICES=("3A021JEHN02756" "9b034f1b" "jetson" "jetsonlowpower")
APPS=("cifar-sparse" "cifar-dense" "tree")

echo "=== Starting results parsing for all device-app pairs ==="

# Parse results for each device-app pair
for device in "${DEVICES[@]}"; do
  for app in "${APPS[@]}"; do
    echo ""
    echo "====================================================="
    echo "Parsing results for $device - $app"
    echo "====================================================="
    
    # Check if we have logs for this device-app pair
    date_dir=$(ls -t "$PWD/data" | head -n 1)
    results_dir="$PWD/data/$date_dir/results/${device}_${app}"
    
    if [ -d "$results_dir" ] && [ "$(ls -A "$results_dir")" ]; then
      python3 scripts/runner.py parse --device "$device" --app "$app"
      echo "Results parsing completed for $device - $app"
    else
      echo "No execution logs found for $device - $app, skipping"
    fi
  done
done

echo ""
echo "=== Results parsing completed ==="
echo "Analysis results are stored in the data/YYYY-MM-DD/results/device_app/analysis directories" 