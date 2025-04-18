#!/bin/bash
# Script to generate schedules for all device-app pairs

# Define devices and applications
DEVICES=("3A021JEHN02756" "9b034f1b" "jetson" "jetsonlowpower")
APPS=("cifar-sparse" "cifar-dense" "tree")

# Number of schedules to generate
NUM_SCHEDULES=30

echo "=== Starting schedule generation for all device-app pairs ==="
echo "Number of schedules per pair: $NUM_SCHEDULES"

# Get the most recent date directory
date_dir=$(ls -t "$PWD/data" | head -n 1)

if [ -z "$date_dir" ]; then
  echo "Error: No data directory found"
  exit 1
fi

echo "Using data directory: data/$date_dir"

# Generate schedules for each device-app pair
for device in "${DEVICES[@]}"; do
  for app in "${APPS[@]}"; do
    echo ""
    echo "====================================================="
    echo "Generating schedules for $device - $app"
    echo "====================================================="
    
    # Check if we have benchmark data for this device-app pair
    csv_path="$PWD/data/$date_dir/$app/${device}_fully.csv"
    
    if [ -f "$csv_path" ]; then
      python3 scripts/runner.py schedule --device "$device" --app "$app" --num-schedules "$NUM_SCHEDULES"
      echo "Schedule generation completed for $device - $app"
    else
      echo "No benchmark data found for $device - $app in $csv_path, skipping"
    fi
  done
done

echo ""
echo "=== Schedule generation completed ==="
echo "All schedules stored in the data/$date_dir/schedules directory"
echo ""
echo "To start the HTTP server for schedule distribution:"
echo "python3 scripts/runner.py server" 