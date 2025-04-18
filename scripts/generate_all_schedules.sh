#!/bin/bash
# Script to generate schedules for all device-app pairs

# Define devices and applications
DEVICES=("3A021JEHN02756" "9b034f1b" "jetson" "jetsonlowpower")
APPS=("cifar-sparse" "cifar-dense" "tree")

# Number of schedules to generate
NUM_SCHEDULES=30

echo "=== Starting schedule generation for all device-app pairs ==="
echo "Number of schedules per pair: $NUM_SCHEDULES"

# Generate schedules for each device-app pair
for device in "${DEVICES[@]}"; do
  for app in "${APPS[@]}"; do
    echo ""
    echo "====================================================="
    echo "Generating schedules for $device - $app"
    echo "====================================================="
    
    python3 scripts/runner.py schedule --device "$device" --app "$app" --num-schedules "$NUM_SCHEDULES"
    
    echo "Schedule generation completed for $device - $app"
  done
done

echo ""
echo "=== Schedule generation completed ==="
echo "All schedules stored in the data/YYYY-MM-DD/schedules directory"
echo ""
echo "To start the HTTP server for schedule distribution:"
echo "python3 scripts/runner.py server" 