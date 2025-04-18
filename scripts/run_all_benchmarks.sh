#!/bin/bash
# Script to run benchmarks for all device-app pairs

# Define devices and applications
DEVICES=("3A021JEHN02756" "9b034f1b")
# DEVICES=("3A021JEHN02756" "9b034f1b" "jetson" "jetsonlowpower")
# APPS=("cifar-sparse" "cifar-dense" "tree")
APPS=("cifar-sparse")


# Number of repetitions for each benchmark
REPEAT=3

echo "=== Starting benchmark collection for all device-app pairs ==="
echo "Number of repetitions: $REPEAT"

# Run benchmarks for each device-app pair
for device in "${DEVICES[@]}"; do
  for app in "${APPS[@]}"; do
    echo ""
    echo "====================================================="
    echo "Running benchmark for $device - $app"
    echo "====================================================="
    
    python3 scripts/runner.py benchmark --device "$device" --app "$app" --repeat "$REPEAT"
    
    # Generate heatmaps right after collecting data
    # Use default excluded stages for now, modify as needed
    python3 scripts/runner.py heatmap --app "$app" --exclude-stages "2,4,8,9"
    
    # Optional: wait a bit between benchmarks to let system cool down
    echo "Waiting 10 seconds before next benchmark..."
    sleep 10
  done
done

echo ""
echo "=== Benchmark collection completed ==="
echo "Data is stored in the data directory with today's date" 