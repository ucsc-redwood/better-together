#!/bin/bash
# Script to run benchmarks for all applications

# Define applications
APPS=("cifar-sparse" "cifar-dense" "tree")

# Number of repetitions for each benchmark
REPEAT=3

echo "=== Starting benchmark collection for all applications ==="
echo "Number of repetitions: $REPEAT"

# Run benchmarks for each application
for app in "${APPS[@]}"; do
  echo ""
  echo "====================================================="
  echo "Running benchmark for $app"
  echo "====================================================="
  
  python3 scripts/runner.py benchmark --app "$app" --repeat "$REPEAT"
  
  # Generate heatmaps right after collecting data
  # Use default excluded stages for now, modify as needed
  python3 scripts/runner.py heatmap --app "$app" --exclude-stages "2,4,8,9"
  
  # Optional: wait a bit between benchmarks to let system cool down
  echo "Waiting 10 seconds before next benchmark..."
  sleep 10
done

echo ""
echo "=== Benchmark collection completed ==="
echo "Data is stored in the data directory with today's date" 