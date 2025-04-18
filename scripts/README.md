# Better Together - Benchmarking and Scheduling Framework

This directory contains scripts for benchmarking, scheduling, and analyzing applications running on heterogeneous computing systems.

## Directory Structure

The refactored framework organizes data as follows:

```
data/
├── YYYY-MM-DD/                # Date-based directory
│   ├── cifar-sparse/          # Application-specific data
│   │   ├── device1_normal.csv # Benchmark data
│   │   ├── device1_fully.csv  
│   │   ├── device1_*.png      # Generated heatmaps
│   │   ├── run_*.txt          # Benchmark logs
│   ├── cifar-dense/           # Another application
│   ├── tree/                  # Another application
│   ├── schedules/             # Directory for schedule JSON files
│   │   ├── device1_app1_vk_schedules.json
│   │   ├── device2_app1_vk_schedules.json
│   ├── results/               # Results from schedule execution
│       ├── device1_app1/      # Device and app specific results
│       │   ├── device1_app1_schedules_000.log
│       │   ├── device1_app1_schedules_001.log
│       │   ├── analysis/      # Analysis outputs
```

## The Runner Script

The `runner.py` script provides a unified interface for all benchmarking and scheduling operations. It's designed to maintain a consistent directory structure and simplify the workflow.

### Basic Usage

The script supports the following tasks:

1. **benchmark** - Run benchmarks and collect data
2. **heatmap** - Generate heatmaps from benchmark data
3. **schedule** - Generate optimized schedules using Z3
4. **run** - Execute schedules on target devices
5. **parse** - Parse and analyze schedule execution results
6. **server** - Start a HTTP server to serve schedule files
7. **pipeline** - Run multiple steps in sequence

### Examples

#### Running Individual Steps

```bash
# Collect benchmark data for cifar-sparse on device 3A021JEHN02756
python3 scripts/runner.py benchmark --device 3A021JEHN02756 --app cifar-sparse --repeat 3

# Generate heatmaps for cifar-sparse, excluding specific stages
python3 scripts/runner.py heatmap --app cifar-sparse --exclude-stages 2,4,8,9

# Generate schedules using Z3 optimizer
python3 scripts/runner.py schedule --device 3A021JEHN02756 --app cifar-sparse --num-schedules 30

# Start the HTTP server to serve schedule files
python3 scripts/runner.py server

# Run schedules on target device
python3 scripts/runner.py run --device 3A021JEHN02756 --app cifar-sparse --num-schedules 30

# Parse and analyze results
python3 scripts/runner.py parse --device 3A021JEHN02756 --app cifar-sparse
```

#### Running the Complete Pipeline

```bash
# Run all steps in sequence
python3 scripts/runner.py pipeline --device 3A021JEHN02756 --app cifar-sparse

# Run only specific steps in the pipeline
python3 scripts/runner.py pipeline --device 3A021JEHN02756 --app cifar-sparse --steps benchmark heatmap schedule
```

### Advanced Usage

You can customize various parameters:

- `--repeat` - Number of times to repeat benchmarks (default: 3)
- `--num-schedules` - Number of schedules to generate/run (default: 30)
- `--exclude-stages` - Stages to exclude from heatmaps (e.g. "2,4,8,9")

## Working With Multiple Devices and Applications

To run the same workload across multiple devices or applications, you can create a simple shell script:

```bash
#!/bin/bash
DEVICES=("3A021JEHN02756" "9b034f1b" "jetson" "jetsonlowpower")
APPS=("cifar-sparse" "cifar-dense" "tree")

# Example: Run benchmarks for all device-app pairs
for device in "${DEVICES[@]}"; do
  for app in "${APPS[@]}"; do
    echo "Processing $device - $app"
    python3 scripts/runner.py benchmark --device $device --app $app
  done
done
```

## Notes

- The script automatically creates necessary directories
- All data is organized by date, making it easy to track changes over time
- The server runs on port 8080 by default
- For the `run` step, the server must be running (the script will start it if needed) 